"""Shared bootstrap utilities for volume 2 chapter agents.

This module is framework-agnostic and intended to be reused by LangChain,
LlamaIndex, and future chapter agent implementations.
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import json
import importlib.util
import logging
import os
import re
import select
import sys
import threading
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterable, Iterator, Protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from shared.llm_models import (
    ALL_MODEL_IDENTIFIERS,
    ModelConfig,
    PROVIDER_DISPLAY_NAMES,
    get_api_key_env_vars,
    get_identifier_mappings,
    resolve_llamaindex_model,
    route_model_for_prompt,
)
from shared.utils import build_task_prompt, get_chapter_logger, log_tool_call, parse_structured_json_response
from shared.web import chunk_text, normalize_response_text, to_sse_line


class AgentManagerProtocol(Protocol):
    framework: str
    model: str
    stream: bool

    def ask_question(self, topic: str) -> Dict[str, Any]:
        ...

def _iter_tool_names(manager: AgentManagerProtocol) -> Iterable[str]:
    names = getattr(manager, "tool_names", None)
    if not names:
        return []
    return [str(name) for name in names if str(name).strip()]


def _print_cli_banner(manager: AgentManagerProtocol) -> None:
    print(f"\n===== {manager.framework} CLI =====")
    print("Type a question and press Enter.")
    print("Type 'exit' to quit.\n")

    tool_names = list(_iter_tool_names(manager))
    if tool_names:
        print("Available local tools:")
        for name in tool_names:
            print(f"  - {name}")
    else:
        print("Available local tools: (none declared)")

    trigger_help = getattr(
        manager,
        "tool_trigger_help",
        "Tools are triggered automatically based on your prompt. You can mention a specific task (for example: calculate) to encourage tool use.",
    )
    print(f"\n{trigger_help}")
    print("Tip: multi-line pasted text is accepted as a single prompt.")
    print("=" * 36)


def _drain_pasted_lines(lines: list[str], max_wait_ms: int = 60) -> list[str]:
    """Collect additional lines already buffered (commonly from bracketed paste).

    This prevents one pasted multi-line prompt from being interpreted as several
    follow-up prompts after the first request returns.
    """
    if not sys.stdin.isatty():
        return lines

    deadline = max_wait_ms / 1000.0

    while True:
        try:
            ready, _, _ = select.select([sys.stdin], [], [], deadline)
        except (OSError, ValueError, AttributeError):
            # stdin may not be selectable (e.g., tests using StringIO)
            break

        if not ready:
            break

        next_line = sys.stdin.readline()
        if next_line == "":
            break

        lines.append(next_line.rstrip("\n"))
        # After first extra line, only keep draining immediately buffered data.
        deadline = 0.0

    return lines


def _read_cli_prompt() -> str | None:
    """Read one logical prompt from stdin.

    Returns:
      - None on EOF
      - a (possibly multi-line) prompt string otherwise
    """
    sys.stdout.write("\n(exit or enter question) > ")
    sys.stdout.flush()

    first_line = sys.stdin.readline()
    if first_line == "":
        return None

    lines = [first_line.rstrip("\n")]
    lines = _drain_pasted_lines(lines)
    return "\n".join(lines).strip()


def get_chapter_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_tool_call(logger: logging.Logger, tool_name: str, fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def wrapper(arg: Any = None) -> Any:
        logger.info("Tool call | name=%s | input=%s", tool_name, arg)
        result = fn(arg)
        logger.info("Tool result | name=%s | output=%s", tool_name, result)
        return result

    wrapper.__name__ = getattr(fn, "__name__", f"{tool_name}_wrapper")
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    return wrapper


def _extract_text_from_content(content: Any) -> str:
    """Extract readable text from common LangChain/LlamaIndex content shapes."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue

            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if isinstance(text, str):
                        parts.append(text)
                    continue

                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue

                delta = item.get("delta")
                if isinstance(delta, str):
                    parts.append(delta)

        return "".join(parts)

    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text

        delta = content.get("delta")
        if isinstance(delta, str):
            return delta

    return ""


def _extract_text_from_message_like(obj: Any) -> str:
    """Best-effort extraction from message/result objects across frameworks."""
    if obj is None:
        return ""

    if isinstance(obj, str):
        return obj

    direct = _extract_text_from_content(obj)
    if direct:
        return direct

    for attr_name in ("content", "text", "response", "message", "output", "delta"):
        value = getattr(obj, attr_name, None)
        text = _extract_text_from_content(value)
        if text:
            return text

        if value is not None and value is not obj:
            nested = _extract_text_from_message_like(value)
            if nested:
                return nested

    if isinstance(obj, dict):
        for key in ("output", "response", "content", "text", "delta"):
            text = _extract_text_from_content(obj.get(key))
            if text:
                return text

        messages = obj.get("messages") or []
        for message in reversed(messages):
            text = _extract_text_from_message_like(message)
            if text:
                return text

    return ""


def _looks_like_llamaindex_stream_handler(result: Any) -> bool:
    return hasattr(result, "stream_events") and callable(getattr(result, "stream_events"))


def _looks_like_langchain_stream_chunk(chunk: Any) -> bool:
    return isinstance(chunk, tuple) and len(chunk) == 2


def _append_stream_text(parts: list[str], text: str) -> None:
    """Append only new incremental content when stream events are cumulative."""
    if not text:
        return

    current = "".join(parts)
    if not current:
        parts.append(text)
        return

    # Some providers emit full accumulated text on every event. Keep only the suffix.
    if text.startswith(current):
        suffix = text[len(current):]
        if suffix:
            parts.append(suffix)
        return

    # Exact duplicate event.
    if text == current or text == parts[-1]:
        return

    parts.append(text)


def _next_stream_chunk(parts: list[str], text: str) -> str:
    """Return only the incremental suffix for cumulative stream events."""
    before = "".join(parts)
    _append_stream_text(parts, text)
    after = "".join(parts)
    if after.startswith(before):
        return after[len(before) :]
    return text if after != before else ""


async def _collect_llamaindex_stream_text(handler: Any) -> str:
    """
    Collect streamed text from a LlamaIndex workflow handler.

    Works with patterns like:
        handler = workflow.run(...)
        async for event in handler.stream_events():
            ...
    """
    return await _collect_stream_text_from_events(handler.stream_events())


async def _collect_stream_text_from_events(event_stream: Any) -> str:
    """Collect text from an async event stream with LLM delta/message payloads."""
    parts: list[str] = []

    async for event in event_stream:
        delta = getattr(event, "delta", None)
        if isinstance(delta, str) and delta:
            _append_stream_text(parts, delta)
            continue

        text = _extract_text_from_message_like(event)
        if text:
            _append_stream_text(parts, text)

    return "".join(parts).strip()


async def _collect_async_event_stream_text(event_stream: Any) -> str:
    """Collect text from an async iterator that yields event-like objects."""
    return await _collect_stream_text_from_events(event_stream)


async def iter_stream_text_chunks(raw_response: Any) -> AsyncIterator[str]:
    """Yield incremental text chunks from framework-specific stream responses."""
    if raw_response is None:
        return

    if isinstance(raw_response, str):
        for chunk in chunk_text(raw_response):
            if chunk:
                yield chunk
        return

    if hasattr(raw_response, "__aiter__"):
        parts: list[str] = []
        async for event in raw_response:
            delta = getattr(event, "delta", None)
            if isinstance(delta, str) and delta:
                chunk = _next_stream_chunk(parts, delta)
                if chunk:
                    yield chunk
                continue

            text = _extract_text_from_message_like(event)
            if text:
                chunk = _next_stream_chunk(parts, text)
                if chunk:
                    yield chunk
        return

    parts: list[str] = []
    for chunk in raw_response:
        if not _looks_like_langchain_stream_chunk(chunk):
            continue

        stream_mode, payload = chunk
        if stream_mode != "messages":
            continue

        message_chunk = payload[0] if isinstance(payload, tuple) and payload else payload
        if message_chunk is None:
            continue

        text = _extract_text_from_message_like(message_chunk)
        if text:
            delta = _next_stream_chunk(parts, text)
            if delta:
                yield delta


def _collect_langchain_stream_text(result: Any) -> str:
    """
    Collect streamed text from LangChain/LangGraph stream output.

    Common shapes:
      ('messages', (message_chunk, metadata))
      ('messages', (message_chunk,))
      ('messages', message_chunk)
    """
    parts: list[str] = []

    for chunk in result:
        if not _looks_like_langchain_stream_chunk(chunk):
            continue

        stream_mode, payload = chunk
        if stream_mode != "messages":
            continue

        message_chunk = None

        if isinstance(payload, tuple):
            if len(payload) >= 1:
                message_chunk = payload[0]
        else:
            message_chunk = payload

        if message_chunk is None:
            continue

        text = _extract_text_from_message_like(message_chunk)
        if text:
            parts.append(text)

    return "".join(parts).strip()


def extract_stream_text(result: Any) -> str:
    """
    Global streamed-output extractor.

    Handles:
      - LangChain/LangGraph iterators
      - LlamaIndex workflow handlers with stream_events()
    """
    if result is None:
        return ""

    if isinstance(result, str):
        return result.strip()

    if hasattr(result, "__aiter__"):
        return run_awaitable_sync(_collect_async_event_stream_text(result))

    if _looks_like_llamaindex_stream_handler(result):
        return run_awaitable_sync(_collect_llamaindex_stream_text(result))

    return _collect_langchain_stream_text(result)


def extract_output_text(result: Any) -> str:
    """
    Global non-stream output extractor.

    Handles plain strings, dict payloads, LangChain message dicts,
    and LlamaIndex result objects.
    """
    if result is None:
        return ""

    if isinstance(result, str):
        return result.strip()

    text = _extract_text_from_message_like(result)
    if text:
        return text.strip()

    if asyncio.isfuture(result) or asyncio.iscoroutine(result) or hasattr(result, "__await__"):
        resolved = run_awaitable_sync(result)
        return extract_output_text(resolved)

    return str(result)

def render_response_text(raw_response: Any, stream: bool) -> str:
    """Single framework-agnostic response normalizer for CLI/web callers."""
    if stream:
        return extract_stream_text(raw_response)
    return extract_output_text(raw_response)


def normalize_agent_api_payload(raw_response: Any, stream: bool) -> dict[str, Any]:
    """Convert agent responses into the chapter_8-compatible API payload shape."""
    response_text = render_response_text(raw_response, stream=stream).strip()
    payload: dict[str, Any] = {
        "response": response_text,
        "raw_answer": response_text,
    }

    if not response_text:
        return payload

    try:
        structured = parse_structured_json_response(response_text)
    except Exception:
        return payload

    final_answer = structured.get("final_answer")
    if isinstance(final_answer, str) and final_answer.strip():
        payload["raw_answer"] = final_answer.strip()

    payload["response"] = structured

    tool_calls = structured.get("tool_calls")
    if isinstance(tool_calls, list):
        payload["tool_calls"] = tool_calls

    return payload


def get_provider_options(model_identifiers: Iterable[str] | None) -> list[dict[str, str]]:
    """Return configured model options formatted for the chapter_8 provider selector."""
    availability = get_model_availability(model_identifiers)
    providers: list[dict[str, str]] = []
    for entry in availability["routable"]:
        provider = format_provider_display_name(str(entry["provider"]))
        providers.append(
            {
                "name": str(entry["identifier"]),
                "display_name": f"{provider} - {entry['model']}",
                "provider": str(entry["provider"]),
                "model": str(entry["model"]),
                "status": "Ready" if entry in availability["configured"] else "Unavailable",
            }
        )
    return providers


def run_awaitable_sync(value: Any) -> Any:
    """Resolve awaitables in both plain Python and running event-loop contexts."""
    if not asyncio.iscoroutine(value) and not hasattr(value, "__await__"):
        return value

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)

    result: Dict[str, Any] = {}
    error: Dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(value)
        except BaseException as exc:  # noqa: BLE001
            error["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in error:
        raise error["error"]
    return result.get("value")


def run_llamaindex_handler_sync(build_result: Callable[[], Any], stream: bool) -> Any:
    """Execute LlamaIndex workflow result construction in an active event loop.

    This avoids `RuntimeError: no running event loop` in synchronous callers by
    creating handler/stream objects inside an async context.
    """

    async def _runner() -> Any:
        result = build_result()

        if stream:
            return await _collect_async_event_stream_text(result)

        return await result

    return run_awaitable_sync(_runner())


def default_chunk_iterator(manager: AgentManagerProtocol, topic: str) -> Iterator[str]:
    """Default chunk iterator used when a manager does not implement one."""
    result = manager.ask_question(topic)

    if result.get("success"):
        response_text = render_response_text(result.get("response"), stream=getattr(manager, "stream", False))
    else:
        response_text = str(result.get("error") or "")

    yield from chunk_text(response_text)


def load_chapter_env() -> Path | None:
    """Load environment variables for provider SDKs.

    Canonical location: `shared/.env` at the repository root.
    Fallbacks remain for local overrides and backwards compatibility.
    """
    chapter_root = Path(__file__).resolve().parent
    candidates = [REPO_ROOT / "shared" / ".env", Path.cwd() / ".env"]
    for base in [chapter_root, *chapter_root.parents]:
        candidates.append(base / ".env")

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if not candidate.exists():
            continue

        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

        return candidate
    return None


# Load environment variables as soon as the shared bootstrap module is imported.
load_chapter_env()


def chapter_root_from_file(file_path: str, levels_up: int = 1) -> Path:
    """Resolve chapter root and ensure it's importable.

    Example for `.../chapter_1/langchain/agent.py`:
      levels_up=1 -> `.../chapter_1`
    """
    root = Path(file_path).resolve().parents[levels_up]
    if str(root) not in sys.path:
        sys.path.append(str(root))
    return root


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("mode", nargs="?", default="cli", choices=["cli", "web"])
    parser.add_argument("--stream", action="store_true", help="Enable /query/stream endpoint in web mode")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--model-identifier", dest="model_identifier", help="Explicit model identifier (skips interactive model selection in CLI mode)")
    return parser


def _provider_is_configured(provider: str) -> bool:
    env_keys = get_api_key_env_vars(provider)
    return any((os.getenv(key) or "").strip() for key in env_keys)


def format_provider_display_name(provider: str) -> str:
    normalized = provider.replace("-", "_").lower()
    return PROVIDER_DISPLAY_NAMES.get(normalized, provider.replace("_", " ").replace("-", " ").title())


def _model_label(model_name: str, model_uri: str, provider: str) -> str:
    provider_label = format_provider_display_name(provider)
    return f"{model_name} ({provider_label}, {model_uri})"


def _build_model_catalog(model_identifiers: Iterable[str] | None) -> list[dict[str, str]]:
    available = get_identifier_mappings()
    selected_names = list(model_identifiers or available.keys())
    catalog = []
    for name in selected_names:
        config = available.get(name)
        if config is None:
            continue
        catalog.append(
            {
                "name": name,
                "provider": config.provider,
                "model": config.model,
                "identifier": name,
                "label": _model_label(config.name, config.model, config.provider),
            }
        )
    return catalog


def get_model_availability(model_identifiers: Iterable[str] | None) -> dict[str, object]:
    catalog = _build_model_catalog(model_identifiers)
    configured = [entry for entry in catalog if _provider_is_configured(entry["provider"])]
    routable = configured or catalog
    unavailable_providers = sorted({
        entry["provider"] for entry in catalog if not _provider_is_configured(entry["provider"])
    })
    return {
        "catalog": catalog,
        "configured": configured,
        "routable": routable,
        "using_fallback_catalog": not configured,
        "unavailable_providers": unavailable_providers,
    }


def get_routable_model_identifiers(
    model_identifiers: Iterable[str] | None, explicit_model_identifier: str | None = None
) -> list[str]:
    if explicit_model_identifier:
        return [explicit_model_identifier]
    availability = get_model_availability(model_identifiers)
    return [entry["identifier"] for entry in availability["routable"]]


def describe_model_availability(model_identifiers: Iterable[str] | None) -> str:
    availability = get_model_availability(model_identifiers)
    configured_labels = [entry["label"] for entry in availability["routable"]]
    lines: list[str] = []
    if availability["using_fallback_catalog"]:
        lines.append("No provider API keys detected; automatic routing will consider every bundled model.")
    else:
        lines.append(
            "Automatic model routing is limited to configured providers: "
            + ", ".join(configured_labels)
            + "."
        )

    unavailable_providers = list(availability["unavailable_providers"])
    if unavailable_providers:
        lines.append(
            "Skipping unavailable providers: "
            + ", ".join(provider.replace("_", " ").title() for provider in unavailable_providers)
            + "."
        )
    return " ".join(lines)


def select_startup_model(model_identifiers: Iterable[str] | None, mode: str, explicit_model_identifier: str | None) -> str:
    """Resolve startup model with env-aware provider filtering.

    - explicit `--model-identifier` always wins
    - in CLI mode, offer an interactive list of configured models
    - default to the OpenAI option when available, otherwise first configured model
    """
    if explicit_model_identifier:
        return explicit_model_identifier

    availability = get_model_availability(model_identifiers)
    if not availability["catalog"]:
        raise ValueError("No model configurations available for startup selection.")

    default_index = 0

    if mode != "cli" or not sys.stdin.isatty() or not sys.stdout.isatty():
        return availability["routable"][default_index]["identifier"]

    print("\nModel selection (configured via environment variables):")
    for idx, entry in enumerate(availability["routable"], start=1):
        default_suffix = " [default]" if (idx - 1) == default_index else ""
        print(f"{idx}. {entry['label']}{default_suffix}")

    while True:
        raw = input(
            f"Select model (1-{len(availability['routable'])}, default {default_index + 1}): "
        ).strip()
        if not raw:
            return availability["routable"][default_index]["identifier"]
        try:
            choice = int(raw) - 1
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        if 0 <= choice < len(availability["routable"]):
            return availability["routable"][choice]["identifier"]
        print("Invalid selection. Please try again.")


def run_interactive_cli(manager: AgentManagerProtocol) -> None:
    _print_cli_banner(manager)
    while True:
        try:
            user_input = _read_cli_prompt()
            if user_input is None:
                print("\nExiting.")
                break

            if user_input.lower() in {"exit", "quit"}:
                print("Exiting.")
                break
            if not user_input:
                print("No prompt provided. Please enter a question.")
                continue

            print("*** Agent is working locally ***")
            result = manager.ask_question(user_input)
            if result.get("local_only"):
                print("*** Local tool response generated (LLM bypassed) ***")
            else:
                print("*** Agent working with LLM, awaiting response ***")

            print("\n============= AGENT RESPONSE =============")

            if result.get("success"):
                response = render_response_text(
                    raw_response=result.get("response"),
                    stream=getattr(manager, "stream", False),
                )
                print(response)
            elif result.get("error"):
                print(result.get("error"))
                
            print("==========================================\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break


def run_mode(manager: AgentManagerProtocol, mode: str, host: str, port: int, stream: bool) -> None:
    if mode == "web":
        try:
            from web import run_web_server

            run_web_server(manager, host=host, port=port, enable_streaming=stream)
            return
        except ImportError:
            print("Error: web.py not found or FastAPI/Uvicorn not installed.")
            print("Install dependencies: pip install fastapi uvicorn")
            raise SystemExit(1)

    run_interactive_cli(manager)
