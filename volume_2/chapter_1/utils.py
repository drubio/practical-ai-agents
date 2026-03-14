"""Shared bootstrap utilities for volume 2 chapter agents.

This module is framework-agnostic and intended to be reused by LangChain,
LlamaIndex, and future chapter agent implementations.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import select
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Protocol


class AgentManagerProtocol(Protocol):
    framework: str
    model: str

    def ask_question(self, topic: str) -> Dict[str, Any]:
        ...

    def iter_answer_chunks(self, topic: str) -> Iterator[str]:
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
        "Tools are triggered automatically based on your prompt. You can mention a specific task (for example: summarize, extract tasks, or calculate) to encourage tool use.",
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
    def wrapper(arg: Any) -> Any:
        logger.info("Tool call | name=%s | input=%s", tool_name, arg)
        result = fn(arg)
        logger.info("Tool result | name=%s | output=%s", tool_name, result)
        return result

    wrapper.__name__ = getattr(fn, "__name__", f"{tool_name}_wrapper")
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    return wrapper


def extract_output_text(result: Any) -> str:
    """Extract a human-readable final answer from framework result payloads."""
    if result is None:
        return ""

    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        output = result.get("output")
        if isinstance(output, str):
            return output

        messages = result.get("messages") or []
        if messages:
            content = getattr(messages[-1], "content", None)
            if isinstance(content, str):
                return content

    response = getattr(result, "response", None)
    if isinstance(response, str):
        return response

    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content

    return str(result)


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


def load_chapter_env() -> Path | None:
    """Load chapter-local `.env` so provider SDKs can find API keys.

    Search order:
    1) current working directory
    2) chapter root (`volume_2/chapter_1`)
    """
    chapter_root = Path(__file__).resolve().parent
    candidates = [Path.cwd() / ".env"]
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


PROVIDER_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"),
    "google_genai": ("GOOGLE_GENAI_API_KEY", "GOOGLE_API_KEY"),
    "xai": ("XAI_API_KEY",),
}


def _provider_is_configured(provider: str) -> bool:
    env_keys = PROVIDER_ENV_KEYS.get(provider, ())
    return any((os.getenv(key) or "").strip() for key in env_keys)


def _model_label(model_name: str, model_uri: str, provider: str) -> str:
    provider_label = provider.replace("_", " ").title()
    return f"{model_name} ({provider_label}, {model_uri})"


def _build_model_catalog(model_identifiers: Iterable[str] | None) -> list[dict[str, str]]:
    try:
        from models import get_identifier_mappings  # local import to avoid circular dependency at module load
    except Exception:  # noqa: BLE001
        return []

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


def select_startup_model(model_identifiers: Iterable[str] | None, mode: str, explicit_model_identifier: str | None) -> str:
    """Resolve startup model with env-aware provider filtering.

    - explicit `--model-identifier` always wins
    - in CLI mode, offer an interactive list of configured models
    - default to the OpenAI option when available, otherwise first configured model
    """
    if explicit_model_identifier:
        return explicit_model_identifier

    catalog = _build_model_catalog(model_identifiers)
    if not catalog:
        raise ValueError("No model configurations available for startup selection.")

    configured = [entry for entry in catalog if _provider_is_configured(entry["provider"])]
    if not configured:
        configured = catalog

    default_index = 0

    if mode != "cli" or not sys.stdin.isatty() or not sys.stdout.isatty():
        return configured[default_index]["identifier"]

    print("\nModel selection (configured via environment variables):")
    for idx, entry in enumerate(configured, start=1):
        default_suffix = " [default]" if (idx - 1) == default_index else ""
        print(f"{idx}. {entry['label']}{default_suffix}")

    while True:
        raw = input(f"Select model (1-{len(configured)}, default {default_index + 1}): ").strip()
        if not raw:
            return configured[default_index]["identifier"]
        try:
            choice = int(raw) - 1
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        if 0 <= choice < len(configured):
            return configured[choice]["identifier"]
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
            print("*** Agent working with LLM, awaiting response ***")
            result = manager.ask_question(user_input)
            print("\n============= LLM RESPONSE =============")
            print(result.get("response") if result.get("success") else result.get("error"))
            print("========================================\n")
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
