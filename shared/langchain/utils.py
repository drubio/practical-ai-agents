"""Shared LangChain chapter utilities for CLI, streaming, and web modes."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from shared.llm_models import (
    ALL_MODEL_IDENTIFIERS,
    get_all_providers,
    get_api_key,
    get_display_name,
    get_identifier_mappings,
    sort_providers_by_display_order,
)
from shared.utils import (
    compact_model_selection_lines,
    get_chapter_logger,
    log_tool_call,
    model_identifiers_for_providers,
    print_initialization_status,
    select_provider_model_identifier,
)
from shared.web import normalize_response_text

__all__ = [
    "ALL_MODEL_IDENTIFIERS",
    "build_common_parser",
    "compact_model_selection_lines",
    "extract_text_content",
    "get_chapter_logger",
    "get_identifier_mappings",
    "langchain_message_tool_calls",
    "langchain_message_type_name",
    "log_tool_call",
    "create_agent_step_state",
    "print_agent_step_message",
    "print_agent_step_output",
    "run_mode",
    "select_startup_model",
    "stream_message_chunks",
]


def extract_text_content(content: Any) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    parts.append(text_value)
            elif (
                "text" in item
                and isinstance(item.get("text"), str)
                and item["text"].strip()
            ):
                parts.append(item["text"])
        return "".join(parts)
    return normalize_response_text(content)


def stream_message_chunks(agent: Any, human_message: Any) -> Iterator[Tuple[Any, str]]:
    stream = agent.stream({"messages": [human_message]}, stream_mode="messages")
    for event in stream:
        chunk = event[0] if isinstance(event, tuple) and len(event) == 2 else event
        yield chunk, extract_text_content(getattr(chunk, "content", ""))


def langchain_message_type_name(message: Any) -> str:
    return message.__class__.__name__ if message is not None else "Message"


def langchain_message_tool_calls(message: Any) -> Any:
    tool_calls = getattr(message, "tool_calls", None)
    return tool_calls if tool_calls else None


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("mode", nargs="?", default="cli", choices=["cli", "web"])
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--model-identifier", dest="model_identifier")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=1000)
    return parser


def select_startup_model(
    model_identifiers: Iterable[str] | None,
    mode: str,
    explicit_model_identifier: str | None,
) -> str:
    if explicit_model_identifier:
        return explicit_model_identifier
    ids = list(model_identifiers or ALL_MODEL_IDENTIFIERS)
    if not ids:
        raise ValueError("No model identifiers configured")

    provider_statuses = {
        provider: (
            "✓ API key configured" if get_api_key(provider) else "✗ API key not found"
        )
        for provider in get_all_providers()
    }
    available_providers = sort_providers_by_display_order(
        [
            provider
            for provider, status in provider_statuses.items()
            if status.startswith("✓")
        ]
    )
    available_model_ids = [
        identifier
        for identifier in model_identifiers_for_providers(available_providers)
        if identifier in ids
    ]

    if mode != "cli" or not sys.stdin.isatty() or not sys.stdout.isatty():
        return (available_model_ids or ids)[0]

    print_initialization_status("LangChain", provider_statuses)
    if not available_model_ids:
        print(
            "No models available for initialized providers; using the first configured model."
        )
        return ids[0]

    selected_model = select_provider_model_identifier(available_providers)
    model_config = get_identifier_mappings()[selected_model]
    print(
        "\nUsing model: "
        f"{get_display_name(model_config.provider)} "
        f"(provider: {model_config.provider}, "
        f"model: {model_config.model} / "
        f"{model_config.name} / "
        f"{model_config.tier})"
    )
    return selected_model


def run_mode(manager: Any, mode: str, host: str, port: int, stream: bool) -> None:
    if mode == "web":
        from shared.langchain.web import run_web_server

        run_web_server(manager, host=host, port=port, stream_default=stream)
        return

    print(f"\n===== {manager.framework} CLI =====")
    names = getattr(manager, "tool_names", []) or []
    if names:
        print("Available local tools:")
        for name in names:
            print(f"  - {name}")
    else:
        print("Available local tools: (none declared)")

    print(
        f"\n{getattr(manager, 'tool_trigger_help', 'Tools are triggered automatically from your prompt.')}"
    )
    print("====================================")

    from shared.utils import interactive_basic_question_loop

    interactive_basic_question_loop(
        manager,
        provider=getattr(manager, "provider", None),
        model_identifier=getattr(manager, "model_identifier", None),
        ask_question=lambda prompt: manager.ask_question(prompt, stream=stream),
    )


def print_step_header(step: str, type_name: str) -> None:
    print(f"\n[{step}] {type_name}")


def print_step_message(step: str, message: Any, content: str | None = None) -> None:
    print_step_header(step, langchain_message_type_name(message))
    print(
        extract_text_content(getattr(message, "content", ""))
        if content is None
        else content
    )


def flush_pending_tool_logs(state: dict[str, Any], logger: Any) -> None:
    sys.stdout.flush()
    while state["pending_tool_logs"]:
        tool_log = state["pending_tool_logs"].pop(0)
        logger.info(
            "Tool call | name=%s | input=%s", tool_log["name"], tool_log["input"]
        )
        logger.info(
            "Tool result | name=%s | output=%s", tool_log["name"], tool_log["output"]
        )


def create_agent_step_state(
    pending_tool_logs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "final_text": "",
        "pending_tool_logs": pending_tool_logs if pending_tool_logs is not None else [],
        "printed_final_header": False,
        "printed_system_message": False,
        "printed_human_message": False,
    }


def print_agent_step_message(
    message: Any,
    state: dict[str, Any],
    logger: Any,
    *,
    stream: bool = False,
) -> None:
    if message is None:
        return

    type_name = langchain_message_type_name(message)
    tool_calls = langchain_message_tool_calls(message)
    if type_name == "SystemMessage":
        if not state["printed_system_message"]:
            print_step_message("STEP 1 - SYSTEM MESSAGE", message)
            state["printed_system_message"] = True
        return
    if type_name == "HumanMessage":
        if not state["printed_human_message"]:
            print_step_message("STEP 2 - USER -> LLM", message)
            state["printed_human_message"] = True
        return
    if "AIMessage" in type_name and tool_calls:
        print_step_header(
            "STEP 3 - LLM -> AGENT TOOL INSTRUCTIONS", "AIMessage.tool_calls"
        )
        print(tool_calls)
        return
    if type_name == "ToolMessage":
        flush_pending_tool_logs(state, logger)
        print_step_message("STEP 4 - TOOL -> LLM", message)
        return
    if type_name == "AIMessageChunk":
        delta = extract_text_content(getattr(message, "content", ""))
        if delta and not state["printed_final_header"]:
            print_step_header("STEP 5 - LLM FINAL MESSAGE", "AIMessage")
            state["printed_final_header"] = True
        if delta:
            print(delta, end="")
        state["final_text"] += delta
        return
    if "AIMessage" in type_name:
        text = extract_text_content(getattr(message, "content", ""))
        if stream:
            if not state["printed_final_header"]:
                print_step_message("STEP 5 - LLM FINAL MESSAGE", message, text)
                state["printed_final_header"] = True
            if not state["final_text"]:
                state["final_text"] = text
        else:
            print_step_message("STEP 5 - LLM FINAL MESSAGE", message, text)
            state["printed_final_header"] = True
            state["final_text"] = text


def print_agent_step_output(
    logger: Any,
    messages: list[Any] | None = None,
    stream_chunks: Any | None = None,
    state: dict[str, Any] | None = None,
) -> str:
    state = state or create_agent_step_state()
    for message in messages or []:
        print_agent_step_message(message, state, logger)
    if stream_chunks is not None:
        for chunk, _text in stream_chunks:
            print_agent_step_message(chunk, state, logger, stream=True)
    print("\n")
    return state["final_text"].strip()
