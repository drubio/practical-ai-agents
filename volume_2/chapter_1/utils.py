"""Shared chapter utilities for CLI and web modes."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from shared.llm_models import ALL_MODEL_IDENTIFIERS, get_identifier_mappings
from shared.utils import get_chapter_logger, log_tool_call
from shared.web import normalize_response_text


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
            elif "text" in item and isinstance(item.get("text"), str) and item["text"].strip():
                parts.append(item["text"])
        return "".join(parts)
    return normalize_response_text(content)


def stream_message_chunks(agent: Any, human_message: Any) -> Iterator[Tuple[Any, str]]:
    stream = agent.stream({"messages": [human_message]}, stream_mode="messages")
    for event in stream:
        chunk = event[0] if isinstance(event, tuple) and len(event) == 2 else event
        yield chunk, extract_text_content(getattr(chunk, "content", ""))


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("mode", nargs="?", default="cli", choices=["cli", "web"])
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--model-identifier", dest="model_identifier")
    return parser


def select_startup_model(model_identifiers: Iterable[str] | None, mode: str, explicit_model_identifier: str | None) -> str:
    if explicit_model_identifier:
        return explicit_model_identifier
    ids = list(model_identifiers or ALL_MODEL_IDENTIFIERS)
    if not ids:
        raise ValueError("No model identifiers configured")
    if mode != "cli" or not sys.stdin.isatty() or not sys.stdout.isatty():
        return ids[0]
    print("\nModel selection:")
    for i, mid in enumerate(ids, start=1):
        print(f"{i}. {mid}" + (" [default]" if i == 1 else ""))
    while True:
        raw = input(f"Select model (1-{len(ids)}, default 1): ").strip()
        if not raw:
            return ids[0]
        if raw.isdigit() and 1 <= int(raw) <= len(ids):
            return ids[int(raw) - 1]
        print("Invalid selection. Try again.")


def run_mode(manager: Any, mode: str, host: str, port: int, stream: bool) -> None:
    if mode == "web":
        from web import run_web_server

        run_web_server(manager, host=host, port=port, stream_default=stream)
        return

    print(f"\n===== {manager.framework} CLI =====")
    print("Type a question and press Enter.")
    print("Type 'exit' to quit.\n")

    names = getattr(manager, "tool_names", []) or []
    if names:
        print("Available local tools:")
        for name in names:
            print(f"  - {name}")
    else:
        print("Available local tools: (none declared)")

    print(f"\n{getattr(manager, 'tool_trigger_help', 'Tools are triggered automatically from your prompt.')}")
    print("Tip: ask for a UUID to force tool usage.")
    print("====================================")
    while True:
        prompt = input("> ").strip()
        if not prompt or prompt.lower() == "exit":
            break
        result: Dict[str, Any] = manager.ask_question(prompt)
        print(result.get("final_text") if result.get("success") else result.get("error"))
