"""Minimal helpers for chapter 2 examples without breaking core CLI flows."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from shared.llm_models import ALL_MODEL_IDENTIFIERS, get_identifier_mappings


def get_chapter_logger(name: str):
    class _Logger:
        def info(self, *args: Any): print(f"[INFO] [{name}]", *args)
        def error(self, *args: Any): print(f"[ERROR] [{name}]", *args)
        exception = error
    return _Logger()


def log_tool_call(logger: Any, tool_name: str, fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def wrapper(arg: Any = None):
        logger.info(f"tool={tool_name} input=", arg)
        out = fn(arg)
        logger.info(f"tool={tool_name} output=", out)
        return out
    return wrapper


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


def run_mode(manager: Any, mode: str, _host: str, _port: int, _stream: bool) -> None:
    if mode == "web":
        raise SystemExit("Web mode removed for this chapter. Use cli mode.")
    print(f"\n===== {manager.framework} CLI =====")
    print("Type a question and press Enter.")
    print("Type 'exit' to quit.\n")

    names = getattr(manager, 'tool_names', []) or []
    if names:
        print('Available local tools:')
        for name in names:
            print(f'  - {name}')
    else:
        print('Available local tools: (none declared)')

    print(f"\n{getattr(manager, 'tool_trigger_help', 'Tools are triggered automatically from your prompt.')}")
    print("Tip: ask for a UUID to force tool usage.")
    print("====================================")
    while True:
        prompt = input("> ").strip()
        if not prompt or prompt.lower() == "exit":
            break
        result: Dict[str, Any] = manager.ask_question(prompt)
        print(result.get("final_text") if result.get("success") else result.get("error"))
