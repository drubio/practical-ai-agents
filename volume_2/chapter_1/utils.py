"""Shared bootstrap utilities for volume 2 chapter agents.

This module is framework-agnostic and intended to be reused by LangChain,
LlamaIndex, and future chapter agent implementations.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Protocol


class AgentManagerProtocol(Protocol):
    framework: str
    model: str

    def ask_question(self, topic: str) -> Dict[str, Any]:
        ...

    def iter_answer_chunks(self, topic: str) -> Iterator[str]:
        ...


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
    parser.add_argument("--model", default="gpt-4.1")
    return parser


def run_interactive_cli(manager: AgentManagerProtocol) -> None:
    print(f"\n===== {manager.framework} CLI =====\n")
    while True:
        try:
            user_input = input("> ").strip()
            if not user_input:
                continue
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
