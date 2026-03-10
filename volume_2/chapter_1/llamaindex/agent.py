#!/usr/bin/env python3
"""LlamaIndex FunctionAgent for volume 2 chapter 1."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any, Dict, Iterator

import sys

CHAPTER_ROOT = Path(__file__).resolve().parents[1]
if str(CHAPTER_ROOT) not in sys.path:
    sys.path.append(str(CHAPTER_ROOT))

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

from utils import build_common_parser, run_mode

from tools import (  # noqa: E402
    analyze_text,
    calculator,
    extract_keywords,
    extract_tasks,
    format_json,
    parse_content,
    resolve_datetime,
    route_workflow,
    score_priority,
    summarize_text,
)


def log_tool(name, fn):
    def wrapper(text: str):
        print("\n------------- LOCAL TOOL CALL -------------")
        print("Tool:", name)
        print("Input:", text)
        result = fn(text)
        print("\n------------- TOOL RESULT -----------------")
        print(result)
        print("-------------------------------------------\n")
        return result

    return wrapper


def _run_sync(value: Any) -> Any:
    if inspect.isawaitable(value):
        return asyncio.run(value)
    return value


def _extract_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    response = getattr(result, "response", None)
    if isinstance(response, str):
        return response
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    return str(result)


TOOLS = [
    FunctionTool.from_defaults(fn=log_tool("summarize_text", summarize_text), name="summarize_text"),
    FunctionTool.from_defaults(fn=log_tool("extract_keywords", extract_keywords), name="extract_keywords"),
    FunctionTool.from_defaults(fn=log_tool("extract_tasks", extract_tasks), name="extract_tasks"),
    FunctionTool.from_defaults(fn=log_tool("score_priority", score_priority), name="score_priority"),
    FunctionTool.from_defaults(fn=log_tool("route_workflow", route_workflow), name="route_workflow"),
    FunctionTool.from_defaults(fn=log_tool("parse_content", parse_content), name="parse_content"),
    FunctionTool.from_defaults(fn=log_tool("resolve_datetime", resolve_datetime), name="resolve_datetime"),
    FunctionTool.from_defaults(fn=log_tool("format_json", format_json), name="format_json"),
    FunctionTool.from_defaults(fn=log_tool("calculator", calculator), name="calculator"),
    FunctionTool.from_defaults(fn=log_tool("analyze_text", analyze_text), name="analyze_text"),
]


class LlamaIndexAgentManager:
    framework = "LlamaIndex Agent"

    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
        self.llm = OpenAI(model=model)
        self.agent = FunctionAgent(
            llm=self.llm,
            tools=TOOLS,
            system_prompt=(
                "You are an AI assistant that can use tools. "
                "Think step-by-step, use tools when needed, and return a concise final answer."
            ),
        )

    def ask_question(self, topic: str) -> Dict[str, Any]:
        try:
            raw = _run_sync(self.agent.run(topic))
            return {
                "success": True,
                "provider": "openai",
                "model": self.model,
                "prompt": topic,
                "response": _extract_text(raw),
            }
        except Exception as exc:
            return {
                "success": False,
                "provider": "openai",
                "model": self.model,
                "prompt": topic,
                "error": str(exc),
                "response": None,
            }

    def iter_answer_chunks(self, topic: str) -> Iterator[str]:
        final = self.ask_question(topic)
        text = final.get("response") or ""
        for i in range(0, len(text), 28):
            yield text[i : i + 28]


def main() -> None:
    parser = build_common_parser("Volume 2 chapter 1 LlamaIndex agent")
    args = parser.parse_args()
    manager = LlamaIndexAgentManager(model=args.model)
    run_mode(manager, args.mode, args.host, args.port, args.stream)


if __name__ == "__main__":
    main()
