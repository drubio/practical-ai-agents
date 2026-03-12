#!/usr/bin/env python3
"""LlamaIndex FunctionAgent for volume 2 chapter 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator

import sys

CHAPTER_ROOT = Path(__file__).resolve().parents[1]
if str(CHAPTER_ROOT) not in sys.path:
    sys.path.append(str(CHAPTER_ROOT))

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from stream import chunk_text

from utils import (
    build_common_parser,
    get_chapter_logger,
    log_tool_call,
    run_awaitable_sync,
    run_mode,
)

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


logger = get_chapter_logger("volume_2.chapter_1.llamaindex.agent")


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
    FunctionTool.from_defaults(fn=log_tool_call(logger, "summarize_text", summarize_text), name="summarize_text"),
    FunctionTool.from_defaults(fn=log_tool_call(logger, "extract_keywords", extract_keywords), name="extract_keywords"),
    FunctionTool.from_defaults(fn=log_tool_call(logger, "extract_tasks", extract_tasks), name="extract_tasks"),
    FunctionTool.from_defaults(fn=log_tool_call(logger, "score_priority", score_priority), name="score_priority"),
    FunctionTool.from_defaults(fn=log_tool_call(logger, "route_workflow", route_workflow), name="route_workflow"),
    FunctionTool.from_defaults(fn=log_tool_call(logger, "parse_content", parse_content), name="parse_content"),
    FunctionTool.from_defaults(fn=log_tool_call(logger, "resolve_datetime", resolve_datetime), name="resolve_datetime"),
    FunctionTool.from_defaults(fn=log_tool_call(logger, "format_json", format_json), name="format_json"),
    FunctionTool.from_defaults(fn=log_tool_call(logger, "calculator", calculator), name="calculator"),
    FunctionTool.from_defaults(fn=log_tool_call(logger, "analyze_text", analyze_text), name="analyze_text"),
]


class LlamaIndexAgentManager:
    framework = "LlamaIndex Agent"

    def __init__(self, model: str = "gpt-5.2"):
        self.model = model
        logger.info("Initializing LlamaIndex agent | model=%s", model)
        self.llm = OpenAI(model=model)
        self.agent = FunctionAgent(
            llm=self.llm,
            tools=TOOLS,
            system_prompt=(
                "You are an AI assistant that can use tools. "
                "Think step-by-step, use tools when needed, and return a concise final answer."
            )
        )

    def ask_question(self, topic: str) -> Dict[str, Any]:
        try:
            logger.info("Processing prompt | chars=%s", len(topic))

            async def _run_agent() -> Any:
                # `FunctionAgent.run(...)` touches asyncio internals during execution.
                # Calling it inside this coroutine ensures those calls happen within
                # the event loop created by `run_awaitable_sync` when needed.
                return await self.agent.run(topic)

            raw = run_awaitable_sync(_run_agent())
            return {
                "success": True,
                "provider": "openai",
                "model": self.model,
                "prompt": topic,
                "response": _extract_text(raw),
            }
        except Exception as exc:
            logger.exception("LlamaIndex ask_question failed")
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
        yield from chunk_text(text)


def main() -> None:
    parser = build_common_parser("Volume 2 chapter 1 LlamaIndex agent")
    args = parser.parse_args()
    manager = LlamaIndexAgentManager(model=args.model)
    run_mode(manager, args.mode, args.host, args.port, args.stream)


if __name__ == "__main__":
    main()
