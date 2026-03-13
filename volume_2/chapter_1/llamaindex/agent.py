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
from stream import chunk_text

from utils import (
    build_common_parser,
    get_chapter_logger,
    log_tool_call,
    run_awaitable_sync,
    run_mode,
    select_startup_model,
)

import tools  # noqa: E402
from models import LLAMAINDEX_MODEL_NAMES


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
    FunctionTool.from_defaults(fn=log_tool_call(logger, "summarize_text", tools.summarize_text), name="summarize_text"),
]


class LlamaIndexAgentManager:
    framework = "LlamaIndex Agent"
    tool_names = ["summarize_text"]
    tool_trigger_help = (
        "Tools are selected automatically from your prompt; you do not need to type a tool name. "
        "If you want a specific behavior, ask explicitly (for example: 'summarize this')."
    )

    def __init__(self, model: str):
        resolved_model, llm = resolve_llamaindex_model(model)
        self.provider = resolved_model.provider
        self.model = resolved_model.model
        logger.info("Initializing LlamaIndex agent | provider=%s | model=%s", self.provider, self.model)
        self.llm = llm
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
                return await self.agent.run(topic)

            raw = run_awaitable_sync(_run_agent())
            return {
                "success": True,
                "provider": self.provider,
                "model": self.model,
                "prompt": topic,
                "response": _extract_text(raw),
            }
        except Exception as exc:
            logger.exception("LlamaIndex ask_question failed")
            return {
                "success": False,
                "provider": self.provider,
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
    startup_model = select_startup_model(LLAMAINDEX_MODEL_NAMES, args.mode, args.model)
    manager = LlamaIndexAgentManager(model=startup_model)
    run_mode(manager, args.mode, args.host, args.port, args.stream)


if __name__ == "__main__":
    main()
