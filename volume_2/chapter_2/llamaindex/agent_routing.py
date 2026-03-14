#!/usr/bin/env python3
"""LlamaIndex multi-tool routing agent for volume 2 chapter 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator
import sys

CHAPTER_1_ROOT = Path(__file__).resolve().parents[2] / "chapter_1"
if str(CHAPTER_1_ROOT) not in sys.path:
    sys.path.append(str(CHAPTER_1_ROOT))

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool

import tools
from models import ALL_MODEL_IDENTIFIERS, resolve_llamaindex_model
from stream import chunk_text
from utils import (
    build_common_parser,
    extract_output_text,
    get_chapter_logger,
    log_tool_call,
    run_awaitable_sync,
    run_mode,
    select_startup_model,
)

logger = get_chapter_logger("volume_2.chapter_2.llamaindex.agent_routing")

ALL_TOOL_NAMES = [
    "summarize_text",
    "extract_keywords",
    "extract_tasks",
    "score_priority",
    "route_workflow",
    "parse_content",
    "resolve_datetime",
    "format_json",
    "calculator",
    "analyze_text",
]


def build_tools(log_tool_call_fn, active_logger):
    return {
        "summarize_text": FunctionTool.from_defaults(
            fn=log_tool_call_fn(active_logger, "summarize_text", tools.summarize_text), name="summarize_text"
        ),
        "extract_keywords": FunctionTool.from_defaults(
            fn=log_tool_call_fn(active_logger, "extract_keywords", tools.extract_keywords), name="extract_keywords"
        ),
        "extract_tasks": FunctionTool.from_defaults(
            fn=log_tool_call_fn(active_logger, "extract_tasks", tools.extract_tasks), name="extract_tasks"
        ),
        "score_priority": FunctionTool.from_defaults(
            fn=log_tool_call_fn(active_logger, "score_priority", tools.score_priority), name="score_priority"
        ),
        "route_workflow": FunctionTool.from_defaults(
            fn=log_tool_call_fn(active_logger, "route_workflow", tools.route_workflow), name="route_workflow"
        ),
        "parse_content": FunctionTool.from_defaults(
            fn=log_tool_call_fn(active_logger, "parse_content", tools.parse_content), name="parse_content"
        ),
        "resolve_datetime": FunctionTool.from_defaults(
            fn=log_tool_call_fn(active_logger, "resolve_datetime", tools.resolve_datetime), name="resolve_datetime"
        ),
        "format_json": FunctionTool.from_defaults(
            fn=log_tool_call_fn(active_logger, "format_json", tools.format_json), name="format_json"
        ),
        "calculator": FunctionTool.from_defaults(
            fn=log_tool_call_fn(active_logger, "calculator", tools.calculator), name="calculator"
        ),
        "analyze_text": FunctionTool.from_defaults(
            fn=log_tool_call_fn(active_logger, "analyze_text", tools.analyze_text), name="analyze_text"
        ),
    }


def select_tools(log_tool_call_fn, active_logger, tool_names):
    available = build_tools(log_tool_call_fn, active_logger)
    return [available[name] for name in tool_names]



class LlamaIndexAgentRoutingManager:
    framework = "LlamaIndex Agent Routing"
    tool_names = ALL_TOOL_NAMES
    model_identifiers = ALL_MODEL_IDENTIFIERS
    tool_trigger_help = (
        "Tools are selected automatically from your prompt; you do not need to type a tool name. "
        "If you want a specific behavior, ask explicitly (for example: 'extract tasks and score priority')."
    )

    def __init__(self, model: str):
        resolved_model, llm = resolve_llamaindex_model(model)
        self.provider = resolved_model.provider
        self.model = resolved_model.model
        logger.info("Initializing LlamaIndex routing agent | provider=%s | model=%s", self.provider, self.model)
        self.llm = llm
        self.agent = FunctionAgent(
            llm=self.llm,
            tools=select_tools(log_tool_call, logger, ALL_TOOL_NAMES),
            system_prompt=(
                "You are an AI assistant that can use tools. "
                "Think step-by-step, use tools when needed, and return a concise final answer."
            ),
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
                "response": extract_output_text(raw),
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
    parser = build_common_parser("Volume 2 chapter 2 LlamaIndex agent routing")
    args = parser.parse_args()
    startup_model = select_startup_model(ALL_MODEL_IDENTIFIERS, args.mode, args.model_identifier)
    manager = LlamaIndexAgentRoutingManager(model=startup_model)
    run_mode(manager, args.mode, args.host, args.port, args.stream)


if __name__ == "__main__":
    main()
