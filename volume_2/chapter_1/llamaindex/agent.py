#!/usr/bin/env python3
"""LlamaIndex FunctionAgent for volume 2 chapter 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import sys

CHAPTER_ROOT = Path(__file__).resolve().parents[1]
if str(CHAPTER_ROOT) not in sys.path:
    sys.path.append(str(CHAPTER_ROOT))

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from utils import (
    build_common_parser,
    get_chapter_logger,
    log_tool_call,
    run_llamaindex_handler_sync,
    run_mode,
    select_startup_model,
)

import tools  # noqa: E402
from models import ALL_MODEL_IDENTIFIERS, resolve_llamaindex_model


logger = get_chapter_logger("volume_2.chapter_1.llamaindex.agent")


TOOLS = [
    FunctionTool.from_defaults(
        fn=log_tool_call(logger, "calculator", tools.calculator),
        name="calculator",
        description="Safely evaluate arithmetic expressions.",
    ),
    FunctionTool.from_defaults(
        fn=log_tool_call(logger, "resolve_datetime", tools.resolve_datetime),
        name="resolve_datetime",
        description="Resolve date/time phrases.",
    ),
    FunctionTool.from_defaults(
        fn=log_tool_call(logger, "generate_uuid", tools.generate_uuid),
        name="generate_uuid",
        description="Generate a unique UUID identifier.",
    ),
]


class LlamaIndexAgentManager:
    framework = "LlamaIndex Agent"
    tool_names = ["calculator", "resolve_datetime", "generate_uuid"]
    tool_trigger_help = (
        "Tools are selected automatically from your prompt; you do not need to type a tool name. "
        "If you want a specific behavior, ask explicitly "
        "(for example: 'calculate 20 * 5', 'parse tomorrow at 2pm', or 'generate a unique ticket ID')."
    )

    def __init__(self, model: str, stream: bool = False):
        resolved_model, llm = resolve_llamaindex_model(model)
        self.provider = resolved_model.provider
        self.model = resolved_model.model
        self.stream = stream

        logger.info(
            "Initializing LlamaIndex agent | provider=%s | model=%s | stream=%s",
            self.provider,
            self.model,
            self.stream,
        )

        self.llm = llm
        self.agent = FunctionAgent(
            llm=self.llm,
            tools=TOOLS,
            system_prompt=(
                "You are an AI assistant that can use tools. "
                "Use the calculator for arithmetic, resolve_datetime for date/time phrases, "
                "and generate_uuid when the user asks for a unique ID, UUID, ticket ID, or identifier. "
                "Think step-by-step, use tools when needed, and return a concise final answer."
            ),
        )

    def ask_question(self, topic: str) -> Dict[str, Any]:
        try:
            local_tool_call = tools.route_tool_for_prompt(topic, self.tool_names)
            if local_tool_call:
                name, tool_input = local_tool_call
                logger.info("Processing prompt locally | tool=%s | chars=%s", name, len(topic))
                observation = tools.run_tool(name, tool_input)
                return {
                    "success": True,
                    "stream": False,
                    "provider": self.provider,
                    "model": self.model,
                    "prompt": topic,
                    "local_only": True,
                    "selected_tool": name,
                    "response": json.dumps(observation, indent=2, ensure_ascii=False),
                }

            logger.info("Processing prompt with LLM | chars=%s", len(topic))

            if self.stream:
                result = run_llamaindex_handler_sync(
                    lambda: self.agent.run(topic).stream_events(),
                    stream=True,
                )
            else:
                result = run_llamaindex_handler_sync(
                    lambda: self.agent.run(topic),
                    stream=False,
                )

            return {
                "success": True,
                "stream": self.stream,
                "provider": self.provider,
                "model": self.model,
                "prompt": topic,
                "response": result,
            }

        except Exception as exc:
            logger.exception("LlamaIndex ask_question failed")
            return {
                "success": False,
                "stream": self.stream,
                "provider": self.provider,
                "model": self.model,
                "prompt": topic,
                "error": str(exc),
                "response": None,
            }


def main() -> None:
    parser = build_common_parser("Volume 2 chapter 1 LlamaIndex agent")
    args = parser.parse_args()
    startup_model = select_startup_model(ALL_MODEL_IDENTIFIERS, args.mode, args.model_identifier)
    manager = LlamaIndexAgentManager(model=startup_model, stream=args.stream)
    run_mode(manager, args.mode, args.host, args.port, args.stream)


if __name__ == "__main__":
    main()
