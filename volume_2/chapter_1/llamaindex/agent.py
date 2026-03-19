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
    build_task_prompt,
    build_common_parser,
    get_chapter_logger,
    log_tool_call,
    run_llamaindex_handler_sync,
    run_mode,
    select_startup_model,
)

import tools  # noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils import ALL_MODEL_IDENTIFIERS, resolve_llamaindex_model


logger = get_chapter_logger("volume_2.chapter_1.llamaindex.agent")

FINAL_RESPONSE_INSTRUCTION = (
    "Treat each non-empty line in the user prompt as a required task and handle every requested step before answering. "
    "You must call calculator for arithmetic expressions or fee calculations, "
    "resolve_datetime for scheduling/date phrases, and generate_uuid for unique IDs. "
    "Return your final answer as JSON with this shape: "
    '{"tool_calls": [{"name": "<tool_name>", "arguments": {}, "output": "<serialized tool output>"}], "final_answer": "<human readable summary>"}. '
    "Include one tool_calls entry for every tool invocation, in execution order, and store that tool's output or result directly on the same object. Use an empty array when no tools are needed. "
    "Capture the actual tool name, arguments, and output/result exactly as used, and summarize the result for the user in final_answer."
)


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
        self.active_model_identifier = model
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
                f"{FINAL_RESPONSE_INSTRUCTION} "
                "Think step-by-step, use tools when needed."
            ),
        )

    def ask_question(self, topic: str) -> Dict[str, Any]:
        try:
            logger.info("Received prompt | chars=%s | multiline=%s", len(topic), "\n" in topic)
            logger.info("Delegating full prompt to LlamaIndex agent")

            if self.stream:
                logger.info("Awaiting streamed LlamaIndex agent response")
                result = run_llamaindex_handler_sync(
                    lambda: self.agent.run(build_task_prompt(topic)).stream_events(),
                    stream=True,
                )
            else:
                logger.info("Awaiting LlamaIndex agent response")
                result = run_llamaindex_handler_sync(
                    lambda: self.agent.run(build_task_prompt(topic)),
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
