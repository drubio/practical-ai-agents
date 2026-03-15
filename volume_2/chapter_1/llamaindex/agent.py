#!/usr/bin/env python3
"""LlamaIndex FunctionAgent for volume 2 chapter 1."""

from __future__ import annotations

import json
from pathlib import Path
import re
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
    FunctionTool.from_defaults(fn=log_tool_call(logger, "calculator", tools.calculator), name="calculator"),
    FunctionTool.from_defaults(fn=log_tool_call(logger, "resolve_datetime", tools.resolve_datetime), name="resolve_datetime"),
    FunctionTool.from_defaults(fn=log_tool_call(logger, "format_json", tools.format_json), name="format_json"),
]


def _pick_local_tool(topic: str) -> tuple[str, str] | None:
    text = (topic or "").strip()
    if not text:
        return None

    calculator_match = re.match(r"^(?:calculate|calc|compute)\s+(.+)$", text, flags=re.IGNORECASE)
    if calculator_match:
        return ("calculator", calculator_match.group(1).strip())

    if any(op in text for op in ["+", "-", "*", "/", "="]):
        return ("calculator", text.replace("=", " ").strip())

    if text.lstrip().startswith(("{", "[")):
        return ("format_json", text)

    format_match = re.match(r"^(?:format\s+json|pretty\s+print\s+json)\s*[:\-]?\s*(.+)$", text, flags=re.IGNORECASE)
    if format_match:
        return ("format_json", format_match.group(1).strip())

    datetime_match = re.match(r"^(?:resolve\s+datetime|parse\s+date(?:time)?|when\s+is)\s+(.+)$", text, flags=re.IGNORECASE)
    if datetime_match:
        return ("resolve_datetime", datetime_match.group(1).strip())

    if any(token in text.lower() for token in ["tomorrow", "next week", "next month", "today", " at "]):
        return ("resolve_datetime", text)

    return None


class LlamaIndexAgentManager:
    framework = "LlamaIndex Agent"
    tool_names = ["calculator", "resolve_datetime", "format_json"]
    tool_trigger_help = (
        "Tools are selected automatically from your prompt; you do not need to type a tool name. "
        "If you want a specific behavior, ask explicitly (for example: 'calculate 20 * 5 or parse tomorrow at 2pm')."
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
                "Think step-by-step, use tools when needed, and return a concise final answer."
            )
        )

    def ask_question(self, topic: str) -> Dict[str, Any]:
        try:
            local_tool_call = _pick_local_tool(topic)
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
