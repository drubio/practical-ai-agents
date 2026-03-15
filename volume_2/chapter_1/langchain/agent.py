#!/usr/bin/env python3
"""LangChain Function-Tool agent"""

from __future__ import annotations

from pathlib import Path
import json
import re
import sys
from typing import Any, Dict

CHAPTER_ROOT = Path(__file__).resolve().parents[1]
if str(CHAPTER_ROOT) not in sys.path:
    sys.path.append(str(CHAPTER_ROOT))

from utils import (
    build_common_parser,
    get_chapter_logger,
    log_tool_call,
    run_mode,
    select_startup_model,
)

from langchain.agents import create_agent
from langchain.tools import tool

import tools
from models import ALL_MODEL_IDENTIFIERS, get_identifier_mappings


logger = get_chapter_logger("volume_2.chapter_1.langchain.agent")


@tool
def calculator_tool(expression: str):
    """Safely evaluate arithmetic expressions."""
    return log_tool_call(logger, "calculator", tools.calculator)(expression)


@tool
def resolve_datetime_tool(text: str):
    """Resolve date/time phrases."""
    return log_tool_call(logger, "resolve_datetime", tools.resolve_datetime)(text)


@tool
def format_json_tool(input: str):
    """Pretty-format JSON-compatible input."""
    return log_tool_call(logger, "format_json", tools.format_json)(input)


AGENT_TOOLS = [calculator_tool, resolve_datetime_tool, format_json_tool]


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



class LangChainAgentManager:
    framework = "LangChain Agent"
    tool_names = ["calculator", "resolve_datetime", "format_json"]
    model_identifiers = ALL_MODEL_IDENTIFIERS
    tool_trigger_help = (
        "Tools are selected automatically from your prompt; you do not need to type a tool name. "
        "If you want a specific behavior, ask explicitly (for example: 'calculate 20 * 5 or parse tomorrow at 2pm')."
    )

    def __init__(self, model: str, stream: bool = True):
        config = get_identifier_mappings().get(model)
        self.provider = config.provider if config else "unknown"
        self.model = config.model if config else model
        self.stream = stream
        logger.info("Initializing LangChain agent | provider=%s | model=%s | stream=%s", self.provider, self.model, self.stream)
        self.agent = create_agent(
            model=f"{self.provider}:{self.model}",
            tools=AGENT_TOOLS,
            system_prompt=(
                "You are an AI assistant that can use tools. "
                "Think step-by-step, use tools when needed, and return a concise final answer."
            ),
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
            input = {"messages": [{"role": "user", "content": topic}]}
            if self.stream:
                result = self.agent.stream(input,
                                           stream_mode=["messages", "updates"])
            else:
                result = self.agent.invoke(input)
                
            return {
                "success": True,
                "stream": self.stream,
                "provider": self.provider,
                "model": self.model,
                "prompt": topic,
                "response": result,
            }     
                
        except Exception as exc:
            logger.exception("LangChain ask_question failed")
            return {
                "success": False,
                "provider": self.provider,
                "model": self.model,
                "stream": self.stream,
                "prompt": topic,
                "error": str(exc),
                "response": None,
            }


def main() -> None:
    parser = build_common_parser("LangChain Agent")
    args = parser.parse_args()
    startup_model = select_startup_model(ALL_MODEL_IDENTIFIERS, args.mode, args.model_identifier)
    manager = LangChainAgentManager(model=startup_model, stream=args.stream)
    run_mode(manager, args.mode, args.host, args.port, args.stream)


if __name__ == "__main__":
    main()
