#!/usr/bin/env python3
"""LangChain Function-Tool agent"""

from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import Any, Dict

CHAPTER_ROOT = Path(__file__).resolve().parents[1]
if str(CHAPTER_ROOT) not in sys.path:
    sys.path.append(str(CHAPTER_ROOT))

from utils import (
    build_task_prompt,
    build_common_parser,
    get_chapter_logger,
    log_tool_call,
    run_mode,
    select_startup_model,
)

from langchain.agents import create_agent
from langchain.tools import tool

import tools
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils import ALL_MODEL_IDENTIFIERS, get_identifier_mappings


logger = get_chapter_logger("volume_2.chapter_1.langchain.agent")

FINAL_RESPONSE_INSTRUCTION = (
    "Treat each non-empty line in the user prompt as a required task and handle every requested step before answering. "
    "You must call calculator for arithmetic expressions or fee calculations, "
    "resolve_datetime for scheduling/date phrases, and generate_uuid for unique IDs. "
    "Return your final answer as JSON with this shape: "
    '{"tool_calls": [{"name": "<tool_name>", "arguments": {}, "output": "<serialized tool output>"}], "final_answer": "<human readable summary>"}. '
    "Include one tool_calls entry for every tool invocation, in execution order, and store that tool's output or result directly on the same object. Use an empty array when no tools are needed. "
    "Capture the actual tool name, arguments, and output/result exactly as used, and summarize the result for the user in final_answer."
)


@tool
def calculator_tool(expression: str):
    """Safely evaluate arithmetic expressions."""
    return log_tool_call(logger, "calculator", tools.calculator)(expression)


@tool
def resolve_datetime_tool(text: str):
    """Resolve date/time phrases."""
    return log_tool_call(logger, "resolve_datetime", tools.resolve_datetime)(text)


@tool
def generate_uuid_tool(_: str = ""):
    """Generate a unique UUID identifier."""
    return log_tool_call(logger, "generate_uuid", tools.generate_uuid)()


AGENT_TOOLS = [calculator_tool, resolve_datetime_tool, generate_uuid_tool]


class LangChainAgentManager:
    framework = "LangChain Agent"
    tool_names = ["calculator", "resolve_datetime", "generate_uuid"]
    model_identifiers = ALL_MODEL_IDENTIFIERS
    tool_trigger_help = (
        "Tools are selected automatically from your prompt; you do not need to type a tool name. "
        "If you want a specific behavior, ask explicitly "
        "(for example: 'calculate 20 * 5', 'parse tomorrow at 2pm', or 'generate a unique ticket ID')."
    )

    def __init__(self, model: str, stream: bool = True):
        config = get_identifier_mappings().get(model)
        self.active_model_identifier = model
        self.provider = config.provider if config else "unknown"
        self.model = config.model if config else model
        self.stream = stream
        logger.info(
            "Initializing LangChain agent | provider=%s | model=%s | stream=%s",
            self.provider,
            self.model,
            self.stream,
        )
        provider_name = "google_genai" if self.provider == "google" else self.provider
        self.agent = create_agent(
            model=f"{provider_name}:{self.model}",
            tools=AGENT_TOOLS,
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
            logger.info("Delegating full prompt to LangChain agent")
            input = {"messages": [{"role": "user", "content": build_task_prompt(topic)}]}
            if self.stream:
                logger.info("Awaiting streamed LangChain agent response")
                result = self.agent.stream(input, stream_mode=["messages", "updates"])
            else:
                logger.info("Awaiting LangChain agent response")
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
