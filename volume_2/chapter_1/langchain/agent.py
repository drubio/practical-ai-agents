#!/usr/bin/env python3
"""LangChain Function-Tool agent"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Iterator

CHAPTER_ROOT = Path(__file__).resolve().parents[1]
if str(CHAPTER_ROOT) not in sys.path:
    sys.path.append(str(CHAPTER_ROOT))

from utils import build_common_parser, chapter_root_from_file, get_chapter_logger, log_tool_call, run_mode

chapter_root_from_file(__file__)

from langchain.agents import create_agent
from langchain.tools import tool

from tools import (
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


logger = get_chapter_logger("volume_2.chapter_1.langchain.agent")


@tool
def summarize_text_tool(text: str):
    """Summarize text."""
    return log_tool_call(logger, "summarize_text", summarize_text)(text)


@tool
def extract_keywords_tool(text: str):
    """Extract keywords."""
    return log_tool_call(logger, "extract_keywords", extract_keywords)(text)


@tool
def extract_tasks_tool(text: str):
    """Extract tasks from text."""
    return log_tool_call(logger, "extract_tasks", extract_tasks)(text)


@tool
def score_priority_tool(text: str):
    """Score priority from text."""
    return log_tool_call(logger, "score_priority", score_priority)(text)


@tool
def route_workflow_tool(text: str):
    """Route workflow from text."""
    return log_tool_call(logger, "route_workflow", route_workflow)(text)


@tool
def parse_content_tool(content: str):
    """Parse content."""
    return log_tool_call(logger, "parse_content", parse_content)(content)


@tool
def resolve_datetime_tool(text: str):
    """Resolve datetime from text."""
    return log_tool_call(logger, "resolve_datetime", resolve_datetime)(text)


@tool
def format_json_tool(input: str):
    """Format JSON-like input."""
    return log_tool_call(logger, "format_json", format_json)(input)


@tool
def calculator_tool(expression: str):
    """Evaluate expression."""
    return log_tool_call(logger, "calculator", calculator)(expression)


@tool
def analyze_text_tool(text: str):
    """Analyze text."""
    return log_tool_call(logger, "analyze_text", analyze_text)(text)


AGENT_TOOLS = [
    summarize_text_tool,
    extract_keywords_tool,
    extract_tasks_tool,
    score_priority_tool,
    route_workflow_tool,
    parse_content_tool,
    resolve_datetime_tool,
    format_json_tool,
    calculator_tool,
    analyze_text_tool,
]


def _extract_output(result: Dict[str, Any]) -> str:
    output = result.get("output")
    if isinstance(output, str):
        return output
    messages = result.get("messages") or []
    if messages:
        content = getattr(messages[-1], "content", None)
        if isinstance(content, str):
            return content
    return str(result)


class LangChainAgentManager:
    framework = "LangChain Agent"

    def __init__(self, model: str = "gpt-5.2"):
        self.model = model
        logger.info("Initializing LangChain agent | model=%s", model)
        self.agent = create_agent(
            model=model,
            tools=AGENT_TOOLS,
            system_prompt=(
                "You are an AI assistant that can use tools. "
                "Think step-by-step, use tools when needed, and return a concise final answer."
            ),
        )

    def ask_question(self, topic: str) -> Dict[str, Any]:
        try:
            logger.info("Processing prompt | chars=%s", len(topic))
            result = self.agent.invoke({"messages": [{"role": "user", "content": topic}]})
            return {
                "success": True,
                "provider": 'openai',
                "model": self.model,
                "prompt": topic,
                "response": _extract_output(result),
            }
        except Exception as exc:
            logger.exception("LangChain ask_question failed")
            return {
                "success": False,
                "provider": 'openai',
                "model": self.model,
                "prompt": topic,
                "error": str(exc),
                "response": None,
            }


def main() -> None:
    parser = build_common_parser("LangChain Agent")
    args = parser.parse_args()
    manager = LangChainAgentManager(model=args.model)
    run_mode(manager, args.mode, args.host, args.port, args.stream)


if __name__ == "__main__":
    main()
