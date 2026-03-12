#!/usr/bin/env python3
"""LangChain multi-tool routing agent for volume 2 chapter 2."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict

CHAPTER_1_ROOT = Path(__file__).resolve().parents[2] / "chapter_1"
if str(CHAPTER_1_ROOT) not in sys.path:
    sys.path.append(str(CHAPTER_1_ROOT))

from langchain.agents import create_agent
from langchain.tools import tool

import tools
from utils import build_common_parser, get_chapter_logger, log_tool_call, run_mode

logger = get_chapter_logger("volume_2.chapter_2.langchain.agent_routing")

CHAPTER_1_TOOL_NAMES = ["summarize_text"]
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
    """Build all available LangChain tools once and return by name."""

    @tool
    def summarize_text_tool(text: str):
        """Summarize text."""
        return log_tool_call_fn(active_logger, "summarize_text", tools.summarize_text)(text)

    @tool
    def extract_keywords_tool(text: str):
        """Extract keywords."""
        return log_tool_call_fn(active_logger, "extract_keywords", tools.extract_keywords)(text)

    @tool
    def extract_tasks_tool(text: str):
        """Extract tasks from text."""
        return log_tool_call_fn(active_logger, "extract_tasks", tools.extract_tasks)(text)

    @tool
    def score_priority_tool(text: str):
        """Score priority from text."""
        return log_tool_call_fn(active_logger, "score_priority", tools.score_priority)(text)

    @tool
    def route_workflow_tool(text: str):
        """Route workflow from text."""
        return log_tool_call_fn(active_logger, "route_workflow", tools.route_workflow)(text)

    @tool
    def parse_content_tool(content: str):
        """Parse content."""
        return log_tool_call_fn(active_logger, "parse_content", tools.parse_content)(content)

    @tool
    def resolve_datetime_tool(text: str):
        """Resolve datetime from text."""
        return log_tool_call_fn(active_logger, "resolve_datetime", tools.resolve_datetime)(text)

    @tool
    def format_json_tool(input: str):
        """Format JSON-like input."""
        return log_tool_call_fn(active_logger, "format_json", tools.format_json)(input)

    @tool
    def calculator_tool(expression: str):
        """Evaluate expression."""
        return log_tool_call_fn(active_logger, "calculator", tools.calculator)(expression)

    @tool
    def analyze_text_tool(text: str):
        """Analyze text."""
        return log_tool_call_fn(active_logger, "analyze_text", tools.analyze_text)(text)

    return {
        "summarize_text": summarize_text_tool,
        "extract_keywords": extract_keywords_tool,
        "extract_tasks": extract_tasks_tool,
        "score_priority": score_priority_tool,
        "route_workflow": route_workflow_tool,
        "parse_content": parse_content_tool,
        "resolve_datetime": resolve_datetime_tool,
        "format_json": format_json_tool,
        "calculator": calculator_tool,
        "analyze_text": analyze_text_tool,
    }


def select_tools(log_tool_call_fn, active_logger, tool_names):
    available = build_tools(log_tool_call_fn, active_logger)
    return [available[name] for name in tool_names]


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


class LangChainAgentRoutingManager:
    framework = "LangChain Agent Routing"
    tool_names = ALL_TOOL_NAMES
    tool_trigger_help = (
        "Tools are selected automatically from your prompt; you do not need to type a tool name. "
        "If you want a specific behavior, ask explicitly (for example: 'extract tasks and score priority')."
    )

    def __init__(self, model: str = "gpt-5.2"):
        self.model = model
        logger.info("Initializing LangChain routing agent | model=%s", model)
        self.agent = create_agent(
            model=model,
            tools=select_tools(log_tool_call, logger, ALL_TOOL_NAMES),
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
                "provider": "openai",
                "model": self.model,
                "prompt": topic,
                "response": _extract_output(result),
            }
        except Exception as exc:
            logger.exception("LangChain ask_question failed")
            return {
                "success": False,
                "provider": "openai",
                "model": self.model,
                "prompt": topic,
                "error": str(exc),
                "response": None,
            }


def main() -> None:
    parser = build_common_parser("Volume 2 chapter 2 LangChain agent routing")
    args = parser.parse_args()
    manager = LangChainAgentRoutingManager(model=args.model)
    run_mode(manager, args.mode, args.host, args.port, args.stream)


if __name__ == "__main__":
    main()
