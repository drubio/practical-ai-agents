#!/usr/bin/env python3
"""Reusable LangChain tool routing for volume 2 chapter 1+."""

from __future__ import annotations

from pathlib import Path
import sys

CHAPTER_1_ROOT = Path(__file__).resolve().parents[2] / "chapter_1"
if str(CHAPTER_1_ROOT) not in sys.path:
    sys.path.append(str(CHAPTER_1_ROOT))

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


def build_tools(log_tool_call, logger):
    """Build all available LangChain tools once and return by name."""

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


def select_tools(log_tool_call, logger, tool_names):
    available = build_tools(log_tool_call, logger)
    return [available[name] for name in tool_names]
