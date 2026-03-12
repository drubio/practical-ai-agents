#!/usr/bin/env python3
"""Reusable LlamaIndex tool routing for volume 2 chapter 1+."""

from __future__ import annotations

from pathlib import Path
import sys

CHAPTER_1_ROOT = Path(__file__).resolve().parents[2] / "chapter_1"
if str(CHAPTER_1_ROOT) not in sys.path:
    sys.path.append(str(CHAPTER_1_ROOT))

from llama_index.core.tools import FunctionTool

import tools

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
    return {
        "summarize_text": FunctionTool.from_defaults(
            fn=log_tool_call(logger, "summarize_text", tools.summarize_text), name="summarize_text"
        ),
        "extract_keywords": FunctionTool.from_defaults(
            fn=log_tool_call(logger, "extract_keywords", tools.extract_keywords), name="extract_keywords"
        ),
        "extract_tasks": FunctionTool.from_defaults(
            fn=log_tool_call(logger, "extract_tasks", tools.extract_tasks), name="extract_tasks"
        ),
        "score_priority": FunctionTool.from_defaults(
            fn=log_tool_call(logger, "score_priority", tools.score_priority), name="score_priority"
        ),
        "route_workflow": FunctionTool.from_defaults(
            fn=log_tool_call(logger, "route_workflow", tools.route_workflow), name="route_workflow"
        ),
        "parse_content": FunctionTool.from_defaults(
            fn=log_tool_call(logger, "parse_content", tools.parse_content), name="parse_content"
        ),
        "resolve_datetime": FunctionTool.from_defaults(
            fn=log_tool_call(logger, "resolve_datetime", tools.resolve_datetime), name="resolve_datetime"
        ),
        "format_json": FunctionTool.from_defaults(
            fn=log_tool_call(logger, "format_json", tools.format_json), name="format_json"
        ),
        "calculator": FunctionTool.from_defaults(
            fn=log_tool_call(logger, "calculator", tools.calculator), name="calculator"
        ),
        "analyze_text": FunctionTool.from_defaults(
            fn=log_tool_call(logger, "analyze_text", tools.analyze_text), name="analyze_text"
        ),
    }


def select_tools(log_tool_call, logger, tool_names):
    available = build_tools(log_tool_call, logger)
    return [available[name] for name in tool_names]
