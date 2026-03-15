#!/usr/bin/env python3
"""LangChain multi-tool + multi-model routing agent for volume 2 chapter 2."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Sequence
import re

CHAPTER_1_ROOT = Path(__file__).resolve().parents[2] / "chapter_1"
if str(CHAPTER_1_ROOT) not in sys.path:
    sys.path.append(str(CHAPTER_1_ROOT))

from langchain.agents import create_agent
from langchain.tools import tool

import tools
from models import ALL_MODEL_IDENTIFIERS, ModelConfig, get_identifier_mappings, route_model_for_prompt
from utils import build_common_parser, get_chapter_logger, log_tool_call, run_mode, select_startup_model

logger = get_chapter_logger("volume_2.chapter_2.langchain.agent_routing")

ALL_TOOL_NAMES = [
    "calculator",
    "resolve_datetime",
    "format_json",
]


def build_tools(log_tool_call_fn, active_logger):
    """Build all available LangChain tools once and return by name."""

    @tool
    def calculator_tool(expression: str):
        """Evaluate expression."""
        return log_tool_call_fn(active_logger, "calculator", tools.calculator)(expression)

    @tool
    def resolve_datetime_tool(text: str):
        """Resolve datetime from text."""
        return log_tool_call_fn(active_logger, "resolve_datetime", tools.resolve_datetime)(text)

    @tool
    def format_json_tool(input: str):
        """Format JSON-like input."""
        return log_tool_call_fn(active_logger, "format_json", tools.format_json)(input)

    return {
        "calculator": calculator_tool,
        "resolve_datetime": resolve_datetime_tool,
        "format_json": format_json_tool,
    }


def select_tools(log_tool_call_fn, active_logger, tool_names):
    available = build_tools(log_tool_call_fn, active_logger)
    return [available[name] for name in tool_names]


def _trigger_match(prompt_l: str, trigger: str) -> bool:
    if " " in trigger or any(ch in trigger for ch in [":", "/", ".", "-"]):
        return trigger in prompt_l
    return re.search(rf"\b{re.escape(trigger)}\b", prompt_l) is not None


def route_tools_for_prompt(prompt: str) -> list[str]:
    """Select relevant tools from the prompt and keep the list minimal when possible."""
    prompt_l = prompt.lower()
    selected = set()

    keyword_routes = {
        "calculator": ["calculate", "math", "equation", "percentage"],
        "resolve_datetime": ["date", "time", "schedule", "tomorrow", "next week"],
        "format_json": ["json", "yaml", "format", "schema"],
    }

    for tool_name, triggers in keyword_routes.items():
        if any(_trigger_match(prompt_l, trigger) for trigger in triggers):
            selected.add(tool_name)

    has_math_expression = any(op in prompt for op in ["+", "*", "/", "="]) or (" - " in prompt)
    if has_math_expression:
        selected.add("calculator")

    if not selected:
        return ALL_TOOL_NAMES

    return [name for name in ALL_TOOL_NAMES if name in selected]


class LangChainAgentRoutingManager:
    framework = "LangChain Agent Routing"
    tool_names = ALL_TOOL_NAMES
    model_identifiers = ALL_MODEL_IDENTIFIERS
    tool_trigger_help = (
        "Tools are selected automatically from your prompt; you do not need to type a tool name. "
        "If you want a specific behavior, ask explicitly (for example: 'calculate 20 * 5 or format this JSON')."
    )

    def __init__(self, model: str, stream: bool = True):
        config = get_identifier_mappings().get(model)
        self.provider = config.provider if config else "unknown"
        self.model = config.model if config else model
        self.stream = stream
        self._agent_cache: dict[tuple[str, tuple[str, ...]], Any] = {}
        logger.info(
            "Initializing LangChain routing agent | provider=%s | initial_model=%s | stream=%s",
            self.provider,
            self.model,
            self.stream,
        )

    def _get_agent(self, provider: str, model: str, selected_tool_names: Sequence[str]):
        key = (f"{provider}:{model}", tuple(selected_tool_names))
        if key not in self._agent_cache:
            logger.info("Building LangChain agent | provider=%s | model=%s | tools=%s", provider, model, ",".join(selected_tool_names))
            self._agent_cache[key] = create_agent(
                model=f"{provider}:{model}",
                tools=select_tools(log_tool_call, logger, selected_tool_names),
                system_prompt=(
                    "You are an AI assistant that can use tools. "
                    "Choose the best tool(s) among those provided, then return a concise final answer."
                ),
            )
        return self._agent_cache[key]

    def _route_model(self, topic: str, selected_tool_names: Sequence[str]) -> ModelConfig:
        return route_model_for_prompt(topic, selected_tool_names, model_identifiers=ALL_MODEL_IDENTIFIERS)

    def ask_question(self, topic: str) -> Dict[str, Any]:
        try:
            logger.info("Processing prompt | chars=%s", len(topic))
            selected_tool_names = route_tools_for_prompt(topic)
            selected_model = self._route_model(topic, selected_tool_names)
            self.provider = selected_model.provider
            self.model = selected_model.model

            logger.info(
                "Routing decision | provider=%s | model=%s | tier=%s | tools=%s",
                selected_model.provider,
                selected_model.model,
                selected_model.tier,
                ",".join(selected_tool_names),
            )

            agent = self._get_agent(selected_model.provider, selected_model.model, selected_tool_names)
            input_payload = {"messages": [{"role": "user", "content": topic}]}
            if self.stream:
                result = agent.stream(input_payload, stream_mode=["messages", "updates"])
            else:
                result = agent.invoke(input_payload)
            return {
                "success": True,
                "stream": self.stream,
                "provider": selected_model.provider,
                "model": selected_model.model,
                "model_name": selected_model.name,
                "model_tier": selected_model.tier,
                "selected_tools": selected_tool_names,
                "prompt": topic,
                "response": result,
            }
        except Exception as exc:
            logger.exception("LangChain ask_question failed")
            return {
                "success": False,
                "stream": self.stream,
                "provider": "unknown",
                "model": self.model,
                "prompt": topic,
                "error": str(exc),
                "response": None,
            }


def main() -> None:
    parser = build_common_parser("Volume 2 chapter 2 LangChain agent routing")
    args = parser.parse_args()
    startup_model = select_startup_model(ALL_MODEL_IDENTIFIERS, args.mode, args.model_identifier)
    manager = LangChainAgentRoutingManager(model=startup_model, stream=args.stream)
    run_mode(manager, args.mode, args.host, args.port, args.stream)


if __name__ == "__main__":
    main()
