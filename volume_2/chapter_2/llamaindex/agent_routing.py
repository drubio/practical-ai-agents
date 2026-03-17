#!/usr/bin/env python3
"""LlamaIndex multi-tool routing agent for volume 2 chapter 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator
import sys

CHAPTER_1_ROOT = Path(__file__).resolve().parents[2] / "chapter_1"
if str(CHAPTER_1_ROOT) not in sys.path:
    sys.path.append(str(CHAPTER_1_ROOT))

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool

import tools
from models import ALL_MODEL_IDENTIFIERS, ModelConfig, resolve_llamaindex_model, route_model_for_prompt
from stream import chunk_text
from utils import (
    build_common_parser,
    get_chapter_logger,
    log_tool_call,
    run_llamaindex_handler_sync,
    run_mode,
    select_startup_model,
)

logger = get_chapter_logger("volume_2.chapter_2.llamaindex.agent_routing")

ALL_TOOL_NAMES = [
    "calculator",
    "resolve_datetime",
    "generate_uuid",
]


def build_tools(log_tool_call_fn, active_logger):
    return {
        "calculator": FunctionTool.from_defaults(
            fn=log_tool_call_fn(active_logger, "calculator", tools.calculator),
            name="calculator",
        ),
        "resolve_datetime": FunctionTool.from_defaults(
            fn=log_tool_call_fn(active_logger, "resolve_datetime", tools.resolve_datetime),
            name="resolve_datetime",
        ),
        "generate_uuid": FunctionTool.from_defaults(
            fn=log_tool_call_fn(active_logger, "generate_uuid", tools.generate_uuid),
            name="generate_uuid",
        ),
    }


def select_tools(log_tool_call_fn, active_logger, tool_names):
    available = build_tools(log_tool_call_fn, active_logger)
    return [available[name] for name in tool_names]


class LlamaIndexAgentRoutingManager:
    framework = "LlamaIndex Agent Routing"
    tool_names = ALL_TOOL_NAMES
    model_identifiers = ALL_MODEL_IDENTIFIERS
    tool_trigger_help = (
        "Tools are selected automatically from your prompt; you do not need to type a tool name. "
        "If you want a specific behavior, ask explicitly (for example: 'calculate 20 * 5' or 'parse tomorrow at 2pm')."
    )

    def __init__(self, model: str, stream: bool = False):
        resolved_model, llm = resolve_llamaindex_model(model)
        self.provider = resolved_model.provider
        self.model = resolved_model.model
        self.stream = stream
        logger.info(
            "Initializing LlamaIndex routing agent | provider=%s | model=%s | stream=%s",
            self.provider,
            self.model,
            self.stream,
        )
        self._llm_cache: dict[str, Any] = {resolved_model.name: llm}
        self._agent_cache: dict[tuple[str, tuple[str, ...]], FunctionAgent] = {}

    def _route_model(self, topic: str, selected_tool_names: list[str]) -> ModelConfig:
        return route_model_for_prompt(topic, selected_tool_names, model_identifiers=ALL_MODEL_IDENTIFIERS)

    def _get_llm(self, model_name: str):
        if model_name not in self._llm_cache:
            _, llm = resolve_llamaindex_model(model_name)
            self._llm_cache[model_name] = llm
        return self._llm_cache[model_name]

    def _get_agent(self, model_name: str, selected_tool_names: list[str]) -> FunctionAgent:
        key = (model_name, tuple(selected_tool_names))
        if key not in self._agent_cache:
            logger.info("Building LlamaIndex agent | model=%s | tools=%s", model_name, ",".join(selected_tool_names))
            self._agent_cache[key] = FunctionAgent(
                llm=self._get_llm(model_name),
                tools=select_tools(log_tool_call, logger, selected_tool_names),
                system_prompt=(
                    "You are an AI assistant that can use tools. "
                    "Choose the best tool(s) among those provided, then return a concise final answer."
                ),
            )
        return self._agent_cache[key]

    def ask_question(self, topic: str) -> Dict[str, Any]:
        try:
            logger.info("Processing prompt | chars=%s", len(topic))
            selected_tool_names = ALL_TOOL_NAMES
            selected_model = self._route_model(topic, selected_tool_names)
            self.provider = selected_model.provider
            self.model = selected_model.model
            agent = self._get_agent(selected_model.name, selected_tool_names)

            if self.stream:
                result = run_llamaindex_handler_sync(
                    lambda: agent.run(topic).stream_events(),
                    stream=True,
                )
            else:
                result = run_llamaindex_handler_sync(
                    lambda: agent.run(topic),
                    stream=False,
                )

            return {
                "success": True,
                "stream": self.stream,
                "provider": self.provider,
                "model": self.model,
                "model_name": selected_model.name,
                "model_tier": selected_model.tier,
                "selected_tools": selected_tool_names,
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

    def iter_answer_chunks(self, topic: str) -> Iterator[str]:
        final = self.ask_question(topic)
        text = final.get("response") or ""
        if self.stream:
            yield from chunk_text(str(text))
            return
        yield from chunk_text(str(text))


def main() -> None:
    parser = build_common_parser("Volume 2 chapter 2 LlamaIndex agent routing")
    args = parser.parse_args()
    startup_model = select_startup_model(ALL_MODEL_IDENTIFIERS, args.mode, args.model_identifier)
    manager = LlamaIndexAgentRoutingManager(model=startup_model, stream=args.stream)
    run_mode(manager, args.mode, args.host, args.port, args.stream)


if __name__ == "__main__":
    main()
