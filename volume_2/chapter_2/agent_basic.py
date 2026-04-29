#!/usr/bin/env python3
"""Simplified LangChain agent for volume 2 chapter 2 (UUID tool only)."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict

CHAPTER_1_ROOT = Path(__file__).resolve().parents[1] / "chapter_1"
if str(CHAPTER_1_ROOT) not in sys.path:
    sys.path.append(str(CHAPTER_1_ROOT))

from langchain.agents import create_agent
from langchain.tools import tool

import tools
from utils import (
    ALL_MODEL_IDENTIFIERS,
    build_common_parser,
    get_chapter_logger,
    get_identifier_mappings,
    log_tool_call,
    run_mode,
    select_startup_model,
)

logger = get_chapter_logger("volume_2.chapter_2.agent_uuid")


@tool
def generate_uuid_tool(_: str = ""):
    """Generate a unique UUID identifier."""
    return log_tool_call(logger, "generate_uuid", tools.generate_uuid)()


class LangChainUuidAgentManager:
    framework = "LangChain UUID Agent"
    tool_names = ["generate_uuid"]
    model_identifiers = ALL_MODEL_IDENTIFIERS

    def __init__(self, model: str, stream: bool = True):
        config = get_identifier_mappings().get(model)
        self.active_model_identifier = model
        self.provider = config.provider if config else "unknown"
        self.model = config.model if config else model
        self.stream = stream

        provider_name = "google_genai" if self.provider == "google" else self.provider
        self.agent = create_agent(
            model=f"{provider_name}:{self.model}",
            tools=[generate_uuid_tool],
            system_prompt=(
                "You are an AI assistant that can use only one tool: generate_uuid. "
                "Use generate_uuid whenever the user asks for a UUID or unique identifier."
            ),
        )

    def ask_question(self, topic: str) -> Dict[str, Any]:
        try:
            input_payload = {"messages": [{"role": "user", "content": topic}]}
            if self.stream:
                result = self.agent.stream(input_payload, stream_mode=["messages", "updates"])
            else:
                result = self.agent.invoke(input_payload)

            return {
                "success": True,
                "stream": self.stream,
                "provider": self.provider,
                "model": self.model,
                "prompt": topic,
                "response": result,
            }
        except Exception as exc:
            logger.exception("LangChain UUID ask_question failed")
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
    parser = build_common_parser("LangChain UUID Agent")
    args = parser.parse_args()
    startup_model = select_startup_model(ALL_MODEL_IDENTIFIERS, args.mode, args.model_identifier)
    manager = LangChainUuidAgentManager(model=startup_model, stream=args.stream)
    run_mode(manager, args.mode, args.host, args.port, args.stream)


if __name__ == "__main__":
    main()
