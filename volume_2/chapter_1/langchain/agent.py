#!/usr/bin/env python3
"""LangChain Function-Tool agent"""

from __future__ import annotations

from pathlib import Path
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
def summarize_text_tool(text: str):
    """Summarize text."""
    return log_tool_call(logger, "summarize_text", tools.summarize_text)(text)


AGENT_TOOLS = [summarize_text_tool]



class LangChainAgentManager:
    framework = "LangChain Agent"
    tool_names = ["summarize_text"]
    model_identifiers = ALL_MODEL_IDENTIFIERS
    tool_trigger_help = (
        "Tools are selected automatically from your prompt; you do not need to type a tool name. "
        "If you want a specific behavior, ask explicitly (for example: 'summarize this')."
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
            logger.info("Processing prompt | chars=%s", len(topic))
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
