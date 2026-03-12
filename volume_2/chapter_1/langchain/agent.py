#!/usr/bin/env python3
"""LangChain Function-Tool agent"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict

CHAPTER_ROOT = Path(__file__).resolve().parents[1]
if str(CHAPTER_ROOT) not in sys.path:
    sys.path.append(str(CHAPTER_ROOT))

from utils import build_common_parser, chapter_root_from_file, get_chapter_logger, log_tool_call, run_mode

chapter_root_from_file(__file__)

from langchain.agents import create_agent
from langchain.tools import tool

import tools
from models import CHAPTER_1_MODEL_NAMES, select_models


logger = get_chapter_logger("volume_2.chapter_1.langchain.agent")


@tool
def summarize_text_tool(text: str):
    """Summarize text."""
    return log_tool_call(logger, "summarize_text", tools.summarize_text)(text)


AGENT_TOOLS = [summarize_text_tool]


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
    tool_names = ["summarize_text"]
    tool_trigger_help = (
        "Tools are selected automatically from your prompt; you do not need to type a tool name. "
        "If you want a specific behavior, ask explicitly (for example: 'summarize this')."
    )

    def __init__(self, model: str = "gpt-5.2"):
        # Chapter 1 intentionally exposes only one configured model.
        configured = select_models(CHAPTER_1_MODEL_NAMES)[0]
        self.model = configured.model
        logger.info("Initializing LangChain agent | configured_model=%s | cli_model=%s", self.model, model)
        self.agent = create_agent(
            model=self.model,
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
