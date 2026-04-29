#!/usr/bin/env python3
"""LangChain agent with explicit message logging."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict

CHAPTER_1_ROOT = Path(__file__).resolve().parents[1] / "chapter_1"
if str(CHAPTER_1_ROOT) not in sys.path:
    sys.path.append(str(CHAPTER_1_ROOT))

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

import tools
from utils import ALL_MODEL_IDENTIFIERS, build_common_parser, get_chapter_logger, get_identifier_mappings, log_tool_call, run_mode, select_startup_model

logger = get_chapter_logger("volume_2.chapter_2.agent_basic")
SYSTEM_PROMPT = "Use generate_uuid when user asks for UUID. Keep responses short."


@tool
def generate_uuid_tool(_: str = ""):
    """Generate a unique UUID identifier."""
    return log_tool_call(logger, "generate_uuid", tools.generate_uuid)()


class LangChainUuidAgentManager:
    framework = "LangChain Basic Agent"
    tool_names = ["generate_uuid"]
    tool_trigger_help = "Tools are triggered automatically. Ask for a UUID/ticket ID to trigger generate_uuid."

    def __init__(self, model: str):
        config = get_identifier_mappings().get(model)
        self.provider = config.provider if config else "openai"
        self.model = config.model if config else model
        provider_name = "google_genai" if self.provider == "google" else self.provider
        self.agent = create_agent(
            model=f"{provider_name}:{self.model}",
            tools=[generate_uuid_tool],
            system_prompt=SYSTEM_PROMPT,
        )

    def ask_question(self, topic: str) -> Dict[str, Any]:
        try:
            print("\n[STEP 1 - SYSTEM MESSAGE] SystemMessage")
            print(SystemMessage(content=SYSTEM_PROMPT).content)
            human = HumanMessage(content=topic)
            print("\n[STEP 2 - USER -> LLM] HumanMessage")
            print(human.content)

            final_text = ""
            for event in self.agent.stream({"messages": [human]}, stream_mode=["messages"]):
                chunk = event[1][0] if isinstance(event, tuple) and len(event) == 2 and isinstance(event[1], list) else event
                name = chunk.__class__.__name__
                text = getattr(chunk, "content", "") or ""

                tool_calls = getattr(chunk, "tool_calls", None)
                if "AIMessage" in name and tool_calls:
                    print("\n[STEP 3 - LLM -> AGENT TOOL INSTRUCTIONS] AIMessage.tool_calls")
                    print(tool_calls)

                if name == "AIMessageChunk":
                    out = text if isinstance(text, str) else str(text)
                    print(out, end="")
                    final_text += out
                elif "ToolMessage" in name:
                    print("\n[STEP 4 - TOOL -> LLM] ToolMessage")
                    print(text)
                elif "AIMessage" in name:
                    print("\n[STEP 5 - LLM FINAL MESSAGE] AIMessage")
                    print(text)

            print("\n")
            return {"success": True, "final_text": final_text.strip()}
        except Exception as exc:
            logger.exception(exc)
            return {"success": False, "error": str(exc)}


def main() -> None:
    parser = build_common_parser("LangChain Basic Agent")
    args = parser.parse_args()
    startup_model = select_startup_model(ALL_MODEL_IDENTIFIERS, args.mode, args.model_identifier)
    manager = LangChainUuidAgentManager(model=startup_model)
    run_mode(manager, args.mode, args.host, args.port, args.stream)


if __name__ == "__main__":
    main()
