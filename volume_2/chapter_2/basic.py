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
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage

import tools
from utils import (
    ALL_MODEL_IDENTIFIERS,
    build_common_parser,
    get_chapter_logger,
    extract_text_content,
    get_identifier_mappings,
    log_tool_call,
    run_mode,
    select_startup_model,
    stream_message_chunks,
)

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

    def ask_question(self, topic: str, stream: bool = False) -> Dict[str, Any]:
        try:
            print("\n[STEP 1 - SYSTEM MESSAGE] SystemMessage")
            print(SystemMessage(content=SYSTEM_PROMPT).content)
            human = HumanMessage(content=topic)
            print("\n[STEP 2 - USER -> LLM] HumanMessage")
            print(human.content)

            final_text = ""
            if stream:
                for chunk, text in stream_message_chunks(self.agent, human):
                    tool_calls = getattr(chunk, "tool_calls", None)
                    if isinstance(chunk, (AIMessage, AIMessageChunk)) and tool_calls:
                        print("\n[STEP 3 - LLM -> AGENT TOOL INSTRUCTIONS] AIMessage.tool_calls")
                        print(tool_calls)

                    if isinstance(chunk, AIMessageChunk):
                        if text:
                            print(text, end="")
                        final_text += text
                    elif isinstance(chunk, ToolMessage):
                        print("\n[STEP 4 - TOOL -> LLM] ToolMessage")
                        print(text)
                    elif isinstance(chunk, AIMessage):
                        print("\n[STEP 5 - LLM FINAL MESSAGE] AIMessage")
                        print(text)
                        final_text += text
            else:
                response = self.agent.invoke({"messages": [human]})
                messages = response.get("messages", []) if isinstance(response, dict) else []
                for message in messages:
                    text = extract_text_content(getattr(message, "content", ""))
                    tool_calls = getattr(message, "tool_calls", None)
                    if isinstance(message, AIMessage) and tool_calls:
                        print("\n[STEP 3 - LLM -> AGENT TOOL INSTRUCTIONS] AIMessage.tool_calls")
                        print(tool_calls)
                    if isinstance(message, ToolMessage):
                        print("\n[STEP 4 - TOOL -> LLM] ToolMessage")
                        print(text)
                    elif isinstance(message, AIMessage):
                        print("\n[STEP 5 - LLM FINAL MESSAGE] AIMessage")
                        print(text)
                        final_text = text

            print("\n")
            normalized_final_text = final_text.strip()
            return {"success": True, "final_text": normalized_final_text, "finalText": normalized_final_text}
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
