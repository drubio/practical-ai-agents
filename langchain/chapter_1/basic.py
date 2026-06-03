#!/usr/bin/env python3
"""LangChain agent with explicit message logging."""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from typing import Any, Dict
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

from shared.langchain.tools import create_generate_uuid_tool
from shared.langchain.utils import (
    ALL_MODEL_IDENTIFIERS,
    build_common_parser,
    create_agent_step_state,
    get_chapter_logger,
    get_identifier_mappings,
    print_agent_step_message,
    print_agent_step_output,
    run_mode,
    select_startup_model,
    stream_message_chunks,
)
from shared.utils import create_langchain_model

logger = get_chapter_logger("langchain.chapter_1.basic")
SYSTEM_PROMPT = "Use generate_uuid when user asks for UUID. Keep responses short."


class LangChainAgentManager:
    """LangChain chapter 1 agent manager with reusable startup hooks."""

    framework = "LangChain Basic Agent"
    prints_own_output = True
    tool_names = ["generate_uuid"]
    tool_trigger_help = "Tools are triggered automatically. Ask for a UUID/ticket ID to trigger generate_uuid."

    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 1000):
        model_config = get_identifier_mappings().get(model)
        self.model_identifier = model
        self.provider = model_config.provider if model_config else "openai"
        self.model = model_config.model if model_config else model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.pending_tool_logs: list[dict[str, Any]] = []
        self.agent = self._build_agent()

    def _create_model(self):
        return create_langchain_model(
            self.model_identifier,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _build_tools(self):
        return [create_generate_uuid_tool(self.pending_tool_logs)]

    def _build_agent(self):
        return create_agent(
            model=self._create_model(),
            tools=self._build_tools(),
            system_prompt=SYSTEM_PROMPT,
        )

    def _build_messages(self, topic: str):
        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=topic),
        ]

    def ask_question(self, topic: str, stream: bool = False) -> Dict[str, Any]:
        try:
            initial_messages = self._build_messages(topic)
            human_message = initial_messages[1]

            self.pending_tool_logs.clear()
            state = create_agent_step_state(self.pending_tool_logs)
            for message in initial_messages:
                print_agent_step_message(message, state, logger)

            if stream:
                final_text = print_agent_step_output(
                    logger,
                    stream_chunks=stream_message_chunks(self.agent, human_message),
                    state=state,
                )
            else:
                response = self.agent.invoke({"messages": [human_message]})
                response_messages = (
                    response.get("messages", []) if isinstance(response, dict) else []
                )
                final_text = print_agent_step_output(
                    logger, response_messages, state=state
                )

            return {
                "success": True,
                "final_text": final_text,
                "finalText": final_text,
                "provider": self.provider,
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        except Exception as exc:
            logger.exception(exc)
            return {"success": False, "error": str(exc)}


def main() -> None:
    parser = build_common_parser("LangChain Basic Agent")
    args = parser.parse_args()
    startup_model = select_startup_model(
        ALL_MODEL_IDENTIFIERS, args.mode, args.model_identifier
    )
    manager = LangChainAgentManager(
        model=startup_model, temperature=args.temperature, max_tokens=args.max_tokens
    )
    run_mode(manager, args.mode, args.host, args.port, args.stream)


if __name__ == "__main__":
    main()
