#!/usr/bin/env python3
"""LangChain basic agent using the shared LLM manager and CLI architecture."""

import os
import sys
from typing import Any, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

from shared.langchain.tools import create_generate_uuid_tool
from shared.langchain.utils import (
    create_agent_step_state,
    extract_text_content,
    get_chapter_logger,
    print_agent_step_message,
    print_agent_step_output,
    stream_message_chunks,
)
from shared.utils import BaseLLMManager, create_langchain_model, interactive_cli

logger = get_chapter_logger("langchain.chapter_1.basic")
SYSTEM_PROMPT = "Use generate_uuid when user asks for UUID. Keep responses short."


class LangChainLLMManager(BaseLLMManager):
    """LangChain basic agent manager with reusable shared-manager hooks."""

    prints_own_output = True
    tool_names = ["generate_uuid"]
    tool_trigger_help = (
        "Tools are triggered automatically. Ask for a UUID/ticket ID to trigger "
        "generate_uuid."
    )

    def __init__(self, log_step_by_step: bool = True, stream: bool = False):
        self.log_step_by_step = log_step_by_step
        self.stream = stream
        self.prints_own_output = log_step_by_step or stream
        self.pending_tool_logs: list[dict[str, Any]] = []
        super().__init__("LangChain Basic Agent")

    def _test_provider(self, provider: str):
        self._create_model(
            self.provider_model_identifier(provider),
            temperature=0.7,
            max_tokens=1000,
        )

    def _create_model(self, selected_model: str, temperature: float, max_tokens: int):
        return create_langchain_model(
            selected_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _build_tools(self):
        return [create_generate_uuid_tool(self.pending_tool_logs)]

    def _build_agent(self, selected_model: str, temperature: float, max_tokens: int):
        return create_agent(
            model=self._create_model(selected_model, temperature, max_tokens),
            tools=self._build_tools(),
            system_prompt=SYSTEM_PROMPT,
        )

    def _build_messages(self, prompt: str):
        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

    def _extract_text(self, result: str) -> str:
        return result

    def ask_question(
        self,
        topic: str,
        provider: str = None,
        template: str = "{topic}",
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> Dict:
        prompt = template.format(topic=topic)
        model_config = self.resolve_model_config(provider)

        if not model_config:
            return {
                "success": False,
                "error": "No providers available",
                "provider": "none",
                "model": "none",
                "prompt": prompt,
                "response": None,
            }

        try:
            self.pending_tool_logs.clear()
            initial_messages = self._build_messages(prompt)
            human_message = initial_messages[1]
            agent = self._build_agent(model_config.name, temperature, max_tokens)

            if self.log_step_by_step:
                state = create_agent_step_state(self.pending_tool_logs)
                for message in initial_messages:
                    print_agent_step_message(message, state, logger)

                if self.stream:
                    final_text = print_agent_step_output(
                        logger,
                        stream_chunks=stream_message_chunks(agent, human_message),
                        state=state,
                    )
                else:
                    response = agent.invoke({"messages": [human_message]})
                    response_messages = (
                        response.get("messages", []) if isinstance(response, dict) else []
                    )
                    final_text = print_agent_step_output(
                        logger,
                        response_messages,
                        state=state,
                    )
            elif self.stream:
                parts = []
                for _chunk, text in stream_message_chunks(agent, human_message):
                    if text:
                        print(text, end="", flush=True)
                        parts.append(text)
                print()
                final_text = "".join(parts).strip()
            else:
                response = agent.invoke({"messages": [human_message]})
                response_messages = (
                    response.get("messages", []) if isinstance(response, dict) else []
                )
                final_text = ""
                for message in reversed(response_messages):
                    content = extract_text_content(getattr(message, "content", ""))
                    if content.strip():
                        final_text = content
                        break

            return {
                "success": True,
                "provider": model_config.provider,
                "model": model_config.model,
                "model_identifier": model_config.name,
                "prompt": prompt,
                "response": self._extract_text(final_text),
                "final_text": final_text,
                "finalText": final_text,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "log_step_by_step": self.log_step_by_step,
                "stream": self.stream,
            }
        except Exception as e:
            logger.exception(e)
            return {
                "success": False,
                "provider": model_config.provider,
                "model": model_config.model,
                "model_identifier": model_config.name,
                "prompt": prompt,
                "error": str(e),
                "response": None,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "log_step_by_step": self.log_step_by_step,
                "stream": self.stream,
            }


def main():
    args = sys.argv[1:]
    stream = "--stream" in args
    log_step_by_step = "--no-log-step-by-step" not in args

    if "web" in args:
        try:
            from shared.essentials.web import run_web_server

            run_web_server(
                lambda: LangChainLLMManager(
                    log_step_by_step=log_step_by_step,
                    stream=stream,
                )
            )
        except ImportError:
            print("Error: shared web API not found or FastAPI not installed.")
            print("Install FastAPI: pip install fastapi uvicorn")
            sys.exit(1)
    else:
        manager = LangChainLLMManager(
            log_step_by_step=log_step_by_step,
            stream=stream,
        )
        interactive_cli(manager)


if __name__ == "__main__":
    main()
