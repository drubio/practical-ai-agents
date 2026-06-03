#!/usr/bin/env python3
"""LangChain agent with explicit message logging."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from shared.langchain import tools
from shared.utils import create_langchain_model
from shared.langchain.utils import (
    ALL_MODEL_IDENTIFIERS,
    build_common_parser,
    extract_text_content,
    get_chapter_logger,
    get_identifier_mappings,
    langchain_message_tool_calls,
    langchain_message_type_name,
    run_mode,
    select_startup_model,
    stream_message_chunks,
)

logger = get_chapter_logger("langchain.chapter_1.basic")
SYSTEM_PROMPT = "Use generate_uuid when user asks for UUID. Keep responses short."


def _make_generate_uuid_tool(pending_tool_logs: list[dict[str, Any]]):
    @tool
    def generate_uuid_tool(tool_input: str = ""):
        """Generate a unique UUID identifier."""
        output = tools.generate_uuid(tool_input)
        pending_tool_logs.append({"name": "generate_uuid", "input": tool_input, "output": output})
        return output

    return generate_uuid_tool


def _print_step_header(step: str, type_name: str) -> None:
    print(f"\n[{step}] {type_name}")


def _print_step_message(step: str, message: Any, content: str | None = None) -> None:
    _print_step_header(step, langchain_message_type_name(message))
    print(extract_text_content(getattr(message, "content", "")) if content is None else content)


def _flush_pending_tool_logs(state: dict[str, Any]) -> None:
    sys.stdout.flush()
    while state["pending_tool_logs"]:
        tool_log = state["pending_tool_logs"].pop(0)
        logger.info("Tool call | name=%s | input=%s", tool_log["name"], tool_log["input"])
        logger.info("Tool result | name=%s | output=%s", tool_log["name"], tool_log["output"])


def _print_basic_agent_step_message(message: Any, state: dict[str, Any], *, stream: bool = False) -> None:
    if message is None:
        return

    type_name = langchain_message_type_name(message)
    tool_calls = langchain_message_tool_calls(message)
    if type_name == "SystemMessage":
        if not state["printed_system_message"]:
            _print_step_message("STEP 1 - SYSTEM MESSAGE", message)
            state["printed_system_message"] = True
        return
    if type_name == "HumanMessage":
        if not state["printed_human_message"]:
            _print_step_message("STEP 2 - USER -> LLM", message)
            state["printed_human_message"] = True
        return
    if "AIMessage" in type_name and tool_calls:
        _print_step_header("STEP 3 - LLM -> AGENT TOOL INSTRUCTIONS", "AIMessage.tool_calls")
        print(tool_calls)
        return
    if type_name == "ToolMessage":
        _flush_pending_tool_logs(state)
        _print_step_message("STEP 4 - TOOL -> LLM", message)
        return
    if type_name == "AIMessageChunk":
        delta = extract_text_content(getattr(message, "content", ""))
        if delta and not state["printed_final_header"]:
            _print_step_header("STEP 5 - LLM FINAL MESSAGE", "AIMessage")
            state["printed_final_header"] = True
        if delta:
            print(delta, end="")
        state["final_text"] += delta
        return
    if "AIMessage" in type_name:
        text = extract_text_content(getattr(message, "content", ""))
        if stream:
            if not state["printed_final_header"]:
                _print_step_message("STEP 5 - LLM FINAL MESSAGE", message, text)
                state["printed_final_header"] = True
            if not state["final_text"]:
                state["final_text"] = text
        else:
            _print_step_message("STEP 5 - LLM FINAL MESSAGE", message, text)
            state["printed_final_header"] = True
            state["final_text"] = text


def _create_basic_agent_step_state(pending_tool_logs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "final_text": "",
        "pending_tool_logs": pending_tool_logs if pending_tool_logs is not None else [],
        "printed_final_header": False,
        "printed_system_message": False,
        "printed_human_message": False,
    }


def _print_basic_agent_step_output(
    messages: list[Any] | None = None,
    stream_chunks: Any | None = None,
    state: dict[str, Any] | None = None,
) -> str:
    state = state or _create_basic_agent_step_state()
    for message in messages or []:
        _print_basic_agent_step_message(message, state)
    if stream_chunks is not None:
        for chunk, _text in stream_chunks:
            _print_basic_agent_step_message(chunk, state, stream=True)
    print("\n")
    return state["final_text"].strip()


class LangChainAgentManager:
    framework = "LangChain Basic Agent"
    prints_own_output = True
    tool_names = ["generate_uuid"]
    tool_trigger_help = "Tools are triggered automatically. Ask for a UUID/ticket ID to trigger generate_uuid."

    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 1000):
        config = get_identifier_mappings().get(model)
        self.model_identifier = model
        self.provider = config.provider if config else "openai"
        self.model = config.model if config else model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.pending_tool_logs: list[dict[str, Any]] = []
        self.agent = create_agent(
            model=create_langchain_model(
                self.provider,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ),
            tools=[_make_generate_uuid_tool(self.pending_tool_logs)],
            system_prompt=SYSTEM_PROMPT,
        )

    def ask_question(self, topic: str, stream: bool = False) -> Dict[str, Any]:
        try:
            initial_messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=topic)]
            human = initial_messages[1]

            self.pending_tool_logs.clear()
            state = _create_basic_agent_step_state(self.pending_tool_logs)
            for message in initial_messages:
                _print_basic_agent_step_message(message, state)

            if stream:
                normalized_final_text = _print_basic_agent_step_output(
                    stream_chunks=stream_message_chunks(self.agent, human),
                    state=state,
                )
            else:
                response = self.agent.invoke({"messages": [human]})
                response_messages = response.get("messages", []) if isinstance(response, dict) else []
                normalized_final_text = _print_basic_agent_step_output(response_messages, state=state)
            return {
                "success": True,
                "final_text": normalized_final_text,
                "finalText": normalized_final_text,
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
    startup_model = select_startup_model(ALL_MODEL_IDENTIFIERS, args.mode, args.model_identifier)
    manager = LangChainAgentManager(model=startup_model, temperature=args.temperature, max_tokens=args.max_tokens)
    run_mode(manager, args.mode, args.host, args.port, args.stream)


if __name__ == "__main__":
    main()
