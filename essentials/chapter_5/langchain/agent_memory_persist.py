"""LLM Memory Gateway - LangChain with persistent session memory."""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

sys.path.append(REPO_ROOT)

if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    from shared.utils import print_cli_help

    print_cli_help(sys.argv[0])
    sys.exit(0)


from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from essentials.chapter_4.langchain.agent_app import LangChainLLMManager as Chapter4LangChainManager
from shared.essentials.utils import interactive_cli


class LangChainLLMManager(Chapter4LangChainManager):
    """Chapter 5 manager with file-backed memory as the default mode."""

    def __init__(self, memory_enabled: bool = True):
        self.memory_enabled = memory_enabled
        self.chains: Dict[Tuple[str, str], RunnableWithMessageHistory] = {}
        self.histories: Dict[str, FileChatMessageHistory] = {}
        super().__init__()
        self.framework = "LangChain Memory+Persistence"

    def _session_file_path(self, session_id: str) -> Path:
        sessions_dir = Path(__file__).resolve().parent / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        return sessions_dir / f"{session_id}.json"

    def _get_history(self, session_id: str) -> FileChatMessageHistory:
        if session_id not in self.histories:
            self.histories[session_id] = FileChatMessageHistory(file_path=str(self._session_file_path(session_id)))
        return self.histories[session_id]

    def _test_provider(self, provider: str):
        if self.memory_enabled:
            self._get_chain(provider, "test-session", temperature=0.7, max_tokens=1000)
        else:
            super()._test_provider(provider)

    def _get_chain(self, provider: str, session_id: str, temperature: float, max_tokens: int):
        key = (provider, session_id)
        if key not in self.chains:
            model = self._create_model(provider, temperature, max_tokens)
            prompt = ChatPromptTemplate.from_messages(
                [
                    MessagesPlaceholder("history"),
                    ("human", "{input}"),
                ]
            )
            self.chains[key] = RunnableWithMessageHistory(
                prompt | model,
                get_session_history=lambda _: self._get_history(session_id),
                input_messages_key="input",
                history_messages_key="history",
            )
        return self.chains[key]

    @staticmethod
    def _extract_token_usage(response_metadata, usage_metadata):
        if isinstance(response_metadata, dict):
            usage = response_metadata.get("token_usage") if isinstance(response_metadata.get("token_usage"), dict) else response_metadata
            compact = {
                "completion_tokens": usage.get("completion_tokens"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
            compact = {k: v for k, v in compact.items() if v is not None}
            if compact:
                return compact

        if isinstance(usage_metadata, dict):
            compact = {
                "completion_tokens": usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens"),
                "prompt_tokens": usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens"),
                "total_tokens": usage_metadata.get("total_tokens"),
            }
            compact = {k: v for k, v in compact.items() if v is not None}
            if compact:
                return compact

        return None

    def ask_question(
        self,
        topic: str,
        provider: str = None,
        template: str = "{topic}",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        session_id: str = "default",
    ) -> Dict:
        if not self.memory_enabled:
            return super().ask_question(topic, provider, template, max_tokens, temperature)

        prompt = template.format(topic=topic)
        provider = self.resolve_model_identifier(provider)
        if not provider:
            return {
                "success": False,
                "error": "No providers available",
                "provider": "none",
                "model": "none",
                "prompt": prompt,
                "response": None,
            }

        model_config = self.resolve_model_config(provider)

        try:
            result = self._get_chain(provider, session_id, temperature, max_tokens).invoke(
                {"input": prompt},
                config={"configurable": {"session_id": session_id}},
            )
            response_metadata = getattr(result, "response_metadata", None)
            usage_metadata = getattr(result, "usage_metadata", None)
            message_id = getattr(result, "id", None)
            finish_reason = None
            if isinstance(response_metadata, dict):
                finish_reason = response_metadata.get("finish_reason") or response_metadata.get("stop_reason")
            token_usage = self._extract_token_usage(response_metadata, usage_metadata)

            payload = {
                "success": True,
                "provider": model_config.provider,
                "model": model_config.model,
                "model_identifier": model_config.name,
                "prompt": prompt,
                "response": self._extract_text(provider, result),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "session_id": session_id,
                "response_metadata": response_metadata,
                "usage_metadata": usage_metadata,
                "token_usage": token_usage,
                "id": message_id,
                "finish_reason": finish_reason,
            }
            return {k: v for k, v in payload.items() if v is not None}
        except Exception as exc:
            return {
                "success": False,
                "provider": model_config.provider,
                "model": model_config.model,
                "model_identifier": model_config.name,
                "prompt": prompt,
                "error": str(exc),
                "response": None,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "session_id": session_id,
            }

    def get_history(self, provider: str, session_id: str) -> Dict:
        messages = self._get_history(session_id).messages
        turns = []
        for msg in messages:
            if not hasattr(msg, "content"):
                continue
            response_metadata = getattr(msg, "response_metadata", None)
            usage_metadata = getattr(msg, "usage_metadata", None)
            additional_kwargs = getattr(msg, "additional_kwargs", None) or {}
            if response_metadata is None:
                response_metadata = additional_kwargs.get("response_metadata")
            if usage_metadata is None:
                usage_metadata = additional_kwargs.get("usage_metadata")
            token_usage = self._extract_token_usage(response_metadata, usage_metadata)
            turn = {"role": msg.type, "content": msg.content}
            if token_usage is not None:
                turn["token_usage"] = token_usage
            turns.append(turn)
        return {
            "provider": provider,
            "session_id": session_id,
            "turns": turns,
            "count": len(messages),
        }

    def reset_memory(self, provider: str = None, session_id: str = None) -> Dict:
        removed = []

        if session_id:
            self.histories.pop(session_id, None)
            for key in list(self.chains.keys()):
                if key[1] == session_id:
                    self.chains.pop(key, None)
            removed.append(session_id)
        else:
            self.chains.clear()
            self.histories.clear()
            removed = ["ALL"]

        if removed == ["ALL"]:
            for path in (Path(__file__).resolve().parent / "sessions").glob("*.json"):
                path.unlink(missing_ok=True)
        else:
            for session in removed:
                self._session_file_path(session).unlink(missing_ok=True)

        return {"status": "cleared", "removed_sessions": removed}


def main():
    args = sys.argv[1:]
    if "web" in args:
        from shared.essentials.web import run_web_server

        run_web_server(lambda: LangChainLLMManager(memory_enabled=True))
    else:
        interactive_cli(LangChainLLMManager(memory_enabled=True))


if __name__ == "__main__":
    main()
