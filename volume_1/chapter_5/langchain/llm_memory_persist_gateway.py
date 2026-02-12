"""LLM Memory Gateway - LangChain with persistent session memory."""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

CHAPTER_4_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "chapter_4"))
CHAPTER_4_LANGCHAIN = os.path.join(CHAPTER_4_ROOT, "langchain")

sys.path.append(CHAPTER_4_ROOT)
sys.path.append(CHAPTER_4_LANGCHAIN)

from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from llm_gateway import LangChainLLMManager as Chapter4LangChainManager
from utils import get_default_model, interactive_cli


class LangChainLLMManager(Chapter4LangChainManager):
    """Chapter 5 manager with file-backed memory as the default mode."""

    def __init__(self, memory_enabled: bool = True):
        self.memory_enabled = memory_enabled
        self.chains: Dict[Tuple[str, str], RunnableWithMessageHistory] = {}
        self.histories: Dict[Tuple[str, str], FileChatMessageHistory] = {}
        super().__init__()
        self.framework = "LangChain+History"

    def _session_file_path(self, provider: str, session_id: str) -> Path:
        sessions_dir = Path(__file__).resolve().parent / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        return sessions_dir / f"{provider}__{session_id}.json"

    def _get_history(self, provider: str, session_id: str) -> FileChatMessageHistory:
        key = (provider, session_id)
        if key not in self.histories:
            self.histories[key] = FileChatMessageHistory(file_path=str(self._session_file_path(provider, session_id)))
        return self.histories[key]

    def _test_provider(self, provider: str):
        if self.memory_enabled:
            self._get_chain(provider, "test-session", temperature=0.7, max_tokens=1000)
        else:
            super()._test_provider(provider)

    def _get_chain(self, provider: str, session_id: str, temperature: float, max_tokens: int):
        key = (provider, session_id)
        if key not in self.chains:
            client = self._create_client(provider, temperature, max_tokens)
            prompt = ChatPromptTemplate.from_messages(
                [
                    MessagesPlaceholder("history"),
                    ("human", "{input}"),
                ]
            )
            self.chains[key] = RunnableWithMessageHistory(
                prompt | client,
                get_session_history=lambda _: self._get_history(provider, session_id),
                input_messages_key="input",
                history_messages_key="history",
            )
        return self.chains[key]

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
        provider = self._resolve_provider(provider)
        if not provider:
            return {
                "success": False,
                "error": "No providers available",
                "provider": "none",
                "model": "none",
                "prompt": prompt,
                "response": None,
            }

        try:
            result = self._get_chain(provider, session_id, temperature, max_tokens).invoke(
                {"input": prompt},
                config={"configurable": {"session_id": session_id}},
            )
            return {
                "success": True,
                "provider": provider,
                "model": get_default_model(provider),
                "prompt": prompt,
                "response": self._extract_text(provider, result),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "session_id": session_id,
            }
        except Exception as exc:
            return {
                "success": False,
                "provider": provider,
                "model": get_default_model(provider),
                "prompt": prompt,
                "error": str(exc),
                "response": None,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "session_id": session_id,
            }

    def get_history(self, provider: str, session_id: str) -> Dict:
        messages = self._get_history(provider, session_id).messages
        return {
            "provider": provider,
            "session_id": session_id,
            "turns": [{"role": msg.type, "content": msg.content} for msg in messages if hasattr(msg, "content")],
            "count": len(messages),
        }

    def reset_memory(self, provider: str = None, session_id: str = None) -> Dict:
        removed = []
        if provider and session_id:
            key = (provider, session_id)
            self.chains.pop(key, None)
            self.histories.pop(key, None)
            removed.append(key)
        elif provider:
            for key in list(self.histories.keys()):
                if key[0] == provider:
                    self.chains.pop(key, None)
                    self.histories.pop(key, None)
                    removed.append(key)
        elif session_id:
            for key in list(self.histories.keys()):
                if key[1] == session_id:
                    self.chains.pop(key, None)
                    self.histories.pop(key, None)
                    removed.append(key)
        else:
            self.chains.clear()
            self.histories.clear()
            removed = ["ALL"]

        if removed == ["ALL"]:
            for path in (Path(__file__).resolve().parent / "sessions").glob("*.json"):
                path.unlink(missing_ok=True)
        else:
            for key in removed:
                if isinstance(key, tuple):
                    self._session_file_path(*key).unlink(missing_ok=True)

        return {"status": "cleared", "removed_sessions": removed}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        from web import run_web_server

        run_web_server(lambda: LangChainLLMManager(memory_enabled=True))
    else:
        interactive_cli(LangChainLLMManager(memory_enabled=True))


if __name__ == "__main__":
    main()
