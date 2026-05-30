"""Agent Memory Gateway - LlamaIndex with persistent session memory."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

CHAPTER_4_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "chapter_4"))
CHAPTER_4_LLAMAINDEX = os.path.join(CHAPTER_4_ROOT, "llamaindex")

sys.path.append(CHAPTER_4_ROOT)
sys.path.append(CHAPTER_4_LLAMAINDEX)

from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import Memory
from llama_index.core.storage.chat_store import SimpleChatStore

from agent_app import LlamaIndexLLMManager as Chapter4LlamaIndexManager
from utils import get_default_model, interactive_cli


class LlamaIndexLLMManager(Chapter4LlamaIndexManager):
    """Chapter 5 manager with file-backed memory as the default mode."""

    def __init__(self, memory_enabled: bool = True):
        self.memory_enabled = memory_enabled
        self.memories: Dict[str, Memory] = {}
        self.chat_engines: Dict[Tuple[str, str], SimpleChatEngine] = {}
        self.chat_stores: Dict[str, SimpleChatStore] = {}
        super().__init__()
        self.framework = "LlamaIndex Memory+Persistence"

    def _session_file_path(self, session_id: str) -> Path:
        sessions_dir = Path(__file__).resolve().parent / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        return sessions_dir / f"{session_id}.json"

    def _session_store_key(self, session_id: str) -> str:
        return session_id

    def _get_chat_store(self, session_id: str) -> SimpleChatStore:
        if session_id in self.chat_stores:
            return self.chat_stores[session_id]

        path = self._session_file_path(session_id)
        chat_store = SimpleChatStore.from_persist_path(str(path)) if path.exists() else SimpleChatStore()
        self.chat_stores[session_id] = chat_store
        return chat_store

    def _get_memory(self, session_id: str) -> Memory:
        if session_id not in self.memories:
            chat_store = self._get_chat_store(session_id)
            store_key = self._session_store_key(session_id)
            self.memories[session_id] = Memory.from_defaults(
                session_id=store_key,
                chat_history=list(chat_store.get_messages(store_key)),
            )
        return self.memories[session_id]

    def _persist_memory(self, session_id: str):
        store_key = self._session_store_key(session_id)
        messages = list(self._get_memory(session_id).get_all())
        chat_store = self._get_chat_store(session_id)
        chat_store.set_messages(store_key, messages)
        chat_store.persist(persist_path=str(self._session_file_path(session_id)))

    def _get_chat_engine(
        self, provider: str, session_id: str, temperature: float, max_tokens: int
    ) -> SimpleChatEngine:
        key = (provider, session_id)
        if key not in self.chat_engines:
            client = self._create_client(provider, temperature=temperature, max_tokens=max_tokens)
            memory = self._get_memory(session_id)
            self.chat_engines[key] = SimpleChatEngine.from_defaults(llm=client, memory=memory)
        return self.chat_engines[key]

    @staticmethod
    def _memory_messages(memory: Memory) -> List[ChatMessage]:
        if hasattr(memory, "get_all"):
            return list(memory.get_all())
        if hasattr(memory, "get_messages"):
            return list(memory.get_messages())
        return list(getattr(memory, "chat_history", []) or [])

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

        model = get_default_model(provider)

        try:
            chat_engine = self._get_chat_engine(provider, session_id, temperature, max_tokens)
            result = chat_engine.chat(prompt)
            response_text = getattr(result, "response", None) or self._extract_text(result)
            self._persist_memory(session_id)

            return {
                "success": True,
                "provider": provider,
                "model": model,
                "prompt": prompt,
                "response": response_text,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "session_id": session_id,
            }
        except Exception as exc:
            return {
                "success": False,
                "provider": provider,
                "model": model,
                "prompt": prompt,
                "error": str(exc),
                "response": None,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "session_id": session_id,
            }

    def get_history(self, provider: str, session_id: str = "default") -> Dict:
        turns = [
            {"role": str(msg.role), "content": msg.content}
            for msg in self._memory_messages(self._get_memory(session_id))
        ]
        return {
            "provider": provider,
            "session_id": session_id,
            "turns": turns,
            "count": len(turns),
        }

    def reset_memory(self, provider: str = None, session_id: str = None) -> Dict:
        removed = []

        if session_id:
            self.memories.pop(session_id, None)
            self.chat_stores.pop(session_id, None)
            for key in list(self.chat_engines.keys()):
                if key[1] == session_id:
                    self.chat_engines.pop(key, None)
            removed.append(session_id)
        else:
            self.memories.clear()
            self.chat_engines.clear()
            self.chat_stores.clear()
            removed = ["ALL"]

        if removed == ["ALL"]:
            for path in (Path(__file__).resolve().parent / "sessions").glob("*.json"):
                path.unlink(missing_ok=True)
        else:
            for session in removed:
                self._session_file_path(session).unlink(missing_ok=True)

        return {"status": "cleared", "removed_sessions": removed}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        from web import run_web_server

        run_web_server(lambda: LlamaIndexLLMManager(memory_enabled=True))
    else:
        interactive_cli(LlamaIndexLLMManager(memory_enabled=True))


if __name__ == "__main__":
    main()
