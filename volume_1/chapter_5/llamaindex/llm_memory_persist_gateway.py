"""LLM Memory Gateway - LlamaIndex with persistent session memory."""

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
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore

from llm_gateway import LlamaIndexLLMManager as Chapter4LlamaIndexManager
from utils import get_default_model, interactive_cli


class LlamaIndexLLMManager(Chapter4LlamaIndexManager):
    """Chapter 5 manager with file-backed memory as the default mode."""

    def __init__(self, memory_enabled: bool = True):
        self.memory_enabled = memory_enabled
        self.memories: Dict[Tuple[str, str], ChatMemoryBuffer] = {}
        self.chat_engines: Dict[Tuple[str, str], SimpleChatEngine] = {}
        self.chat_stores: Dict[Tuple[str, str], SimpleChatStore] = {}
        super().__init__()
        self.framework = "LlamaIndex+History"

    def _session_file_path(self, provider: str, session_id: str) -> Path:
        sessions_dir = Path(__file__).resolve().parent / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        return sessions_dir / f"{provider}__{session_id}.json"

    def _session_store_key(self, provider: str, session_id: str) -> str:
        return f"{provider}__{session_id}"

    def _get_chat_store(self, provider: str, session_id: str) -> SimpleChatStore:
        key = (provider, session_id)
        if key in self.chat_stores:
            return self.chat_stores[key]

        path = self._session_file_path(provider, session_id)
        chat_store = SimpleChatStore.from_persist_path(str(path)) if path.exists() else SimpleChatStore()
        self.chat_stores[key] = chat_store
        return chat_store

    def _get_memory(self, provider: str, session_id: str) -> ChatMemoryBuffer:
        key = (provider, session_id)
        if key not in self.memories:
            chat_store = self._get_chat_store(provider, session_id)
            self.memories[key] = ChatMemoryBuffer.from_defaults(
                chat_store=chat_store,
                chat_store_key=self._session_store_key(provider, session_id),
            )
        return self.memories[key]

    def _persist_memory(self, provider: str, session_id: str):
        chat_store = self._get_chat_store(provider, session_id)
        chat_store.persist(persist_path=str(self._session_file_path(provider, session_id)))

    def _get_chat_engine(
        self, provider: str, session_id: str, temperature: float, max_tokens: int
    ) -> SimpleChatEngine:
        key = (provider, session_id)
        if key not in self.chat_engines:
            client = self._create_client(provider, temperature=temperature, max_tokens=max_tokens)
            memory = self._get_memory(provider, session_id)
            self.chat_engines[key] = SimpleChatEngine.from_defaults(llm=client, memory=memory)
        return self.chat_engines[key]

    @staticmethod
    def _memory_messages(memory: ChatMemoryBuffer) -> List[ChatMessage]:
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
            self._persist_memory(provider, session_id)

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
            for msg in self._memory_messages(self._get_memory(provider, session_id))
        ]
        return {
            "provider": provider,
            "session_id": session_id,
            "turns": turns,
            "count": len(turns),
        }

    def reset_memory(self, provider: str = None, session_id: str = None) -> Dict:
        removed = []

        if provider and session_id:
            key = (provider, session_id)
            self.memories.pop(key, None)
            self.chat_engines.pop(key, None)
            self.chat_stores.pop(key, None)
            removed.append(key)
        elif provider:
            for key in list(self.memories.keys()):
                if key[0] == provider:
                    self.memories.pop(key, None)
                    self.chat_engines.pop(key, None)
                    self.chat_stores.pop(key, None)
                    removed.append(key)
        elif session_id:
            for key in list(self.memories.keys()):
                if key[1] == session_id:
                    self.memories.pop(key, None)
                    self.chat_engines.pop(key, None)
                    self.chat_stores.pop(key, None)
                    removed.append(key)
        else:
            self.memories.clear()
            self.chat_engines.clear()
            self.chat_stores.clear()
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

        run_web_server(lambda: LlamaIndexLLMManager(memory_enabled=True))
    else:
        interactive_cli(LlamaIndexLLMManager(memory_enabled=True))


if __name__ == "__main__":
    main()
