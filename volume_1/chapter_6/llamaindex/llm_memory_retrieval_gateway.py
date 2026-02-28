"""Chapter 6 retrieval memory gateway for LlamaIndex."""

import os
import re
import sys
from typing import Dict, List, Optional

CHAPTER_4_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "chapter_4"))
CHAPTER_5_LLAMAINDEX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "chapter_5", "llamaindex"))

sys.path.append(CHAPTER_4_ROOT)
sys.path.append(CHAPTER_5_LLAMAINDEX)

from llama_index.core.llms import ChatMessage
from llama_index.core.utils import get_tokenizer
import bm25s

from llm_memory_structured_gateway import STRUCTURED_TEMPLATE, LlamaIndexLLMManager as Chapter5StructuredManager
from utils import get_default_model, interactive_cli, parse_structured_json_response


class LlamaIndexLLMManager(Chapter5StructuredManager):
    """Chapter 6 manager that retrieves only relevant memory before prompting the LLM."""

    _STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "how",
        "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
        "we", "what", "when", "where", "which", "who", "why", "with", "you",
    }

    def __init__(self, memory_enabled: bool = True, retrieval_k: int = 4):
        # Disable inherited full-memory chat engine replay. Chapter 6 builds retrieval prompts directly.
        super().__init__(memory_enabled=False)
        self.retrieval_memory_enabled = memory_enabled
        self.retrieval_k = max(1, retrieval_k)
        self.framework = "LlamaIndex+Memory+Retrieval"
        self._tokenizer = get_tokenizer()

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer(text))

    @classmethod
    def _tokenize_overlap(cls, text: str) -> set:
        tokens = set(re.findall(r"[a-zA-Z0-9_]+", str(text).lower()))
        return {token for token in tokens if len(token) > 2 and token not in cls._STOP_WORDS}

    def _overlap_score(self, query_tokens: set, content_tokens: set) -> int:
        if not query_tokens or not content_tokens:
            return 0
        return len(query_tokens.intersection(content_tokens))

    def _select_retrieved_messages(self, topic: str, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        topic_text = str(topic or "").strip()
        query_tokens = self._tokenize_overlap(topic_text)
        if not topic_text or not query_tokens:
            return []

        message_records = []
        for idx, msg in enumerate(messages):
            content = str(getattr(msg, "content", "") or "")
            if not content:
                continue
            overlap_tokens = self._tokenize_overlap(content)
            if not overlap_tokens:
                continue
            message_records.append(
                {
                    "idx": idx,
                    "role": str(getattr(msg, "role", "unknown")),
                    "content": content,
                    "overlap_tokens": overlap_tokens,
                }
            )

        if message_records:
            corpus = [record["content"] for record in message_records]
            retriever = bm25s.BM25()
            retriever.index(bm25s.tokenize(corpus, stopwords="en"))

            top_n = min(len(message_records), max(self.retrieval_k * 3, self.retrieval_k))
            matches, scores = retriever.retrieve(
                bm25s.tokenize([topic_text], stopwords="en"),
                k=top_n,
            )
            strong = []
            for local_idx, score in zip(matches[0], scores[0]):
                bm25_score = float(score)
                message_idx = int(local_idx)
                if bm25_score <= 0 or message_idx < 0 or message_idx >= len(message_records):
                    continue
                record = message_records[message_idx]
                overlap = self._overlap_score(query_tokens, record["overlap_tokens"])
                overlap_ratio = overlap / len(query_tokens)
                if overlap >= 2 or overlap_ratio >= 0.4:
                    strong.append((bm25_score, record))

            if strong:
                top = sorted(strong, key=lambda item: (-item[0], item[1]["idx"]))[: self.retrieval_k]
                chronological = sorted(top, key=lambda item: item[1]["idx"])
                return [
                    {
                        "role": record["role"],
                        "content": record["content"],
                        "relevance_score": score,
                    }
                    for score, record in chronological
                ]

        fallback = []
        for idx, msg in enumerate(messages):
            content = str(getattr(msg, "content", "") or "")
            if not content:
                continue
            content_tokens = self._tokenize_overlap(content)
            if not content_tokens:
                continue
            score = self._overlap_score(query_tokens, content_tokens)
            overlap_ratio = score / len(query_tokens)
            if score >= 2 or overlap_ratio >= 0.4:
                fallback.append((score, idx, msg))

        if not fallback:
            return []

        fallback.sort(key=lambda item: (-item[0], item[1]))
        top = sorted(fallback[: self.retrieval_k], key=lambda item: item[1])
        return [
            {
                "role": str(getattr(msg, "role", "unknown")),
                "content": str(getattr(msg, "content", "")),
                "relevance_score": score,
            }
            for score, _, msg in top
        ]

    def _append_to_memory(self, provider: str, session_id: str, role: str, content: str):
        memory = self._get_memory(provider, session_id)
        chat_message = ChatMessage(role=role, content=content)

        if hasattr(memory, "put"):
            memory.put(chat_message)
        elif hasattr(memory, "put_messages"):
            memory.put_messages([chat_message])
        elif hasattr(memory, "chat_history") and isinstance(memory.chat_history, list):
            memory.chat_history.append(chat_message)

    def ask_question(
        self,
        topic: str,
        provider: Optional[str] = None,
        template: str = STRUCTURED_TEMPLATE,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        session_id: str = "default",
    ) -> Dict:
        effective_template = STRUCTURED_TEMPLATE if template == "{topic}" else template
        provider = self._resolve_provider(provider)
        base_prompt = effective_template.format(topic=topic)

        if not provider:
            return {
                "success": False,
                "error": "No providers available",
                "provider": "none",
                "model": "none",
                "prompt": base_prompt,
                "response": None,
            }

        memory = self._get_memory(provider, session_id) if self.retrieval_memory_enabled else None
        messages = self._memory_messages(memory) if memory else []
        retrieved = self._select_retrieved_messages(topic, messages) if self.retrieval_memory_enabled else []

        retrieved_context = "\n".join(f"[{item['role']}] {item['content']}" for item in retrieved)
        retrieval_augmented_topic = (
            f"Relevant memory snippets:\n{retrieved_context}\n\nCurrent user topic: {topic}"
            if retrieved_context
            else topic
        )
        retrieval_prompt = effective_template.format(topic=retrieval_augmented_topic)

        full_history_context = "\n".join(f"[{getattr(msg, 'role', 'unknown')}] {getattr(msg, 'content', '')}" for msg in messages)
        prompt_without_retrieval = (
            f"{full_history_context}\n\nCurrent user topic: {topic}" if full_history_context else topic
        )

        tokens_with_retrieval = self._estimate_tokens(retrieval_prompt)
        tokens_without_retrieval = self._estimate_tokens(prompt_without_retrieval)
        estimated_saved = max(0, tokens_without_retrieval - tokens_with_retrieval)
        reduction_percent = (
            round((estimated_saved / tokens_without_retrieval) * 100, 2) if tokens_without_retrieval else 0.0
        )

        try:
            client = self._create_client(provider, temperature=temperature, max_tokens=max_tokens)
            result = client.chat([ChatMessage(role="user", content=retrieval_prompt)])
            raw_response = self._extract_text(result)

            result_payload = {
                "provider": provider,
                "model": get_default_model(provider),
                "session_id": session_id,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            try:
                parsed = parse_structured_json_response(raw_response)
            except Exception:
                parsed = {
                    "answer": raw_response,
                    "summary": "Model returned plain-text output instead of strict JSON.",
                    "keywords": [],
                    "distilled": raw_response,
                    "metadata": {
                        "confidence": "low",
                        "notes": "Structured parser fallback applied for non-JSON response.",
                    },
                }

            parsed["metadata"] = {
                **(parsed.get("metadata") or {}),
                **self._build_metadata(result_payload, raw_response),
                "retrieval": {
                    "history_messages_available": len(messages),
                    "retrieved_messages_count": len(retrieved),
                    "retrieved_messages": retrieved,
                    "tokens_with_memory_retrieval": tokens_with_retrieval,
                    "tokens_without_memory_retrieval": tokens_without_retrieval,
                    "estimated_tokens_saved": estimated_saved,
                    "estimated_token_reduction_percent": reduction_percent,
                },
            }

            if self.retrieval_memory_enabled:
                self._append_to_memory(provider, session_id, "user", topic)
                self._append_to_memory(provider, session_id, "assistant", raw_response)
                self._persist_memory(provider, session_id)

            return {
                "success": True,
                "provider": provider,
                "model": get_default_model(provider),
                "prompt": retrieval_prompt,
                "response": parsed,
                "raw_answer": parsed.get("answer", raw_response),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "session_id": session_id,
            }
        except Exception as exc:
            return {
                "success": False,
                "provider": provider,
                "model": get_default_model(provider),
                "prompt": retrieval_prompt,
                "error": str(exc),
                "response": None,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "session_id": session_id,
            }


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        from web import run_web_server

        run_web_server(lambda: LlamaIndexLLMManager(memory_enabled=True))
    else:
        interactive_cli(LlamaIndexLLMManager(memory_enabled=True))


if __name__ == "__main__":
    main()
