"""Chapter 6 retrieval memory gateway for LangChain."""

import os
import re
import sys
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_community.retrievers import BM25Retriever

CHAPTER_4_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "chapter_4"))
CHAPTER_5_LANGCHAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "chapter_5", "langchain"))

sys.path.append(CHAPTER_4_ROOT)
sys.path.append(CHAPTER_5_LANGCHAIN)

from llm_memory_structured_gateway import STRUCTURED_TEMPLATE, LangChainLLMManager as Chapter5StructuredManager
from utils import get_default_model, interactive_cli, parse_structured_json_response


class LangChainLLMManager(Chapter5StructuredManager):
    """Chapter 6 manager that retrieves only relevant memory before prompting the LLM."""

    def __init__(self, memory_enabled: bool = True, retrieval_k: int = 4):
        # Keep Chapter 5's inherited full-history chain disabled because this manager
        # builds its own prompt with retrieved snippets instead of replaying all turns.
        super().__init__(memory_enabled=False)
        # Store Chapter 6 retrieval-memory behavior separately for clarity.
        self.retrieval_memory_enabled = memory_enabled
        self.retrieval_k = max(1, retrieval_k)
        self.framework = "LangChain+Memory+Retrieval"

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return int(count_tokens_approximately([HumanMessage(content=text)]))

    @staticmethod
    def _tokenize(text: str) -> set:
        stop_words = {
            "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "how",
            "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
            "we", "what", "when", "where", "which", "who", "why", "with", "you",
        }
        tokens = set(re.findall(r"[a-zA-Z0-9_]+", str(text).lower()))
        return {token for token in tokens if len(token) > 2 and token not in stop_words}

    def _overlap_score(self, query_tokens: set, content: str) -> int:
        content_tokens = self._tokenize(content)
        if not query_tokens or not content_tokens:
            return 0
        return len(query_tokens.intersection(content_tokens))

    def _select_retrieved_messages(self, topic: str, messages: List) -> List[Dict[str, str]]:
        query_tokens = self._tokenize(topic)
        if not query_tokens:
            return []

        documents = []
        for idx, msg in enumerate(messages):
            content = str(getattr(msg, "content", "") or "")
            if not content:
                continue
            role = getattr(msg, "type", "unknown")
            documents.append(Document(page_content=content, metadata={"idx": idx, "role": role}))

        if documents:
            retriever = BM25Retriever.from_documents(documents, k=self.retrieval_k, include_score=True)
            retrieved_docs = retriever.invoke(topic)
            strong_docs = []
            for doc in retrieved_docs:
                bm25 = float(doc.metadata.get("bm25Score", 0))
                if bm25 <= 0:
                    continue
                overlap = self._overlap_score(query_tokens, str(doc.page_content))
                overlap_ratio = overlap / len(query_tokens)
                if overlap >= 2 or overlap_ratio >= 0.4:
                    strong_docs.append(doc)

            if strong_docs:
                top = sorted(strong_docs, key=lambda doc: doc.metadata.get("idx", 0))
                return [
                    {
                        "role": str(doc.metadata.get("role", "unknown")),
                        "content": str(doc.page_content),
                        "relevance_score": float(doc.metadata.get("bm25Score", 0)),
                    }
                    for doc in top
                ]

        fallback = []
        for idx, msg in enumerate(messages):
            content = str(getattr(msg, "content", "") or "")
            if not content:
                continue
            overlap = self._overlap_score(query_tokens, content)
            overlap_ratio = overlap / len(query_tokens)
            if overlap >= 2 or overlap_ratio >= 0.4:
                fallback.append((overlap, idx, msg))

        if not fallback:
            return []

        fallback.sort(key=lambda item: (-item[0], item[1]))
        top = sorted(fallback[: self.retrieval_k], key=lambda item: item[1])
        return [
            {
                "role": getattr(msg, "type", "unknown"),
                "content": str(getattr(msg, "content", "")),
                "relevance_score": score,
            }
            for score, _, msg in top
        ]

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

        messages = self._get_history(provider, session_id).messages if self.retrieval_memory_enabled else []
        retrieved = self._select_retrieved_messages(topic, messages) if self.retrieval_memory_enabled else []

        retrieved_context = "\n".join(f"[{item['role']}] {item['content']}" for item in retrieved)
        retrieval_augmented_topic = (
            f"Relevant memory snippets:\n{retrieved_context}\n\nCurrent user topic: {topic}"
            if retrieved_context
            else topic
        )
        retrieval_prompt = effective_template.format(topic=retrieval_augmented_topic)

        full_history_context = "\n".join(
            f"[{getattr(msg, 'type', 'unknown')}] {getattr(msg, 'content', '')}" for msg in messages
        )
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
            result = client.invoke(self._build_messages(retrieval_prompt))
            raw_response = self._extract_text(provider, result)
            response_metadata = getattr(result, "response_metadata", None)
            usage_metadata = getattr(result, "usage_metadata", None)
            token_usage = self._extract_token_usage(response_metadata, usage_metadata)

            result_payload = {
                "provider": provider,
                "model": get_default_model(provider),
                "session_id": session_id,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_metadata": response_metadata,
                "usage_metadata": usage_metadata,
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
                history = self._get_history(provider, session_id)
                history.add_user_message(topic)
                history.add_ai_message(raw_response)

            payload = {
                "success": True,
                "provider": provider,
                "model": get_default_model(provider),
                "prompt": retrieval_prompt,
                "response": parsed,
                "raw_answer": parsed.get("answer", raw_response),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "session_id": session_id,
                "response_metadata": response_metadata,
                "usage_metadata": usage_metadata,
                "token_usage": token_usage,
            }
            return {k: v for k, v in payload.items() if v is not None}
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

        run_web_server(lambda: LangChainLLMManager(memory_enabled=True))
    else:
        interactive_cli(LangChainLLMManager(memory_enabled=True))


if __name__ == "__main__":
    main()
