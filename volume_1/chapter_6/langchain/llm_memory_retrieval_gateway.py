"""Chapter 6 retrieval memory gateway for LangChain."""

import os
import re
import sys
from typing import Dict, List, Optional

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
        return max(1, int(len(text) / 4))

    @staticmethod
    def _tokenize(text: str) -> set:
        return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))

    def _score_message(self, query_tokens: set, content: str) -> int:
        content_tokens = self._tokenize(content)
        if not query_tokens or not content_tokens:
            return 0
        return len(query_tokens.intersection(content_tokens))

    def _select_retrieved_messages(self, topic: str, messages: List) -> List[Dict[str, str]]:
        query_tokens = self._tokenize(topic)
        scored = []
        for idx, msg in enumerate(messages):
            content = getattr(msg, "content", "")
            if not content:
                continue
            score = self._score_message(query_tokens, str(content))
            if score > 0:
                scored.append((score, idx, msg))

        if not scored:
            return []

        # Keep most relevant messages, then restore chronological order.
        scored.sort(key=lambda item: (-item[0], item[1]))
        top = sorted(scored[: self.retrieval_k], key=lambda item: item[1])
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
