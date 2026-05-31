"""Agent Memory Structured Gateway - LlamaIndex with structured JSON responses."""

import ast
import os
import sys
from typing import Dict, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

sys.path.append(REPO_ROOT)

from essentials.chapter_5.llamaindex.agent_memory_persist import LlamaIndexLLMManager as Chapter5LlamaIndexManager
from shared.essentials.utils import interactive_cli, parse_structured_json_response


STRUCTURED_TEMPLATE = """
Given the topic below, provide:

1. A direct factual answer (if possible)
2. A summary of what the question is about
3. Relevant keywords
4. A distilled answer (short phrase or value-only form of the answer)

Respond in the following JSON format:
{{
  "answer": "...",
  "summary": "...",
  "keywords": ["...", "..."],
  "distilled": "...",
  "metadata": {{
    "confidence": "high|medium|low",
    "notes": "optional extra context"
  }}
}}

Topic: {topic}
""".strip()


class LlamaIndexLLMManager(Chapter5LlamaIndexManager):
    """Chapter 5 structured manager layered on persistent memory."""

    def __init__(self, memory_enabled: bool = True):
        super().__init__(memory_enabled=memory_enabled)
        self.framework = "LlamaIndex Structured Output"

    @staticmethod
    def _extract_dict_from_text(raw_response: str, key: str) -> Optional[Dict]:
        marker = f"{key}="
        idx = raw_response.find(marker)
        if idx == -1:
            return None

        start = raw_response.find("{", idx)
        if start == -1:
            return None

        depth = 0
        in_string = False
        quote = ""
        escape = False

        for pos in range(start, len(raw_response)):
            ch = raw_response[pos]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_string = False
                continue

            if ch in ('"', "'"):
                in_string = True
                quote = ch
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return ast.literal_eval(raw_response[start : pos + 1])
                    except Exception:
                        return None

        return None

    def _build_metadata(self, result: Dict, raw_response: str) -> Dict:
        response_metadata = result.get("response_metadata")
        usage_metadata = result.get("usage_metadata")

        if response_metadata is None:
            response_metadata = self._extract_dict_from_text(raw_response, "response_metadata")
        if usage_metadata is None:
            usage_metadata = self._extract_dict_from_text(raw_response, "usage_metadata")

        token_usage = None
        if isinstance(response_metadata, dict) and isinstance(response_metadata.get("token_usage"), dict):
            usage = response_metadata.get("token_usage")
            token_usage = {
                "completion_tokens": usage.get("completion_tokens"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        elif isinstance(usage_metadata, dict):
            token_usage = {
                "completion_tokens": usage_metadata.get("output_tokens"),
                "prompt_tokens": usage_metadata.get("input_tokens"),
                "total_tokens": usage_metadata.get("total_tokens"),
            }

        if isinstance(token_usage, dict):
            token_usage = {k: v for k, v in token_usage.items() if v is not None}
            if not token_usage:
                token_usage = None

        metadata = {
            "provider": result.get("provider"),
            "model": result.get("model"),
            "framework": self.framework,
            "session_id": result.get("session_id"),
            "temperature": result.get("temperature"),
            "max_tokens": result.get("max_tokens"),
            "raw_response_chars": len(raw_response),
            "token_usage": token_usage,
        }
        return {k: v for k, v in metadata.items() if v is not None}

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
        result = super().ask_question(topic, provider, effective_template, max_tokens, temperature, session_id)

        if not result.get("success"):
            return result

        original_response = result.get("response")
        raw_response = original_response if isinstance(original_response, str) else str(original_response)
        metadata_source = original_response if isinstance(original_response, str) else str(original_response)
        try:
            parsed = parse_structured_json_response(raw_response)
            parsed["metadata"] = {
                **(parsed.get("metadata") or {}),
                **self._build_metadata(result, metadata_source),
            }
        except Exception as exc:
            return {
                **result,
                "success": False,
                "error": f"Failed to parse structured JSON response: {exc}",
                "response": None,
                "raw_response": raw_response,
            }

        return {
            **result,
            "response": parsed,
            "raw_answer": parsed.get("answer", raw_response),
        }


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        from shared.essentials.web import run_web_server

        run_web_server(lambda: LlamaIndexLLMManager(memory_enabled=True))
    else:
        interactive_cli(LlamaIndexLLMManager(memory_enabled=True))


if __name__ == "__main__":
    main()
