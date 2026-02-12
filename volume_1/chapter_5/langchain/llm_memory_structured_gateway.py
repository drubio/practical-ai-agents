"""LLM Memory Structured Gateway - LangChain with structured JSON responses."""

import json
import os
import sys
from typing import Dict, Optional

CHAPTER_4_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "chapter_4"))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(CHAPTER_4_ROOT)
sys.path.append(CURRENT_DIR)

from llm_memory_persist_gateway import LangChainLLMManager as Chapter5LangChainManager
from utils import interactive_cli


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
  "distilled": "..."
}}

Topic: {topic}
""".strip()


class LangChainLLMManager(Chapter5LangChainManager):
    """Chapter 5 structured manager layered on persistent memory."""

    def __init__(self, memory_enabled: bool = True):
        super().__init__(memory_enabled=memory_enabled)
        self.framework = "LangChain+Structured"

    @staticmethod
    def _parse_structured_response(raw: str):
        content = raw.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return json.loads(content.strip())

    def ask_question(
        self,
        topic: str,
        provider: Optional[str] = None,
        template: str = STRUCTURED_TEMPLATE,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        session_id: str = "default",
    ) -> Dict:
        result = super().ask_question(topic, provider, template, max_tokens, temperature, session_id)

        if not result.get("success"):
            return result

        raw_response = str(result.get("response", ""))
        try:
            parsed = self._parse_structured_response(raw_response)
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
        from web import run_web_server

        run_web_server(lambda: LangChainLLMManager(memory_enabled=True))
    else:
        interactive_cli(LangChainLLMManager(memory_enabled=True))


if __name__ == "__main__":
    main()
