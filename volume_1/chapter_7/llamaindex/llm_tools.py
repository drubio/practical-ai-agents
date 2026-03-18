"""LLM Tools - LlamaIndex with Wikipedia."""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

from llama_index.core.llms import ChatMessage

CHAPTER_4_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "chapter_4"))
CHAPTER_6_LLAMAINDEX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "chapter_6", "llamaindex"))
CHAPTER_7_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(CHAPTER_4_ROOT)
sys.path.append(CHAPTER_6_LLAMAINDEX)
sys.path.append(CHAPTER_7_ROOT)

from llm_memory_retrieval import LlamaIndexLLMManager as Chapter6LlamaIndexManager
from tools import build_tools_prompt, run_tool
from utils import get_default_model, interactive_cli

TOOLS_TEMPLATE = """
You are a helpful assistant with access to external tools.

Available tools:
{tools}

Return strict JSON:
{{
  "tool_calls": [{{"name": "tool_name", "arguments": {{"arg": "value"}}, "output": null}}],
  "final_answer": "string"
}}

Rules:
- If no tool is needed, set tool_calls to an empty array.
- If a tool is needed, add one object per tool call to tool_calls and keep final_answer short.
- Return JSON only.

User topic: {topic}
""".strip()

FOLLOW_UP_TEMPLATE = """
You already requested a tool and now have the result.

Original user topic: {topic}
Tool call: {tool_call}
Tool output: {tool_output}

Return strict JSON:
{{
  "tool_calls": [{{"name": "tool_name", "arguments": {{"arg": "value"}}, "output": "serialized tool output"}}],
  "final_answer": "final response for the user"
}}
""".strip()


class LlamaIndexLLMManager(Chapter6LlamaIndexManager):
    """Chapter 7 LlamaIndex manager with retrieval-aware tool orchestration."""

    def __init__(self, memory_enabled: bool = True, retrieval_k: int = 4):
        super().__init__(memory_enabled=memory_enabled, retrieval_k=retrieval_k)
        self.framework = "LlamaIndex Tools"

    @staticmethod
    def _extract_json_object(raw: str) -> Dict:
        text = raw.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                raise
            return json.loads(match.group(0))

    def _invoke_json_step(self, provider: str, prompt: str, temperature: float, max_tokens: int) -> Tuple[Dict, object]:
        client = self._create_client(provider, temperature=temperature, max_tokens=max_tokens)
        result = client.chat([ChatMessage(role="user", content=prompt)])
        text = self._extract_text(result)
        payload = self._extract_json_object(text)
        tool_calls = payload.get("tool_calls")
        if not isinstance(tool_calls, list):
            payload["tool_calls"] = []
        return payload, result

    def _build_retrieval_context(self, provider: str, topic: str, session_id: str) -> Tuple[str, Dict[str, object]]:
        memory = self._get_memory(provider, session_id) if self.retrieval_memory_enabled else None
        messages = self._memory_messages(memory) if memory else []
        retrieved = self._select_retrieved_messages(topic, messages) if self.retrieval_memory_enabled else []

        retrieved_context = "\n".join(f"[{item['role']}] {item['content']}" for item in retrieved)
        retrieval_augmented_topic = (
            f"Relevant memory snippets:\n{retrieved_context}\n\nCurrent user topic: {topic}" if retrieved_context else topic
        )

        full_history_context = "\n".join(
            f"[{getattr(msg, 'role', 'unknown')}] {getattr(msg, 'content', '')}" for msg in messages
        )
        prompt_without_retrieval = (
            f"{full_history_context}\n\nCurrent user topic: {topic}" if full_history_context else topic
        )

        tokens_with_retrieval = self._estimate_tokens(retrieval_augmented_topic)
        tokens_without_retrieval = self._estimate_tokens(prompt_without_retrieval)
        estimated_saved = max(0, tokens_without_retrieval - tokens_with_retrieval)
        reduction_percent = round((estimated_saved / tokens_without_retrieval) * 100, 2) if tokens_without_retrieval else 0.0

        retrieval_metadata = {
            "history_messages_available": len(messages),
            "retrieved_messages_count": len(retrieved),
            "retrieved_messages": retrieved,
            "tokens_with_memory_retrieval": tokens_with_retrieval,
            "tokens_without_memory_retrieval": tokens_without_retrieval,
            "estimated_tokens_saved": estimated_saved,
            "estimated_token_reduction_percent": reduction_percent,
        }
        return retrieval_augmented_topic, retrieval_metadata

    @staticmethod
    def _resolve_tools_template(template: Optional[str]) -> str:
        candidate = (template or "").strip()
        if not candidate or candidate == "{topic}" or "{tools}" not in candidate:
            return TOOLS_TEMPLATE
        return template

    def ask_question(
        self,
        topic: str,
        provider: Optional[str] = None,
        template: str = TOOLS_TEMPLATE,
        max_tokens: int = 1000,
        temperature: float = 0.2,
        session_id: str = "default",
    ) -> Dict:
        template = self._resolve_tools_template(template)
        provider = self._resolve_provider(provider)
        prompt = template.format(topic=topic, tools=build_tools_prompt())

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
        retrieval_topic, retrieval_metadata = self._build_retrieval_context(provider, topic, session_id)
        retrieval_prompt = template.format(topic=retrieval_topic, tools=build_tools_prompt())

        try:
            first_step, _ = self._invoke_json_step(provider, retrieval_prompt, temperature=temperature, max_tokens=max_tokens)
            tool_calls = first_step.get("tool_calls") or []
            final_answer = str(first_step.get("final_answer", "")).strip()

            executed_tool_calls = [dict(tool_call) for tool_call in tool_calls]
            if executed_tool_calls:
                for tool_call in executed_tool_calls:
                    tool_output = run_tool(tool_call["name"], tool_call.get("arguments") or {})
                    tool_call["output"] = tool_output

                follow_up_prompt = FOLLOW_UP_TEMPLATE.format(
                    topic=topic,
                    tool_call=json.dumps(executed_tool_calls, ensure_ascii=False),
                    tool_output=json.dumps(executed_tool_calls, ensure_ascii=False),
                )
                second_step, _ = self._invoke_json_step(provider, follow_up_prompt, temperature=temperature, max_tokens=max_tokens)
                final_answer = str(second_step.get("final_answer", final_answer)).strip() or final_answer

            raw_response = json.dumps(
                {
                    "tool_calls": executed_tool_calls,
                    "final_answer": final_answer,
                },
                ensure_ascii=False,
            )

            payload = {
                "tool_calls": executed_tool_calls,
                "final_answer": final_answer,
                "metadata": {
                    **self._build_metadata(
                        {
                            "provider": provider,
                            "model": model,
                            "session_id": session_id,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                        raw_response,
                    ),
                    "retrieval": retrieval_metadata,
                },
            }

            if self.retrieval_memory_enabled:
                self._append_to_memory(provider, session_id, "user", topic)
                self._append_to_memory(provider, session_id, "assistant", raw_response)
                self._persist_memory(provider, session_id)

            return {
                "success": True,
                "provider": provider,
                "model": model,
                "prompt": retrieval_prompt,
                "response": payload,
                "raw_answer": final_answer,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "session_id": session_id,
            }
        except Exception as exc:
            return {
                "success": False,
                "provider": provider,
                "model": model,
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
