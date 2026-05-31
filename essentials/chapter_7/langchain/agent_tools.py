"""Agent Tools - LangChain with Wikipedia."""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

sys.path.append(REPO_ROOT)

from essentials.chapter_6.langchain.agent_memory_retrieval import LangChainLLMManager as Chapter6LangChainManager
from essentials.chapter_7.tools import build_tools_prompt, run_tool
from shared.essentials.utils import get_default_model, interactive_cli

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
- DEFAULT BEHAVIOR: For most user queries, first call "get_wikipedia_evidence_pack" with arguments {{"query": <user topic>}}.
  This is especially important for topics involving factual claims, names, dates, definitions, history, science, places, people, events, or "what is/why/how" questions.
- Use an empty tool_calls array only if the user request is purely creative writing, brainstorming fiction, or personal preference with no need for sources.
- If you call a tool, include it in tool_calls with output set to null and keep final_answer short until you see the tool output.
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


class LangChainLLMManager(Chapter6LangChainManager):
    """Chapter 7 LangChain manager with retrieval-aware tool orchestration."""

    def __init__(self, memory_enabled: bool = True, retrieval_k: int = 4):
        super().__init__(memory_enabled=memory_enabled, retrieval_k=retrieval_k)
        self.framework = "LangChain Tools"

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
        result = client.invoke(self._build_messages(prompt))
        text = self._extract_text(provider, result)
        payload = self._extract_json_object(text)
        tool_calls = payload.get("tool_calls")
        if not isinstance(tool_calls, list):
            payload["tool_calls"] = []
        return payload, result

    def _load_retrieval_messages(self, session_id: str) -> List:
        """Load persisted history using Chapter 6's session-only memory contract."""
        return self._get_history(session_id).messages

    def _build_retrieval_context(self, topic: str, session_id: str) -> Tuple[str, Dict[str, object]]:
        messages: List = self._load_retrieval_messages(session_id) if self.retrieval_memory_enabled else []
        retrieved = self._select_retrieved_messages(topic, messages) if self.retrieval_memory_enabled else []

        retrieved_context = "\n".join(f"[{item['role']}] {item['content']}" for item in retrieved)
        retrieval_augmented_topic = (
            f"Relevant memory snippets:\n{retrieved_context}\n\nCurrent user topic: {topic}" if retrieved_context else topic
        )

        full_history_context = "\n".join(
            f"[{getattr(msg, 'type', 'unknown')}] {getattr(msg, 'content', '')}" for msg in messages
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


    def _safe_reset_corrupt_history(self, session_id: str) -> None:
        session_path = self._session_file_path(session_id)
        if not session_path.exists():
            return

        backup_path = session_path.with_suffix(f"{session_path.suffix}.corrupt")
        try:
            if backup_path.exists():
                backup_path.unlink()
            session_path.replace(backup_path)
        except Exception:
            # If backup move fails, continue by overwriting in place.
            pass

        session_path.write_text("[]", encoding="utf-8")

    def _get_history(self, session_id: str):
        history = super()._get_history(session_id)
        try:
            _ = history.messages
            return history
        except Exception as exc:
            message = str(exc)
            known_corruption_signals = (
                "string indices must be integers",
                "Expecting value",
                "JSON",
            )
            if not any(signal in message for signal in known_corruption_signals):
                raise

            self.histories.pop(session_id, None)
            self._safe_reset_corrupt_history(session_id)
            repaired_history = super()._get_history(session_id)
            _ = repaired_history.messages
            return repaired_history

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
        retrieval_topic, retrieval_metadata = self._build_retrieval_context(topic, session_id)
        retrieval_prompt = template.format(topic=retrieval_topic, tools=build_tools_prompt())

        try:
            first_step, first_result = self._invoke_json_step(provider, retrieval_prompt, temperature=temperature, max_tokens=max_tokens)
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

            response_metadata = getattr(first_result, "response_metadata", None)
            usage_metadata = getattr(first_result, "usage_metadata", None)
            token_usage = self._extract_token_usage(response_metadata, usage_metadata)

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
                            "response_metadata": response_metadata,
                            "usage_metadata": usage_metadata,
                        },
                        raw_response,
                    ),
                    "retrieval": retrieval_metadata,
                },
            }

            if self.retrieval_memory_enabled:
                history = self._get_history(session_id)
                history.add_user_message(topic)
                history.add_ai_message(raw_response)

            result = {
                "success": True,
                "provider": provider,
                "model": model,
                "prompt": retrieval_prompt,
                "response": payload,
                "raw_answer": final_answer,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "session_id": session_id,
                "response_metadata": response_metadata,
                "usage_metadata": usage_metadata,
                "token_usage": token_usage,
            }
            return {k: v for k, v in result.items() if v is not None}
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
        from shared.essentials.web import run_web_server

        run_web_server(lambda: LangChainLLMManager(memory_enabled=True))
    else:
        interactive_cli(LangChainLLMManager(memory_enabled=True))


if __name__ == "__main__":
    main()
