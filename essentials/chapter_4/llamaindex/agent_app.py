"""LLM application to chat with multiple LLMs - LlamaIndex Python framework implementation."""

import os
import sys
from typing import Dict, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from llama_index.core.llms import ChatMessage
from shared.utils import BaseLLMManager, interactive_cli
from shared.utils import create_llamaindex_model


class LlamaIndexLLMManager(BaseLLMManager):
    """LlamaIndex implementation with reusable hooks for chapter extensions."""

    def __init__(self, stream: bool = False):
        self.stream = stream
        self.prints_own_output = stream
        super().__init__("LlamaIndex")

    def _test_provider(self, provider: str):
        self._create_model(self.provider_model_identifier(provider), temperature=0.7, max_tokens=1000)

    def _create_model(self, selected_model: str, temperature: float, max_tokens: int):
        return create_llamaindex_model(
            selected_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _create_client(self, provider: str, temperature: float, max_tokens: int):
        return self._create_model(provider, temperature=temperature, max_tokens=max_tokens)

    def _resolve_provider(self, provider: Optional[str]):
        return self.resolve_model_identifier(provider)

    @staticmethod
    def _extract_stream_delta(chunk) -> str:
        for attr in ("delta", "content_delta"):
            value = getattr(chunk, attr, None)
            if isinstance(value, str) and value:
                return value
        content = getattr(getattr(chunk, "message", None), "content", None)
        if isinstance(content, str):
            return content
        return str(chunk) if chunk is not None else ""

    @staticmethod
    def _extract_text(result) -> str:
        content = getattr(getattr(result, "message", None), "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif hasattr(item, "text"):
                    parts.append(str(item.text))
            if parts:
                return "\n".join(parts)
        return str(content if content is not None else result)

    def _stream_model(self, model, messages) -> str:
        parts = []
        for chunk in model.stream_chat(messages):
            delta = self._extract_stream_delta(chunk)
            if delta:
                print(delta, end="", flush=True)
                parts.append(delta)
        print()
        return "".join(parts)

    def ask_question(
        self,
        topic: str,
        provider: str = None,
        template: str = "{topic}",
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> Dict:
        prompt = template.format(topic=topic)
        model_config = self.resolve_model_config(provider)

        if not model_config:
            return {
                "success": False,
                "error": "No providers available",
                "provider": "none",
                "model": "none",
                "prompt": prompt,
                "response": None,
            }

        try:
            model = self._create_model(model_config.name, temperature=temperature, max_tokens=max_tokens)
            messages = [ChatMessage(role="user", content=prompt)]
            if self.stream:
                response_text = self._stream_model(model, messages)
            else:
                result = model.chat(messages)
                response_text = self._extract_text(result)
            return {
                "success": True,
                "provider": model_config.provider,
                "model": model_config.model,
                "model_identifier": model_config.name,
                "prompt": prompt,
                "response": response_text,
                "temperature": temperature,
                "stream": self.stream,
                "max_tokens": max_tokens,
            }
        except Exception as exc:
            return {
                "success": False,
                "provider": model_config.provider,
                "model": model_config.model,
                "model_identifier": model_config.name,
                "prompt": prompt,
                "error": str(exc),
                "response": None,
                "temperature": temperature,
                "stream": self.stream,
                "max_tokens": max_tokens,
            }


def main():
    args = sys.argv[1:]
    stream = "--stream" in args
    if "web" in args:
        try:
            from shared.essentials.web import run_web_server

            run_web_server(lambda: LlamaIndexLLMManager(stream=stream))
        except ImportError:
            print("Error: shared web API not found or FastAPI not installed.")
            print("Install FastAPI: pip install fastapi uvicorn")
            sys.exit(1)
    else:
        manager = LlamaIndexLLMManager(stream=stream)
        interactive_cli(manager)


if __name__ == "__main__":
    main()
