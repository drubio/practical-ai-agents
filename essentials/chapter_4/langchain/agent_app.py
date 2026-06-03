"""LLM application to chat with multiple LLMs - LangChain Python framework implementation."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from typing import Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from shared.utils import BaseLLMManager, interactive_cli
from shared.utils import create_langchain_model


class LangChainLLMManager(BaseLLMManager):
    """LangChain implementation with reusable hooks for chapter extensions."""

    def __init__(self, stream: bool = False):
        self.stream = stream
        self.prints_own_output = stream
        super().__init__("LangChain")

    def _test_provider(self, provider: str):
        self._create_model(self.provider_model_identifier(provider), temperature=0.7, max_tokens=1000)

    def _create_model(self, selected_model: str, temperature: float, max_tokens: int):
        return create_langchain_model(
            selected_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _build_messages(self, prompt: str):
        return [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content=prompt),
        ]

    def _extract_text(self, provider: str, result) -> str:
        if provider == "google" and hasattr(result, "text"):
            return str(result.text)
        return str(result.content)

    @staticmethod
    def _extract_stream_delta(chunk) -> str:
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        return str(content) if content else ""

    def _stream_model(self, model, messages) -> str:
        parts = []
        for chunk in model.stream(messages):
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
            messages = self._build_messages(prompt)
            if self.stream:
                response_text = self._stream_model(model, messages)
            else:
                result = model.invoke(messages)
                response_text = self._extract_text(model_config.provider, result)
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
        except Exception as e:
            return {
                "success": False,
                "provider": model_config.provider,
                "model": model_config.model,
                "model_identifier": model_config.name,
                "prompt": prompt,
                "error": str(e),
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
            run_web_server(lambda: LangChainLLMManager(stream=stream))
        except ImportError:
            print("Error: shared web API not found or FastAPI not installed.")
            print("Install FastAPI: pip install fastapi uvicorn")
            sys.exit(1)
    else:
        manager = LangChainLLMManager(stream=stream)
        interactive_cli(manager)


if __name__ == "__main__":
    main()
