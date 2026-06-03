"""LLM application to chat with multiple LLMs - LlamaIndex Python framework implementation."""

import os
import sys
from typing import Dict, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from llama_index.core.llms import ChatMessage
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike

from shared.essentials.utils import BaseLLMManager, get_api_key, interactive_cli
from shared.llm_models import resolve_model_config


class LlamaIndexLLMManager(BaseLLMManager):
    """LlamaIndex implementation with reusable hooks for chapter extensions."""

    def __init__(self):
        super().__init__("LlamaIndex")

    def _test_provider(self, provider: str):
        self._create_model(self.provider_model_identifier(provider), temperature=0.7, max_tokens=1000)

    def _create_model(self, selected_model: str, temperature: float, max_tokens: int):
        config = resolve_model_config(selected_model)
        provider = config.provider
        if provider == "anthropic":
            return Anthropic(
                api_key=get_api_key(provider),
                model=config.model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if provider == "openai":
            return OpenAI(
                api_key=get_api_key(provider),
                model=config.model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if provider == "google":
            return GoogleGenAI(
                api_key=get_api_key(provider),
                model=config.model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if provider == "xai":
            return OpenAILike(
                api_key=get_api_key(provider),
                api_base="https://api.x.ai/v1",
                model=config.model,
                is_chat_model=True,
                is_function_calling_model=False,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if provider == "deepseek":
            return OpenAILike(
                api_key=get_api_key(provider),
                api_base="https://api.deepseek.com",
                model=config.model,
                is_chat_model=True,
                is_function_calling_model=False,
                temperature=temperature,
                max_tokens=max_tokens,
            )        
        raise ValueError(f"Unsupported provider: {provider}")

    def _create_client(self, provider: str, temperature: float, max_tokens: int):
        return self._create_model(provider, temperature=temperature, max_tokens=max_tokens)

    def _resolve_provider(self, provider: Optional[str]):
        return self.resolve_model_identifier(provider)

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

    def ask_question(
        self,
        topic: str,
        provider: str = None,
        template: str = "{topic}",
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> Dict:
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

        model_config = resolve_model_config(provider)

        try:
            model = self._create_model(provider, temperature=temperature, max_tokens=max_tokens)
            result = model.chat([ChatMessage(role="user", content=prompt)])
            return {
                "success": True,
                "provider": model_config.provider,
                "model": model_config.model,
                "model_identifier": model_config.name,
                "prompt": prompt,
                "response": self._extract_text(result),
                "temperature": temperature,
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
                "max_tokens": max_tokens,
            }


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        try:
            from shared.essentials.web import run_web_server

            run_web_server(LlamaIndexLLMManager)
        except ImportError:
            print("Error: shared web API not found or FastAPI not installed.")
            print("Install FastAPI: pip install fastapi uvicorn")
            sys.exit(1)
    else:
        manager = LlamaIndexLLMManager()
        interactive_cli(manager)


if __name__ == "__main__":
    main()
