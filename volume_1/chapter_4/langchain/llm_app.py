import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Optional
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from utils import get_api_key, get_default_model, BaseLLMManager, interactive_cli


class LangChainLLMManager(BaseLLMManager):
    """LangChain implementation with reusable hooks for chapter extensions."""

    def __init__(self):
        super().__init__("LangChain")

    def _test_provider(self, provider: str):
        self._create_client(provider, temperature=0.7, max_tokens=1000)

    def _create_client(self, provider: str, temperature: float, max_tokens: int):
        if provider == "anthropic":
            return ChatAnthropic(
                api_key=get_api_key(provider),
                model=get_default_model(provider),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if provider == "openai":
            return ChatOpenAI(
                api_key=get_api_key(provider),
                model=get_default_model(provider),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if provider == "google":
            return ChatGoogleGenerativeAI(
                api_key=get_api_key(provider),
                model=get_default_model(provider),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if provider == "xai":
            return ChatOpenAI(
                api_key=get_api_key(provider),
                base_url="https://api.x.ai/v1",
                model=get_default_model(provider),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        raise ValueError(f"Unsupported provider: {provider}")

    def _resolve_provider(self, provider: Optional[str]):
        available = self.get_available_providers()
        if provider and provider in available:
            return provider
        if available:
            return available[0]
        return None

    def _build_messages(self, prompt: str):
        return [
            SystemMessage(content="You are a helpful AI assistant."),
            HumanMessage(content=prompt),
        ]

    def _extract_text(self, provider: str, result) -> str:
        if provider == "google" and hasattr(result, "text"):
            return str(result.text)
        return str(result.content)

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

        model = get_default_model(provider)

        try:
            client = self._create_client(provider, temperature=temperature, max_tokens=max_tokens)
            result = client.invoke(self._build_messages(prompt))
            return {
                "success": True,
                "provider": provider,
                "model": model,
                "prompt": prompt,
                "response": self._extract_text(provider, result),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        except Exception as e:
            return {
                "success": False,
                "provider": provider,
                "model": model,
                "prompt": prompt,
                "error": str(e),
                "response": None,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        try:
            from web import run_web_server
            run_web_server(LangChainLLMManager)
        except ImportError:
            print("Error: web.py not found or FastAPI not installed.")
            print("Install FastAPI: pip install fastapi uvicorn")
            sys.exit(1)
    else:
        manager = LangChainLLMManager()
        interactive_cli(manager)


if __name__ == "__main__":
    main()
