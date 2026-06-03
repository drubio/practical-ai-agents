"""Essentials-book utility helpers built on top of repository shared utilities."""

from __future__ import annotations

from typing import Dict, List, Optional

from shared.utils import (
    display_provider_response,
    get_selected_model_details,
    get_user_parameters,
    model_identifiers_for_providers,
    parse_structured_json_response,
    print_initialization_status,
    select_provider_model_identifier,
)
from shared.llm_models import (
    get_all_providers,
    get_api_key,
    get_display_name,
    get_model_config,
    get_provider_model_identifier,
    resolve_model_identifier,
    sort_providers_by_display_order,
)


class EssentialsLLMManager:
    """Base class for Essentials LLM managers with provider initialization."""

    def __init__(self, framework_name: str):
        self.framework = framework_name
        self.initialization_messages = {}
        self._check_providers()

    def _check_providers(self):
        for provider in get_all_providers():
            if get_api_key(provider):
                try:
                    self._test_provider(provider)
                    self.initialization_messages[provider] = "✓ Initialized successfully"
                except Exception as exc:
                    self.initialization_messages[provider] = f"✗ Failed: {exc}"
            else:
                self.initialization_messages[provider] = "✗ API key not found"

    def _test_provider(self, provider: str):
        raise NotImplementedError("Subclasses must implement _test_provider")

    def get_available_providers(self) -> List[str]:
        return [provider for provider, status in self.initialization_messages.items() if status.startswith("✓")]

    def display_initialization_status(self) -> None:
        print_initialization_status(self.framework, self.initialization_messages)

    def resolve_model_identifier(self, selection: Optional[str]) -> str | None:
        return resolve_model_identifier(selection, self.get_available_providers())

    def resolve_model_config(self, selection: Optional[str]):
        selected_model = self.resolve_model_identifier(selection)
        return get_model_config(selected_model) if selected_model else None

    def provider_model_identifier(self, provider: str) -> str:
        return get_provider_model_identifier(provider)

    def ask_question(
        self,
        topic: str,
        provider: str = None,
        template: str = "{topic}",
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> Dict:
        raise NotImplementedError("Subclasses must implement ask_question")


def manager_supports_interactive_memory(manager: EssentialsLLMManager) -> bool:
    full_memory_supported = (
        getattr(manager, "memory_enabled", False) is True
        and hasattr(manager, "ask_question")
        and hasattr(manager, "get_history")
        and hasattr(manager, "reset_memory")
    )
    retrieval_memory_supported = (
        getattr(manager, "retrieval_memory_enabled", False) is True
        and hasattr(manager, "ask_question")
        and hasattr(manager, "get_history")
        and hasattr(manager, "reset_memory")
    )
    return full_memory_supported or retrieval_memory_supported



def interactive_basic_question_loop(
    manager,
    provider: Optional[str] = None,
    model_identifier: Optional[str] = None,
    *,
    ask_question=None,
    prompt: str = "\nAsk a question (or 'exit'): ",
) -> None:
    """Run the shared minimal question/exit loop used by non-memory CLIs."""
    while True:
        user_input = input(prompt).strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting.")
            break
        if not user_input:
            print("Input cannot be empty. Please try again.")
            continue
        response = ask_question(user_input) if ask_question else manager.ask_question(topic=user_input, provider=provider)
        if not getattr(manager, "prints_own_output", False):
            display_provider_response(provider or getattr(manager, "provider", "unknown"), response, manager.framework)


def interactive_cli(manager: EssentialsLLMManager, model_identifier: Optional[str] = None):
    print("=" * 60)
    print(f"Agent Application - {manager.framework} Framework")
    print("=" * 60)

    manager.display_initialization_status()
    available_providers = manager.get_available_providers()
    if not available_providers:
        print("No providers available. Check your .env file.")
        return

    temperature, max_tokens = get_user_parameters()
    print(f"\nUsing temperature: {temperature}, max tokens: {max_tokens}")
    available_providers = sort_providers_by_display_order(available_providers)
    available_model_identifiers = model_identifiers_for_providers(available_providers)
    if not available_model_identifiers:
        print("No models available for initialized providers.")
        return

    memory_supported = manager_supports_interactive_memory(manager)

    if model_identifier:
        if model_identifier not in available_model_identifiers:
            print(f"Model identifier '{model_identifier}' is not available for initialized providers.")
            return
    else:
        model_identifier = select_provider_model_identifier(available_providers)
    model_config = get_model_config(model_identifier)
    provider = model_identifier
    provider_name = model_config.provider
    selected_model_details = get_selected_model_details(model_identifier)
    print(
        "\nUsing model: "
        f"{selected_model_details['display_name']} "
        f"(provider: {provider_name}, "
        f"model: {model_config.model} / "
        f"{model_config.name} / "
        f"{model_config.tier})"
    )
    session_id = "default"
    if memory_supported:
        session_id_input = input("Enter memory session ID (default: 'default'): ").strip()
        if session_id_input:
            session_id = session_id_input
        print(f"Using memory session: {session_id}")
    print("\n" + "=" * 50)
    print(f"{manager.framework.upper()} INTERACTIVE MODE - {get_display_name(provider_name).upper()}")
    print("=" * 50)
    if not memory_supported:
        interactive_basic_question_loop(
            manager,
            provider=model_identifier,
            model_identifier=model_identifier,
            ask_question=lambda user_input: manager.ask_question(
                topic=user_input,
                provider=model_identifier,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
        print(f"\nThank you for using the {manager.framework} Agent Application!")
        return

    while True:
        user_input = input("\nAsk a question (or 'history', 'clear', 'exit'): " if memory_supported else "\nAsk a question (or 'exit'): ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting.")
            break
        if not user_input:
            print("Input cannot be empty. Please try again.")
            continue
        if user_input.lower() == "history":
            if memory_supported and hasattr(manager, "get_history"):
                history = manager.get_history(model_identifier, session_id)
                print(f"\n🧠 Memory for {get_display_name(provider_name)} (session: {session_id}):")
                for turn in history["turns"]:
                    print(f"[{turn['role'].capitalize()}] {turn['content']}")
                if not history["turns"]:
                    print("No memory yet.")
            else:
                print("⚠️ This manager does not support memory history.")
        elif user_input.lower() == "clear":
            if memory_supported and hasattr(manager, "reset_memory"):
                manager.reset_memory(model_identifier, session_id)
                print(f"✅ Memory cleared for session '{session_id}'")
            else:
                print("⚠️ This manager does not support memory reset.")
        else:
            kwargs = {"topic": user_input, "provider": model_identifier, "temperature": temperature, "max_tokens": max_tokens}
            if memory_supported:
                kwargs["session_id"] = session_id
            response = manager.ask_question(**kwargs)
            display_provider_response(model_identifier, response, manager.framework)

    print(f"\nThank you for using the {manager.framework} Agent Application!")


# Backwards-compatible alias for chapter code that imports BaseLLMManager.
BaseLLMManager = EssentialsLLMManager
