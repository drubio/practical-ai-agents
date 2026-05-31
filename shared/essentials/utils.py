"""Essentials-book utility helpers built on top of repository shared utilities."""

from __future__ import annotations

from typing import Dict, List

from shared.utils import (
    format_filename,
    get_all_provider_names,
    get_default_model,
    get_non_empty_input,
    normalize_response_text,
    parse_structured_json_response,
    save_response_to_file,
    display_provider_response,
    format_provider_summary,
    get_all_providers,
    get_api_key,
    get_default_model_details,
    get_display_name,
    get_user_choice,
    get_user_parameters,
    print_initialization_status,
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


def interactive_cli(manager: EssentialsLLMManager):
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
    print("\nAvailable providers:")
    for provider in available_providers:
        print(f"- {format_provider_summary(provider)}")

    memory_supported = manager_supports_interactive_memory(manager)

    choice_idx = get_user_choice([get_display_name(p) for p in available_providers], "Select a provider:")
    provider = available_providers[choice_idx]
    selected_provider_details = get_default_model_details(provider)
    print(
        "\nUsing provider: "
        f"{selected_provider_details['display_name']} "
        f"(provider: {selected_provider_details['canonical_provider']}, "
        f"default model: {selected_provider_details['default_model']} / "
        f"{selected_provider_details['default_model_identifier']} / "
        f"{selected_provider_details['default_model_tier']})"
    )
    session_id = "default"
    if memory_supported:
        session_id_input = input("Enter memory session ID (default: 'default'): ").strip()
        if session_id_input:
            session_id = session_id_input
        print(f"Using memory session: {session_id}")
    print("\n" + "=" * 50)
    print(f"{manager.framework.upper()} INTERACTIVE MODE - {get_display_name(provider).upper()}")
    print("=" * 50)
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
                history = manager.get_history(provider, session_id)
                print(f"\n🧠 Memory for {get_display_name(provider)} (session: {session_id}):")
                for turn in history["turns"]:
                    print(f"[{turn['role'].capitalize()}] {turn['content']}")
                if not history["turns"]:
                    print("No memory yet.")
            else:
                print("⚠️ This manager does not support memory history.")
        elif user_input.lower() == "clear":
            if memory_supported and hasattr(manager, "reset_memory"):
                manager.reset_memory(provider, session_id)
                print(f"✅ Memory cleared for session '{session_id}'")
            else:
                print("⚠️ This manager does not support memory reset.")
        else:
            kwargs = {"topic": user_input, "provider": provider, "temperature": temperature, "max_tokens": max_tokens}
            if memory_supported:
                kwargs["session_id"] = session_id
            display_provider_response(provider, manager.ask_question(**kwargs), manager.framework)

    print(f"\nThank you for using the {manager.framework} Agent Application!")


# Backwards-compatible alias for chapter code that imports BaseLLMManager.
BaseLLMManager = EssentialsLLMManager
