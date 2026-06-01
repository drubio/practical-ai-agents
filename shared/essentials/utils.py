"""Essentials-book utility helpers built on top of repository shared utilities."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from shared.utils import (
    format_filename,
    get_all_provider_names,
    get_default_model as _base_get_default_model,
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
from shared.llm_models import ALL_MODEL_IDENTIFIERS, get_identifier_mappings


_SELECTED_MODEL_IDENTIFIER: ContextVar[Optional[str]] = ContextVar("essentials_selected_model_identifier", default=None)


def get_default_model(provider: str) -> str:
    """Return the active model for a provider, honoring per-query overrides."""
    selected_identifier = _SELECTED_MODEL_IDENTIFIER.get()
    if selected_identifier:
        config = get_identifier_mappings().get(selected_identifier)
        if config and config.provider == provider:
            return config.model
    return _base_get_default_model(provider)


def get_model_identifier_config(model_identifier: Optional[str]):
    if model_identifier is None:
        return None
    return get_identifier_mappings().get(str(model_identifier).strip())


def model_identifiers_for_providers(providers: Iterable[str] | None = None) -> List[str]:
    provider_set = set(providers) if providers is not None else None
    mappings = get_identifier_mappings()
    return [identifier for identifier in ALL_MODEL_IDENTIFIERS if identifier in mappings and (provider_set is None or mappings[identifier].provider in provider_set)]


MODEL_PROVIDER_PREFIXES = (
    ("google_genai_", "Google"),
    ("anthropic_", "Anthropic"),
    ("openai_", "OpenAI"),
    ("xai_", "xAI"),
    ("deepseek_", "DeepSeek"),
)


def _provider_and_model_name(model_identifier: str) -> tuple[str, str]:
    for prefix, provider_name in MODEL_PROVIDER_PREFIXES:
        if model_identifier.startswith(prefix):
            return provider_name, model_identifier[len(prefix):]
    provider_name, _, model_name = model_identifier.partition("_")
    return (provider_name.title() if provider_name else "Other"), (model_name or model_identifier)


def compact_model_selection_lines(model_identifiers: Sequence[str]) -> List[str]:
    lines: List[str] = []
    current_provider: Optional[str] = None
    current_options: List[str] = []

    def flush_current() -> None:
        nonlocal current_options
        if current_provider and current_options:
            lines.append(f"{current_provider}: " + " | ".join(current_options))
        current_options = []

    for index, model_identifier in enumerate(model_identifiers, start=1):
        provider_name, model_name = _provider_and_model_name(model_identifier)
        if provider_name != current_provider or len(current_options) == 3:
            flush_current()
            current_provider = provider_name
        default_suffix = " [default]" if index == 1 else ""
        current_options.append(f"{index}. {model_name}{default_suffix}")
    flush_current()
    return lines


@contextmanager
def selected_model_context(model_identifier: Optional[str]) -> Iterator[None]:
    token = _SELECTED_MODEL_IDENTIFIER.set(model_identifier)
    try:
        yield
    finally:
        _SELECTED_MODEL_IDENTIFIER.reset(token)


def _model_option_label(model_identifier: str) -> str:
    config = get_identifier_mappings()[model_identifier]
    return f"{config.model} ({config.tier}; {model_identifier})"


def select_model_identifier(model_identifiers: Sequence[str], prompt: str = "Select a model:") -> str:
    choice_idx = get_user_choice([_model_option_label(identifier) for identifier in model_identifiers], prompt)
    return model_identifiers[choice_idx]


def select_provider_model_identifier(providers: Sequence[str]) -> str:
    provider_idx = get_user_choice(
        [f"{get_display_name(provider)} ({len(model_identifiers_for_providers([provider]))} models)" for provider in providers],
        "Select a provider:",
    )
    provider = providers[provider_idx]
    provider_models = model_identifiers_for_providers([provider])
    return select_model_identifier(provider_models, f"Select a {get_display_name(provider)} model:")


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
    #print("\nAvailable providers:")
    #for provider_name in available_providers:
    #    print(f"- {format_provider_summary(provider_name)}")

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
    model_config = get_identifier_mappings()[model_identifier]
    provider = model_config.provider
    selected_provider_details = get_default_model_details(provider)
    print(
        "\nUsing model: "
        f"{selected_provider_details['display_name']} "
        f"(provider: {model_config.provider}, "
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
            with selected_model_context(model_identifier):
                response = manager.ask_question(**kwargs)
            display_provider_response(provider, response, manager.framework)

    print(f"\nThank you for using the {manager.framework} Agent Application!")


# Backwards-compatible alias for chapter code that imports BaseLLMManager.
BaseLLMManager = EssentialsLLMManager
