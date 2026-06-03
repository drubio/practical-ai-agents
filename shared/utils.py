"""Shared utility helpers reused across chapter implementations."""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from dotenv import load_dotenv

from shared.llm_models import (
    get_all_providers,
    get_api_key,
    get_display_name,
    get_model_config,
    get_provider_model_identifier,
    resolve_model_config,
    resolve_model_identifier,
    sort_providers_by_display_order,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / "shared" / ".env")


class BaseLLMManager:
    """Reusable base class for chapter LLM managers with provider initialization."""

    def __init__(self, framework_name: str):
        self.framework = framework_name
        self.initialization_messages = {}
        self._check_providers()

    def _check_providers(self) -> None:
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
        return [
            provider
            for provider, status in self.initialization_messages.items()
            if status.startswith("✓")
        ]

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


def manager_supports_interactive_memory(manager: BaseLLMManager) -> bool:
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
        response = (
            ask_question(user_input)
            if ask_question
            else manager.ask_question(topic=user_input, provider=provider)
        )
        if not getattr(manager, "prints_own_output", False):
            display_provider_response(
                provider or getattr(manager, "provider", "unknown"),
                response,
                manager.framework,
            )


def interactive_cli(manager: BaseLLMManager, model_identifier: Optional[str] = None) -> None:
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
            print(
                f"Model identifier '{model_identifier}' is not available for initialized providers."
            )
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
    print(
        f"{manager.framework.upper()} INTERACTIVE MODE - "
        f"{get_display_name(provider_name).upper()}"
    )
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
        user_input = input("\nAsk a question (or 'history', 'clear', 'exit'): ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting.")
            break
        if not user_input:
            print("Input cannot be empty. Please try again.")
            continue
        if user_input.lower() == "history":
            if hasattr(manager, "get_history"):
                history = manager.get_history(model_identifier, session_id)
                print(
                    f"\n🧠 Memory for {get_display_name(provider_name)} "
                    f"(session: {session_id}):"
                )
                for turn in history["turns"]:
                    print(f"[{turn['role'].capitalize()}] {turn['content']}")
                if not history["turns"]:
                    print("No memory yet.")
            else:
                print("⚠️ This manager does not support memory history.")
        elif user_input.lower() == "clear":
            if hasattr(manager, "reset_memory"):
                manager.reset_memory(model_identifier, session_id)
                print(f"✅ Memory cleared for session '{session_id}'")
            else:
                print("⚠️ This manager does not support memory reset.")
        else:
            kwargs = {
                "topic": user_input,
                "provider": model_identifier,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if memory_supported:
                kwargs["session_id"] = session_id
            response = manager.ask_question(**kwargs)
            display_provider_response(model_identifier, response, manager.framework)

    print(f"\nThank you for using the {manager.framework} Agent Application!")


def normalize_response_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        content_match = re.search(
            r"content=(['\"])((?:\\.|(?!\1).)*)\1\s+additional_kwargs=",
            payload,
            flags=re.DOTALL,
        )
        if content_match:
            raw_quoted_content = f"{content_match.group(1)}{content_match.group(2)}{content_match.group(1)}"
            try:
                decoded = ast.literal_eval(raw_quoted_content)
                if isinstance(decoded, str):
                    return decoded
            except Exception:
                return content_match.group(2)

        try:
            maybe_json = json.loads(payload)
            if isinstance(maybe_json, dict):
                for key in ("answer", "distilled", "content", "text", "message", "summary", "response"):
                    value = maybe_json.get(key)
                    if isinstance(value, str) and value.strip():
                        return value
        except Exception:
            pass
        return payload

    if isinstance(payload, dict):
        for key in ("content", "text", "message", "answer", "final_answer", "distilled", "summary", "response"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value

    if hasattr(payload, "content") and isinstance(payload.content, str):
        return payload.content
    return str(payload)


def get_selected_model_details(selected_model: str) -> Dict[str, str]:
    config = resolve_model_config(selected_model)
    return {
        "provider": config.provider,
        "display_name": get_display_name(config.provider),
        "selected_model": config.model,
        "selected_model_identifier": config.name,
        "selected_model_tier": config.tier,
    }


def create_langchain_model(
    selected_model: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 1000,
):
    """Create a provider-specific LangChain chat model from a model identifier."""
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import ChatOpenAI

    config = resolve_model_config(selected_model)
    provider = config.provider
    model_name = config.model
    if provider == "anthropic":
        return ChatAnthropic(
            api_key=get_api_key(provider),
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "openai":
        return ChatOpenAI(
            api_key=get_api_key(provider),
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "google":
        return ChatGoogleGenerativeAI(
            api_key=get_api_key(provider),
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "xai":
        return ChatOpenAI(
            api_key=get_api_key(provider),
            base_url=os.getenv("XAI_API_BASE", "https://api.x.ai/v1"),
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "deepseek":
        return ChatOpenAI(
            api_key=get_api_key(provider),
            base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def create_llamaindex_model(
    selected_model: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 1000,
):
    """Create a provider-specific LlamaIndex LLM from a model identifier."""
    from llama_index.llms.anthropic import Anthropic
    from llama_index.llms.google_genai import GoogleGenAI
    from llama_index.llms.openai import OpenAI
    from llama_index.llms.openai_like import OpenAILike

    config = resolve_model_config(selected_model)
    provider = config.provider
    model_name = config.model
    if provider == "anthropic":
        return Anthropic(
            api_key=get_api_key(provider),
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "openai":
        return OpenAI(
            api_key=get_api_key(provider),
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "google":
        return GoogleGenAI(
            api_key=get_api_key(provider),
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "xai":
        return OpenAILike(
            api_key=get_api_key(provider),
            api_base=os.getenv("XAI_API_BASE", "https://api.x.ai/v1"),
            model=model_name,
            is_chat_model=True,
            is_function_calling_model=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "deepseek":
        return OpenAILike(
            api_key=get_api_key(provider),
            api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
            model=model_name,
            is_chat_model=True,
            is_function_calling_model=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def parse_structured_json_response(raw: Any) -> Dict[str, Any]:
    if raw is None:
        content = ""
    elif isinstance(raw, str):
        content = raw.strip()
    elif isinstance(raw, dict):
        content = json.dumps(raw, ensure_ascii=False)
    elif hasattr(raw, "content") and isinstance(raw.content, str):
        content = raw.content.strip()
    else:
        content = normalize_response_text(raw).strip()

    if not content:
        raise ValueError("Structured content is empty")

    content_match = re.search(
        r'content=("|\')((?:\\.|(?!\1).)*)\1\s+additional_kwargs=',
        content,
        flags=re.DOTALL,
    )
    if content_match:
        raw_quoted_content = f"{content_match.group(1)}{content_match.group(2)}{content_match.group(1)}"
        try:
            decoded = ast.literal_eval(raw_quoted_content)
            if isinstance(decoded, str):
                content = decoded.strip()
        except Exception:
            pass

    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    if content.startswith("[") and content.endswith("]"):
        try:
            blocks = ast.literal_eval(content)
            if isinstance(blocks, list):
                for block in blocks:
                    if isinstance(block, dict):
                        maybe_text = block.get("text") or block.get("content")
                        if isinstance(maybe_text, str) and maybe_text.strip():
                            return parse_structured_json_response(maybe_text)
        except Exception:
            pass

    start = content.find("{")
    if start == -1:
        raise ValueError("No JSON object found in structured content")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(content)):
        ch = content[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(content[start : idx + 1])
                if isinstance(parsed, dict):
                    return parsed
                break

    raise ValueError("Parsed structured content is not a JSON object")


def get_chapter_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_tool_call(logger: logging.Logger, tool_name: str, fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def wrapper(arg: Any = None) -> Any:
        logger.info("Tool call | name=%s | input=%s", tool_name, arg)
        result = fn(arg)
        logger.info("Tool result | name=%s | output=%s", tool_name, result)
        return result

    wrapper.__name__ = getattr(fn, "__name__", f"{tool_name}_wrapper")
    return wrapper


def get_user_parameters():
    temp_input = input("Temperature (0.0-2.0, default 0.7): ").strip()
    try:
        temperature = float(temp_input) if temp_input else 0.7
        temperature = max(0.0, min(2.0, temperature))
    except ValueError:
        temperature = 0.7
        print(f"Invalid temperature, using default: {temperature}")

    tokens_input = input("Max tokens (default 1000): ").strip()
    try:
        max_tokens = int(tokens_input) if tokens_input else 1000
        max_tokens = max(1, min(4000, max_tokens))
    except ValueError:
        max_tokens = 1000
        print(f"Invalid max tokens, using default: {max_tokens}")

    return temperature, max_tokens


def display_provider_response(provider: str, response: Dict[str, Any], framework: str = "") -> None:
    framework_suffix = f" ({framework})" if framework else ""
    print(f"\n=== {get_display_name(provider)}{framework_suffix} answered: ===")
    config_parts = []
    if response.get("temperature") is not None:
        config_parts.append(f"temp: {response['temperature']}")
    if response.get("max_tokens") is not None:
        config_parts.append(f"max_tokens: {response['max_tokens']}")
    if response.get("model"):
        config_parts.append(f"model: {response['model']}")
    if config_parts:
        print(f"[{', '.join(config_parts)}]")
    if response.get("success"):
        raw = response.get("response", "")
        if isinstance(raw, (dict, list)):
            print(json.dumps(raw, indent=2, ensure_ascii=False))
        elif hasattr(raw, "content"):
            print(str(raw.content))
        else:
            print(normalize_response_text(raw))
    else:
        print(f"Error: {response.get('error', 'Unknown error')}")
    print("=" * 60)


def print_initialization_status(framework: str, messages: Dict[str, str]) -> None:
    print(f"\n=== {framework} Framework - Provider Status ===")
    for provider, message in messages.items():
        print(f"{get_display_name(provider)}: {message}")
    print("=" * 50 + "\n")


def get_user_choice(options: List[str], prompt: str) -> int:
    print(f"\n{prompt}")
    for index, option in enumerate(options, start=1):
        print(f"{index}. {option}")
    while True:
        try:
            raw_choice = input(f"Select an option (1-{len(options)}, default 1): ").strip()
            choice = (int(raw_choice) if raw_choice else 1) - 1
            if 0 <= choice < len(options):
                return choice
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def compact_model_selection_lines(model_identifiers: Sequence[str]) -> list[str]:
    """Format model identifiers into compact provider-grouped CLI selection lines."""
    lines: list[str] = []
    current_provider: Optional[str] = None
    current_options: list[str] = []

    def flush_current() -> None:
        nonlocal current_options
        if current_provider and current_options:
            lines.append(f"{current_provider}: " + " | ".join(current_options))
        current_options = []

    for index, model_identifier in enumerate(model_identifiers, start=1):
        provider_name, model_name = provider_and_model_name(model_identifier)
        if provider_name != current_provider or len(current_options) == 3:
            flush_current()
            current_provider = provider_name
        first_suffix = " [first]" if index == 1 else ""
        current_options.append(f"{index}. {model_name}{first_suffix}")
    flush_current()
    return lines


def model_identifiers_for_providers(providers: Iterable[str] | None = None) -> list[str]:
    """Return configured model identifiers, optionally filtered to provider names."""
    from shared.llm_models import ALL_MODEL_IDENTIFIERS, get_identifier_mappings

    provider_set = set(providers) if providers is not None else None
    mappings = get_identifier_mappings()
    return [
        identifier
        for identifier in ALL_MODEL_IDENTIFIERS
        if identifier in mappings and (provider_set is None or mappings[identifier].provider in provider_set)
    ]


def model_option_label(model_identifier: str) -> str:
    """Return a detailed CLI label for a configured model identifier."""
    from shared.llm_models import get_identifier_mappings

    config = get_identifier_mappings()[model_identifier]
    return f"{config.model} ({config.tier}; {model_identifier})"


def select_model_identifier(model_identifiers: Sequence[str], prompt: str = "Select a model:") -> str:
    """Prompt for one model identifier from a list using the shared choice helper."""
    choice_idx = get_user_choice([model_option_label(identifier) for identifier in model_identifiers], prompt)
    return model_identifiers[choice_idx]


def select_provider_model_identifier(providers: Sequence[str]) -> str:
    """Prompt for a provider, then prompt for a model identifier under that provider."""
    provider_idx = get_user_choice(
        [f"{get_display_name(provider)} ({len(model_identifiers_for_providers([provider]))} models)" for provider in providers],
        "Select a provider:",
    )
    provider = providers[provider_idx]
    return select_model_identifier(model_identifiers_for_providers([provider]), f"Select a {get_display_name(provider)} model:")
