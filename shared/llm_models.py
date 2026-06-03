"""Shared LLM model registry and routing helpers.

This module centralizes provider/model metadata so multiple volumes can reuse
consistent model identifiers and explicit selections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv


SHARED_ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(SHARED_ENV_PATH)


@dataclass(frozen=True)
class ModelConfig:
    """LLM model metadata used for deterministic provider/tier routing."""

    name: str
    provider: str
    model: str
    tier: str
    strengths: tuple[str, ...] = field(default_factory=tuple)


ALL_MODEL_IDENTIFIERS = [
    "openai_gpt_5_4_pro",
    "openai_gpt_5_4",
    "openai_gpt_5_mini",
    "anthropic_claude_opus_4_6",
    "anthropic_claude_sonnet_4_6",
    "anthropic_claude_haiku_4_5",
    "google_genai_gemini_3_1_pro",
    "google_genai_gemini_3_flash",
    "google_genai_gemini_3_1_flash_lite",
    "xai_grok_4",
    "xai_grok_3",
    "xai_grok_3_mini",
    "deepseek_4_flash",
    "deepseek_4_pro",
]

PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI GPT",
    "anthropic": "Anthropic Claude",
    "google": "Google Gemini",
    "xai": "xAI Grok",
    "deepseek": "DeepSeek",
}

PROVIDER_API_KEY_ENV_VARS = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GOOGLE_GENAI_API_KEY", "GOOGLE_API_KEY"),
    "xai": ("XAI_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
}


def get_identifier_mappings() -> dict[str, ModelConfig]:
    """Return all configured models keyed by reusable model identifier."""
    return {
        "openai_gpt_5_4_pro": ModelConfig(
            name="openai_gpt_5_4_pro",
            provider="openai",
            model="gpt-5.4-pro",
            tier="advanced",
            strengths=("precision", "analysis", "long-form"),
        ),
        "openai_gpt_5_4": ModelConfig(
            name="openai_gpt_5_4",
            provider="openai",
            model="gpt-5.4",
            tier="standard",
            strengths=("balanced", "reasoning", "general"),
        ),
        "openai_gpt_5_mini": ModelConfig(
            name="openai_gpt_5_mini",
            provider="openai",
            model="gpt-5-mini",
            tier="lite",
            strengths=("cost-efficient", "fast", "general"),
        ),
        "anthropic_claude_opus_4_6": ModelConfig(
            name="anthropic_claude_opus_4_6",
            provider="anthropic",
            model="claude-opus-4-6",
            tier="advanced",
            strengths=("deep-reasoning", "coding", "planning"),
        ),
        "anthropic_claude_sonnet_4_6": ModelConfig(
            name="anthropic_claude_sonnet_4_6",
            provider="anthropic",
            model="claude-sonnet-4-6",
            tier="standard",
            strengths=("coding", "reasoning", "balanced"),
        ),
        "anthropic_claude_haiku_4_5": ModelConfig(
            name="anthropic_claude_haiku_4_5",
            provider="anthropic",
            model="claude-haiku-4-5",
            tier="lite",
            strengths=("speed", "coding", "cost-efficient"),
        ),
        "google_genai_gemini_3_1_pro": ModelConfig(
            name="google_genai_gemini_3_1_pro",
            provider="google",
            model="gemini-3.1-pro-preview",
            tier="advanced",
            strengths=("reasoning", "tool-use", "multimodal"),
        ),
        "google_genai_gemini_3_flash": ModelConfig(
            name="google_genai_gemini_3_flash",
            provider="google",
            model="gemini-3-flash-preview",
            tier="standard",
            strengths=("long-context", "research", "synthesis"),
        ),
        "google_genai_gemini_3_1_flash_lite": ModelConfig(
            name="google_genai_gemini_3_1_flash_lite",
            provider="google",
            model="gemini-3.1-flash-lite-preview",
            tier="lite",
            strengths=("fast", "retrieval", "classification"),
        ),
        "xai_grok_4": ModelConfig(
            name="xai_grok_4",
            provider="xai",
            model="grok-4",
            tier="advanced",
            strengths=("deep-analysis", "social", "long-form"),
        ),
        "xai_grok_3": ModelConfig(
            name="xai_grok_3",
            provider="xai",
            model="grok-3",
            tier="standard",
            strengths=("social", "trends", "analysis"),
        ),
        "xai_grok_3_mini": ModelConfig(
            name="xai_grok_3_mini",
            provider="xai",
            model="grok-3-mini",
            tier="lite",
            strengths=("social", "fast", "cost-efficient"),
        ),
        "deepseek_4_pro": ModelConfig(
            name="deepseek_4_pro",
            provider="deepseek",
            model="deepseek-v4-pro",
            tier="advanced",
            strengths=("deep-analysis", "reasoning", "long-form"),
        ),
        "deepseek_4_flash": ModelConfig(
            name="deepseek_4_flash",
            provider="deepseek",
            model="deepseek-v4-flash",
            tier="lite",
            strengths=("social", "fast", "cost-efficient"),
        ),        
    }


def get_model_config(selected_model: str) -> ModelConfig:
    """Return metadata for an explicit model identifier.

    The public selection value is the model identifier (for example
    ``openai_gpt_5_4``), not a bare provider. Bare providers are only used for
    API-key checks and provider grouping.
    """
    try:
        return get_identifier_mappings()[selected_model]
    except KeyError as exc:
        raise ValueError(f"Unknown model identifier '{selected_model}'") from exc


def get_models_for_provider(provider: str) -> list[ModelConfig]:
    mappings = get_identifier_mappings()
    return [mappings[identifier] for identifier in ALL_MODEL_IDENTIFIERS if identifier in mappings and mappings[identifier].provider == provider]


def get_provider_model_config(provider: str) -> ModelConfig:
    """Return the configured model used when only a provider is available."""
    for config in get_models_for_provider(provider):
        return config
    raise ValueError(f"No model identifiers configured for provider '{provider}'")


def get_provider_model_identifier(provider: str) -> str:
    return get_provider_model_config(provider).name


def resolve_model_identifier(selection: str | None, available_providers: Sequence[str] | None = None) -> str | None:
    """Resolve a model identifier or provider name to a configured model identifier."""
    available = set(available_providers) if available_providers is not None else None
    mappings = get_identifier_mappings()

    if selection in mappings:
        provider = mappings[selection].provider
        return selection if available is None or provider in available else None
    if selection in PROVIDER_DISPLAY_NAMES:
        return get_provider_model_identifier(selection) if available is None or selection in available else None
    if selection:
        return None
    if available:
        for provider in sort_providers_by_display_order(list(available)):
            return get_provider_model_identifier(provider)
    return None


def resolve_model_config(selection: str) -> ModelConfig:
    """Resolve either an explicit model identifier or a provider alias."""
    mappings = get_identifier_mappings()
    if selection in mappings:
        return mappings[selection]
    if selection in PROVIDER_DISPLAY_NAMES:
        return get_provider_model_config(selection)
    raise ValueError(f"Unknown model or provider selection '{selection}'")


def get_api_key_env_vars(provider: str) -> tuple[str, ...]:
    return PROVIDER_API_KEY_ENV_VARS.get(provider, ())


def get_api_key(provider: str) -> str | None:
    for env_var in get_api_key_env_vars(provider):
        value = (os.getenv(env_var) or "").strip()
        if value:
            return value
    return None



def get_display_name(provider: str) -> str:
    return PROVIDER_DISPLAY_NAMES.get(provider, provider.replace("_", " ").title())


def get_public_provider_names() -> list[str]:
    return list(PROVIDER_DISPLAY_NAMES)


def get_all_providers() -> list[str]:
    return get_public_provider_names()


def sort_providers_by_display_order(providers: Sequence[str]) -> list[str]:
    """Return providers in the canonical display order, with unknowns last."""
    display_rank = {provider: index for index, provider in enumerate(PROVIDER_DISPLAY_NAMES)}
    return sorted(
        providers,
        key=lambda provider: (
            display_rank.get(provider, len(display_rank)),
            get_display_name(provider).casefold(),
            provider,
        ),
    )
