"""Shared LLM model registry and routing helpers.

This module centralizes provider/model metadata so multiple volumes can reuse
consistent model identifiers, default selections, and provider aliases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Iterable, Sequence

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
]

PROVIDER_ALIASES = {
    "google": "google_genai",
    "google-genai": "google_genai",
}

PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI GPT",
    "anthropic": "Anthropic Claude",
    "google": "Google Gemini",
    "google_genai": "Google Gemini",
    "xai": "xAI Grok",
}

PROVIDER_API_KEY_ENV_VARS = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"),
    "google_genai": ("GOOGLE_GENAI_API_KEY", "GOOGLE_API_KEY"),
    "xai": ("XAI_API_KEY",),
}

PROVIDER_DEFAULT_MODEL_IDENTIFIERS = {
    "openai": "openai_gpt_5_4",
    "anthropic": "anthropic_claude_sonnet_4_6",
    "google": "google_genai_gemini_3_flash",
    "google_genai": "google_genai_gemini_3_flash",
    "xai": "xai_grok_4",
}


def normalize_provider(provider: str) -> str:
    return PROVIDER_ALIASES.get(provider, provider)


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
            provider="google_genai",
            model="gemini-3.1-pro-preview",
            tier="advanced",
            strengths=("reasoning", "tool-use", "multimodal"),
        ),
        "google_genai_gemini_3_flash": ModelConfig(
            name="google_genai_gemini_3_flash",
            provider="google_genai",
            model="gemini-3-flash-preview",
            tier="standard",
            strengths=("long-context", "research", "synthesis"),
        ),
        "google_genai_gemini_3_1_flash_lite": ModelConfig(
            name="google_genai_gemini_3_1_flash_lite",
            provider="google_genai",
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
    }


def get_default_model_config(provider: str) -> ModelConfig:
    normalized = normalize_provider(provider)
    identifier = PROVIDER_DEFAULT_MODEL_IDENTIFIERS[normalized]
    return get_identifier_mappings()[identifier]


def get_provider_catalog(providers: Sequence[str] | None = None) -> dict[str, dict[str, str]]:
    selected_providers = providers or list(PROVIDER_DEFAULT_MODEL_IDENTIFIERS)
    catalog: dict[str, dict[str, str]] = {}
    for provider in selected_providers:
        config = get_default_model_config(provider)
        catalog[provider] = {
            "provider": provider,
            "canonical_provider": config.provider,
            "api_key_env": PROVIDER_API_KEY_ENV_VARS[normalize_provider(provider)][0],
            "default_model": config.model,
            "default_model_identifier": config.name,
            "display_name": PROVIDER_DISPLAY_NAMES.get(provider, provider.replace('_', ' ').title()),
        }
    return catalog


def get_api_key_env_vars(provider: str) -> tuple[str, ...]:
    return PROVIDER_API_KEY_ENV_VARS.get(normalize_provider(provider), ())


def get_api_key(provider: str) -> str | None:
    for env_var in get_api_key_env_vars(provider):
        value = (os.getenv(env_var) or "").strip()
        if value:
            return value
    return None


def get_default_model_name(provider: str) -> str:
    return get_default_model_config(provider).model


def get_display_name(provider: str) -> str:
    normalized = normalize_provider(provider)
    return PROVIDER_DISPLAY_NAMES.get(provider, PROVIDER_DISPLAY_NAMES.get(normalized, provider.replace("_", " ").title()))


def get_public_provider_names() -> list[str]:
    return ["anthropic", "openai", "google", "xai"]


def get_all_providers() -> list[str]:
    return get_public_provider_names()


def select_models(model_identifiers: Sequence[str]) -> list[ModelConfig]:
    available = get_identifier_mappings()
    return [available[identifier] for identifier in model_identifiers]


def _infer_provider(prompt_l: str, selected_tools: Iterable[str]) -> str:
    tools = {name.lower() for name in selected_tools}
    social_markers = {"social", "tweet", "x.com", "reddit", "viral", "engagement", "thread", "post"}
    coding_markers = {"code", "bug", "debug", "refactor", "typescript", "python", "sql", "api"}
    research_markers = {"research", "paper", "citation", "study", "benchmark", "literature", "compare"}

    if any(token in prompt_l for token in {"chatgpt", "openai", "gpt"}):
        return "openai"
    if any(token in prompt_l for token in {"claude", "anthropic"}):
        return "anthropic"
    if any(token in prompt_l for token in {"gemini", "google"}):
        return "google"
    if any(token in prompt_l for token in {"grok", "xai"}):
        return "xai"
    if any(token in prompt_l for token in social_markers):
        return "xai"

    calculator_is_primary = "calculator" in tools and len(tools) <= 2
    if any(token in prompt_l for token in coding_markers) or calculator_is_primary:
        return "anthropic"
    if any(token in prompt_l for token in research_markers) or "parse_content" in tools:
        return "google"
    return "openai"


def _infer_tier(prompt_l: str, selected_tools: Iterable[str]) -> str:
    tools = {name.lower() for name in selected_tools}
    advanced_tools = {"analyze_text", "extract_tasks", "route_workflow", "summarize_text"}
    lite_tools = {"calculator", "resolve_datetime", "extract_keywords", "score_priority"}
    advanced_markers = {
        "architecture", "multi-step", "deep", "strategy", "tradeoff", "production design", "root cause", "long-form", "thorough",
    }
    lite_markers = {"quick", "brief", "short", "one-liner", "cheap", "fast"}

    if tools & advanced_tools:
        return "advanced"
    if tools and tools <= lite_tools:
        return "lite"
    if any(token in prompt_l for token in advanced_markers) or len(prompt_l) > 900:
        return "advanced"
    if any(token in prompt_l for token in lite_markers):
        return "lite"
    return "standard"


def route_model_for_prompt(prompt: str, selected_tools: Sequence[str], model_identifiers: Sequence[str] | None = None) -> ModelConfig:
    prompt_l = prompt.lower()
    selected_pool = set(model_identifiers or ALL_MODEL_IDENTIFIERS)
    provider = normalize_provider(_infer_provider(prompt_l, selected_tools))
    tier = _infer_tier(prompt_l, selected_tools)

    provider_tier_candidates = {
        "openai": {"advanced": "openai_gpt_5_4_pro", "standard": "openai_gpt_5_4", "lite": "openai_gpt_5_mini"},
        "anthropic": {"advanced": "anthropic_claude_opus_4_6", "standard": "anthropic_claude_sonnet_4_6", "lite": "anthropic_claude_haiku_4_5"},
        "google_genai": {"advanced": "google_genai_gemini_3_1_pro", "standard": "google_genai_gemini_3_flash", "lite": "google_genai_gemini_3_1_flash_lite"},
        "xai": {"advanced": "xai_grok_4", "standard": "xai_grok_3", "lite": "xai_grok_3_mini"},
    }

    candidates = provider_tier_candidates.get(provider, provider_tier_candidates["openai"])
    for tier_name in [tier, "standard", "lite", "advanced"]:
        candidate_name = candidates[tier_name]
        if candidate_name in selected_pool:
            return get_identifier_mappings()[candidate_name]

    if selected_pool:
        first_available = next(iter(selected_pool))
        return get_identifier_mappings()[first_available]

    all_available = get_identifier_mappings()
    if not all_available:
        raise ValueError("No model configurations available.")
    return next(iter(all_available.values()))


def resolve_llamaindex_model(selected_model: str):
    config = get_identifier_mappings().get(selected_model)
    if config is None:
        raise ValueError(f"Unknown model identifier '{selected_model}'")

    provider = config.provider
    model = config.model
    normalized_provider = normalize_provider(provider)
    llamaindex_provider = "google" if normalized_provider == "google_genai" else normalized_provider

    if llamaindex_provider == "openai":
        from llama_index.llms.openai import OpenAI
        llm = OpenAI(model=model)
    elif llamaindex_provider == "anthropic":
        from llama_index.llms.anthropic import Anthropic
        llm = Anthropic(model=model)
    elif llamaindex_provider == "google":
        from llama_index.llms.google_genai import GoogleGenAI
        llm = GoogleGenAI(model=model)
    elif llamaindex_provider == "xai":
        from llama_index.llms.openai_like import OpenAILike
        llm = OpenAILike(
            model=model,
            api_base=os.getenv("XAI_API_BASE", "https://api.x.ai/v1"),
            api_key=os.getenv("XAI_API_KEY"),
            is_chat_model=True,
            is_function_calling_model=True,
        )
    else:
        supported = "anthropic, google, openai, xai"
        raise ValueError(
            f"Unsupported provider '{config.provider}' for '{selected_model}'. Supported: {supported}"
        )

    return config, llm
