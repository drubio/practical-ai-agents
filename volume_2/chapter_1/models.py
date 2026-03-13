"""Model registry and routing helpers for volume 2 agents.

This mirrors the local-tool architecture used in tools.py:
- chapter 1 can import a constrained model subset
- chapter 2 can import all models and route among them
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ModelConfig:
    """LLM model metadata used for deterministic provider/tier routing."""

    name: str
    provider: str
    model: str
    tier: str
    strengths: tuple[str, ...] = field(default_factory=tuple)


ALL_MODEL_IDENTIFIERS = [
    "openai_gpt_4o_mini",
    "openai_gpt_5_2",
    "openai_gpt_4_1",
    "anthropic_claude_3_5_haiku",
    "anthropic_claude_3_5_sonnet",
    "anthropic_claude_3_7_sonnet",
    "google_gemini_1_5_flash",
    "google_gemini_1_5_pro",
    "google_gemini_2_5_flash",
    "xai_grok_3_mini",
    "xai_grok_3",
    "xai_grok_4",
]


def get_identifier_mappings() -> dict[str, ModelConfig]:
    """Return all configured models keyed by reusable model identifier."""
    return {
        "openai_gpt_4o_mini": ModelConfig(
            name="openai_gpt_4o_mini",
            provider="openai",
            model="gpt-4o-mini",
            tier="lite",
            strengths=("cost-efficient", "fast", "general"),
        ),
        "openai_gpt_5_2": ModelConfig(
            name="openai_gpt_5_2",
            provider="openai",
            model="gpt-5.2",
            tier="standard",
            strengths=("balanced", "reasoning", "general"),
        ),
        "openai_gpt_4_1": ModelConfig(
            name="openai_gpt_4_1",
            provider="openai",
            model="gpt-4.1",
            tier="advanced",
            strengths=("precision", "analysis", "long-form"),
        ),
        "anthropic_claude_3_5_haiku": ModelConfig(
            name="anthropic_claude_3_5_haiku",
            provider="anthropic",
            model="claude-3-5-haiku-latest",
            tier="lite",
            strengths=("speed", "coding", "cost-efficient"),
        ),
        "anthropic_claude_3_5_sonnet": ModelConfig(
            name="anthropic_claude_3_5_sonnet",
            provider="anthropic",
            model="claude-3-5-sonnet-latest",
            tier="standard",
            strengths=("coding", "reasoning", "balanced"),
        ),
        "anthropic_claude_3_7_sonnet": ModelConfig(
            name="anthropic_claude_sonnet_4_5",
            provider="anthropic",
            model="claude-sonnet-4-5",
            tier="advanced",
            strengths=("deep-reasoning", "coding", "planning"),
        ),
        "google_gemini_1_5_flash": ModelConfig(
            name="google_gemini_1_5_flash",
            provider="google",
            model="gemini-1.5-flash",
            tier="lite",
            strengths=("fast", "retrieval", "classification"),
        ),
        "google_gemini_1_5_pro": ModelConfig(
            name="google_gemini_1_5_pro",
            provider="google",
            model="gemini-1.5-pro",
            tier="standard",
            strengths=("long-context", "research", "synthesis"),
        ),
        "google_gemini_2_5_flash": ModelConfig(
            name="google_gemini_2_5_flash",
            provider="google",
            model="gemini-2.5-flash",
            tier="advanced",
            strengths=("reasoning", "tool-use", "multimodal"),
        ),
        "xai_grok_3_mini": ModelConfig(
            name="xai_grok_3_mini",
            provider="xai",
            model="grok-3-mini",
            tier="lite",
            strengths=("social", "fast", "cost-efficient"),
        ),
        "xai_grok_3": ModelConfig(
            name="xai_grok_3",
            provider="xai",
            model="grok-3",
            tier="standard",
            strengths=("social", "trends", "analysis"),
        ),
        "xai_grok_4": ModelConfig(
            name="xai_grok_4",
            provider="xai",
            model="grok-4",
            tier="advanced",
            strengths=("deep-analysis", "social", "long-form"),
        ),
    }


def select_models(model_identifiers: Sequence[str]) -> list[ModelConfig]:
    available = get_identifier_mappings()
    return [available[identifier] for identifier in model_identifiers]


def _infer_provider(prompt_l: str, selected_tools: Iterable[str]) -> str:
    """Classify prompt into a best-fit provider domain."""
    tools = {name.lower() for name in selected_tools}

    social_markers = {"social", "tweet", "x.com", "reddit", "viral", "engagement", "thread", "post"}
    coding_markers = {"code", "bug", "debug", "refactor", "typescript", "python", "sql", "api"}
    research_markers = {"research", "paper", "citation", "study", "benchmark", "literature", "compare"}

    if any(token in prompt_l for token in social_markers):
        return "xai"

    if any(token in prompt_l for token in coding_markers) or "calculator" in tools:
        return "anthropic"

    if any(token in prompt_l for token in research_markers) or "parse_content" in tools:
        return "google"

    if any(token in prompt_l for token in {"chatgpt", "openai", "gpt"}):
        return "openai"
    if any(token in prompt_l for token in {"claude", "anthropic"}):
        return "anthropic"
    if any(token in prompt_l for token in {"gemini", "google"}):
        return "google"
    if any(token in prompt_l for token in {"grok", "xai"}):
        return "xai"

    return "openai"


def _infer_tier(prompt_l: str, selected_tools: Iterable[str]) -> str:
    tools = {name.lower() for name in selected_tools}

    advanced_tools = {"analyze_text", "extract_tasks", "route_workflow", "summarize_text"}
    lite_tools = {"calculator", "resolve_datetime", "format_json", "extract_keywords", "score_priority"}

    advanced_markers = {
        "architecture",
        "multi-step",
        "deep",
        "strategy",
        "tradeoff",
        "production design",
        "root cause",
        "long-form",
        "thorough",
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
    """Route prompt to a model from the available set.

    The chosen model is constrained to `model_identifiers` (defaults to ALL_MODEL_IDENTIFIERS).
    """
    prompt_l = prompt.lower()
    selected_pool = set(model_identifiers or ALL_MODEL_IDENTIFIERS)

    provider = _infer_provider(prompt_l, selected_tools)
    tier = _infer_tier(prompt_l, selected_tools)

    provider_tier_candidates = {
        "openai": {
            "lite": "openai_gpt_4o_mini",
            "standard": "openai_gpt_5_2",
            "advanced": "openai_gpt_4_1",
        },
        "anthropic": {
            "lite": "anthropic_claude_3_5_haiku",
            "standard": "anthropic_claude_3_5_sonnet",
            "advanced": "anthropic_claude_3_7_sonnet",
        },
        "google": {
            "lite": "google_gemini_1_5_flash",
            "standard": "google_gemini_1_5_pro",
            "advanced": "google_gemini_2_5_flash",
        },
        "xai": {
            "lite": "xai_grok_3_mini",
            "standard": "xai_grok_3",
            "advanced": "xai_grok_4",
        },
    }

    fallback_order = [tier, "standard", "lite", "advanced"]
    candidates = provider_tier_candidates.get(provider, provider_tier_candidates["openai"])
    for tier_name in fallback_order:
        candidate_name = candidates[tier_name]
        if candidate_name in selected_pool:
            return get_identifier_mappings()[candidate_name]

    # Final safety fallback.
    if selected_pool:
        first_available = next(iter(selected_pool))
        return get_identifier_mappings()[first_available]

    all_available = get_identifier_mappings()
    if not all_available:
        raise ValueError("No model configurations available.")
    return next(iter(all_available.values()))


@dataclass(frozen=True)
class ResolvedLlamaIndexModel:
    provider: str
    model: str


def _create_openai_llm(model: str):
    from llama_index.llms.openai import OpenAI

    return OpenAI(model=model)


def _create_xai_llm(model: str):
    from llama_index.llms.openai_like import OpenAILike

    return OpenAILike(
        model=model,
        api_base=os.getenv("XAI_API_BASE", "https://api.x.ai/v1"),
        api_key=os.getenv("XAI_API_KEY"),
        is_chat_model=True,
        is_function_calling_model=True,
    )


def _create_anthropic_llm(model: str):
    from llama_index.llms.anthropic import Anthropic

    return Anthropic(model=model, api_key=os.getenv("ANTHROPIC_API_KEY"))


def _create_google_genai_llm(model: str):
    from llama_index.llms.google_genai import GoogleGenAI

    return GoogleGenAI(model=model, api_key=os.getenv("GOOGLE_API_KEY"))


LLAMAINDEX_PROVIDER_FACTORIES = {
    "openai": _create_openai_llm,
    "xai": _create_xai_llm,
    "anthropic": _create_anthropic_llm,
    "google": _create_google_genai_llm,
    "google_genai": _create_google_genai_llm,
}


def resolve_llamaindex_model(selected_model: str):
    available = get_identifier_mappings()
    if selected_model in available:
        config = available[selected_model]
        provider, raw_model_name = config.provider, config.model
    elif ":" in selected_model:
        provider, raw_model_name = selected_model.split(":", 1)
    else:
        raise ValueError(
            f"Unsupported model '{selected_model}'. Use a model identifier from ALL_MODEL_IDENTIFIERS or provider:model syntax."
        )

    factory = LLAMAINDEX_PROVIDER_FACTORIES.get(provider)
    if factory is None:
        supported = ", ".join(sorted(LLAMAINDEX_PROVIDER_FACTORIES))
        raise ValueError(f"Unsupported provider '{provider}' for '{selected_model}'. Supported: {supported}")

    resolved = ResolvedLlamaIndexModel(provider=provider, model=raw_model_name)
    return resolved, factory(raw_model_name)
