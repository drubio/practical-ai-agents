"""Shared web/server helpers reusable across book chapters."""

from __future__ import annotations

import ast
import asyncio
import io
import json
import re
from contextlib import redirect_stdout
from typing import Any, AsyncIterator, Callable, Iterable, Optional, Union

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn


class SharedQueryRequest(BaseModel):
    topic: str = Field(..., description="Prompt or question")
    provider: Optional[Union[str, int]] = None
    model_identifier: Optional[Union[str, int]] = None
    modelIdentifier: Optional[Union[str, int]] = None
    template: str = "{topic}"
    max_tokens: int = 1000
    temperature: float = 0.7
    session_id: Optional[str] = "default"
    sessionId: Optional[str] = None


class SharedQueryAllRequest(BaseModel):
    topic: str
    template: str = "{topic}"
    max_tokens: int = 1000
    temperature: float = 0.7
    session_id: Optional[str] = "default"
    sessionId: Optional[str] = None


class SharedResetMemoryRequest(BaseModel):
    provider: Optional[Union[str, int]] = None
    session_id: Optional[str] = None
    sessionId: Optional[str] = None


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


def chunk_text(text: str, chunk_size: int = 28) -> Iterable[str]:
    clean = text or ""
    if not clean:
        yield ""
        return
    for index in range(0, len(clean), chunk_size):
        yield clean[index : index + chunk_size]


async def iter_text_chunks(text: str, chunk_size: int = 28, delay_seconds: float = 0.0) -> AsyncIterator[str]:
    for part in chunk_text(text, chunk_size=chunk_size):
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        yield part


def to_sse_line(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def resolve_session_id(*candidates: Any, default: str = "default") -> str:
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return default


def result_is_success(result: Any) -> bool:
    return isinstance(result, dict) and bool(result.get("success"))


async def stream_text_sse(
    text: str,
    *,
    chunk_size: int = 28,
    delay_seconds: float = 0.0,
    event_type: str = "chunk",
) -> AsyncIterator[str]:
    async for part in iter_text_chunks(text, chunk_size=chunk_size, delay_seconds=delay_seconds):
        if part:
            yield to_sse_line({"type": event_type, "content": part})


def capture_console_output(func: Callable[[], Any]) -> tuple[Any, str]:
    with io.StringIO() as buffer, redirect_stdout(buffer):
        result = func()
        return result, buffer.getvalue()


async def run_manager_in_thread(func: Callable[[], Any]) -> Any:
    result, _ = await asyncio.to_thread(capture_console_output, func)
    return result


def build_manager(manager_class_or_factory):
    if not callable(manager_class_or_factory):
        raise TypeError("Invalid manager class/factory provided")
    try:
        return manager_class_or_factory()
    except TypeError:
        return manager_class_or_factory


def supports_memory(manager) -> bool:
    return bool(
        getattr(manager, "memory_enabled", False)
        and hasattr(manager, "get_history")
        and hasattr(manager, "reset_memory")
    )


def supports_memory_retrieval(manager) -> bool:
    return bool(
        getattr(manager, "retrieval_memory_enabled", False)
        and hasattr(manager, "get_history")
        and hasattr(manager, "reset_memory")
    )


def supports_session_memory(manager) -> bool:
    return supports_memory(manager) or supports_memory_retrieval(manager)


def supports_coagent(manager) -> bool:
    return bool(getattr(manager, "coagent", False))


def supports_history(manager) -> bool:
    return hasattr(manager, "get_history") and callable(getattr(manager, "get_history"))


def supports_reset_memory(manager) -> bool:
    return hasattr(manager, "reset_memory") and callable(getattr(manager, "reset_memory"))


def run_uvicorn_app(app: FastAPI, framework_name: str, host: str = "0.0.0.0", port: int = 8000):
    print(f"Starting web server for {framework_name}")
    print(f"Docs: http://{host}:{port}/docs")
    print(f"Health: http://{host}:{port}/")
    uvicorn.run(app, host=host, port=port)


def _call_manager_method(manager: Any, snake_name: str, camel_name: str, default: Any = None) -> Any:
    method = getattr(manager, snake_name, None) or getattr(manager, camel_name, None)
    if callable(method):
        return method()
    return default


def _manager_attr(manager: Any, snake_name: str, camel_name: str, default: Any = None) -> Any:
    return getattr(manager, snake_name, getattr(manager, camel_name, default))


def manager_available_providers(manager: Any) -> list[str]:
    """Return available providers for managers that expose either Python or JS naming."""
    providers = _call_manager_method(manager, "get_available_providers", "getAvailableProviders", None)
    if providers is None:
        provider = getattr(manager, "provider", None)
        providers = [provider] if provider else []
    return list(providers or [])


def manager_initialization_messages(manager: Any) -> dict[str, str]:
    """Return provider initialization statuses from Python- or JS-style managers."""
    return dict(_manager_attr(manager, "initialization_messages", "initializationMessages", {}) or {})


def provider_selection_map(manager: Any) -> dict[str, str]:
    from shared.llm_models import sort_providers_by_display_order

    sorted_providers = sort_providers_by_display_order(manager_available_providers(manager))
    return {str(index): provider for index, provider in enumerate(sorted_providers, start=1)}


def available_model_identifiers(manager: Any, *, require_available_provider: bool = True) -> list[str]:
    from shared.llm_models import ALL_MODEL_IDENTIFIERS, get_identifier_mappings

    mappings = get_identifier_mappings()
    available_providers = set(manager_available_providers(manager))
    if not require_available_provider:
        return [identifier for identifier in ALL_MODEL_IDENTIFIERS if identifier in mappings]
    return [
        identifier
        for identifier in ALL_MODEL_IDENTIFIERS
        if identifier in mappings and mappings[identifier].provider in available_providers
    ]


def model_identifiers_for_provider(provider: str) -> list[str]:
    from shared.llm_models import ALL_MODEL_IDENTIFIERS, get_identifier_mappings

    mappings = get_identifier_mappings()
    return [identifier for identifier in ALL_MODEL_IDENTIFIERS if identifier in mappings and mappings[identifier].provider == provider]


def model_payload(
    model_identifier: str,
    manager: Any,
    idx: int | None = None,
    *,
    require_available_provider: bool = True,
    default_status: str = "Unknown",
) -> dict[str, Any]:
    from shared.llm_models import ALL_MODEL_IDENTIFIERS, get_identifier_mappings

    config = get_identifier_mappings()[model_identifier]
    if idx is None:
        try:
            idx = available_model_identifiers(manager, require_available_provider=require_available_provider).index(model_identifier) + 1
        except ValueError:
            idx = ALL_MODEL_IDENTIFIERS.index(model_identifier) + 1 if model_identifier in ALL_MODEL_IDENTIFIERS else 0
    canonical_name = f"{config.provider}:{config.model}"
    status = manager_initialization_messages(manager).get(config.provider, default_status)
    return {
        "id": str(idx),
        "name": canonical_name,
        "display_name": f"{config.provider.capitalize()} ({model_identifier})",
        "provider": config.provider,
        "default_model": config.model,
        "model": config.model,
        "model_identifier": model_identifier,
        "default_model_identifier": model_identifier,
        "model_tier": config.tier,
        "default_model_tier": config.tier,
        "strengths": list(config.strengths),
        "status": status,
        "framework": str(getattr(manager, "framework", "unknown")),
    }


def model_payloads(manager: Any, *, require_available_provider: bool = True, default_status: str = "Unknown") -> list[dict[str, Any]]:
    return [
        model_payload(identifier, manager, idx, require_available_provider=require_available_provider, default_status=default_status)
        for idx, identifier in enumerate(available_model_identifiers(manager, require_available_provider=require_available_provider), start=1)
    ]


def provider_payload(provider: str, manager: Any, *, require_available_provider: bool = True, default_status: str = "Unknown") -> dict[str, Any]:
    from shared.utils import get_default_model_details

    details = get_default_model_details(provider)
    model_identifiers = model_identifiers_for_provider(provider)
    return {
        "name": provider,
        "display_name": details["display_name"],
        "provider": details["canonical_provider"],
        "default_model": details["default_model"],
        "default_model_identifier": details["default_model_identifier"],
        "default_model_tier": details["default_model_tier"],
        "models": [model_payload(identifier, manager, require_available_provider=require_available_provider, default_status=default_status) for identifier in model_identifiers],
        "model_identifiers": model_identifiers,
        "status": manager_initialization_messages(manager).get(provider, default_status),
    }


def normalize_provider_input(manager: Any, provider: Optional[Union[str, int]]) -> str | None:
    if provider is None:
        return None

    from shared.llm_models import get_all_providers

    provider_map = provider_selection_map(manager)
    available = {name.lower() for name in manager_available_providers(manager)}
    configured = {name.lower() for name in get_all_providers()}

    if isinstance(provider, int):
        return provider_map.get(str(provider))

    candidate = str(provider).strip()
    if not candidate:
        return None
    if candidate in provider_map:
        return provider_map[candidate]

    lowered = candidate.lower()
    if lowered in available or lowered in configured:
        return lowered
    return candidate


def normalize_model_identifier_input(
    manager: Any,
    model_identifier: Optional[Union[str, int]],
    *,
    require_available_provider: bool = True,
) -> str | None:
    if model_identifier is None:
        return None

    from shared.llm_models import get_identifier_mappings

    mappings = get_identifier_mappings()
    payloads = model_payloads(manager, require_available_provider=require_available_provider)
    model_map = {payload["id"]: payload["model_identifier"] for payload in payloads}
    canonical_map = {payload["name"].lower(): payload["model_identifier"] for payload in payloads}

    if isinstance(model_identifier, int):
        return model_map.get(str(model_identifier))

    candidate = str(model_identifier).strip()
    if not candidate:
        return None
    if candidate in model_map:
        return model_map[candidate]
    if candidate in mappings:
        return candidate
    lowered = candidate.lower()
    if lowered in canonical_map:
        return canonical_map[lowered]
    for identifier, config in mappings.items():
        if identifier.lower() == lowered or config.model.lower() == lowered:
            return identifier
    return None
