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
