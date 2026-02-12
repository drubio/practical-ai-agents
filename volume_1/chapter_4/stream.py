"""Streaming helpers for chapter web APIs."""

from __future__ import annotations

import asyncio
import ast
import re
import json
from typing import AsyncIterator, Iterable


def normalize_response_text(payload) -> str:
    """Normalize model responses to plain text for streaming transports."""
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
                for key in ("answer", "distilled", "content", "text", "message", "summary"):
                    value = maybe_json.get(key)
                    if isinstance(value, str) and value.strip():
                        return value
        except Exception:
            pass

        return payload
    if isinstance(payload, dict):
        for key in ("content", "text", "message", "answer", "final_answer", "distilled", "summary"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
    if hasattr(payload, "content") and isinstance(payload.content, str):
        return payload.content
    return str(payload)


def chunk_text(text: str, chunk_size: int = 28) -> Iterable[str]:
    """Chunk text so non-streaming providers can still emit incremental tokens."""
    clean = text or ""
    if not clean:
        yield ""
        return

    for i in range(0, len(clean), chunk_size):
        yield clean[i : i + chunk_size]


async def iter_text_chunks(text: str, chunk_size: int = 28, delay_seconds: float = 0.0) -> AsyncIterator[str]:
    """Yield text chunks asynchronously with optional pacing delay."""
    for part in chunk_text(text, chunk_size=chunk_size):
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        yield part
