"""Streaming helpers for volume 2 chapter APIs."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Iterable


def normalize_response_text(payload) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("response", "answer", "content", "text", "message"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
    return str(payload)


def chunk_text(text: str, chunk_size: int = 28) -> Iterable[str]:
    clean = text or ""
    if not clean:
        yield ""
        return
    for i in range(0, len(clean), chunk_size):
        yield clean[i : i + chunk_size]


async def iter_text_chunks(text: str, chunk_size: int = 28, delay_seconds: float = 0.0) -> AsyncIterator[str]:
    for part in chunk_text(text, chunk_size=chunk_size):
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        yield part


def to_sse_line(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"
