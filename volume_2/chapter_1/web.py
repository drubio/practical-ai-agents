"""Reusable web API bootstrap for volume 2 chapter apps.

This module intentionally mirrors the chapter 4 API contract so the
chapter_8 UI can use volume_2 agent servers as a drop-in replacement.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from utils import (
    chunk_text,
    get_provider_options,
    get_chapter_logger,
    iter_stream_text_chunks,
    normalize_agent_api_payload,
    to_sse_line,
)


logger = get_chapter_logger("volume_2.chapter_1.web")


class QueryRequest(BaseModel):
    topic: str = Field(..., description="Prompt or question")
    provider: Optional[str] = None
    template: str = "{topic}"
    max_tokens: int = 1000
    temperature: float = 0.7
    session_id: Optional[str] = "default"
    sessionId: Optional[str] = None


class QueryAllRequest(BaseModel):
    topic: str
    template: str = "{topic}"
    max_tokens: int = 1000
    temperature: float = 0.7
    session_id: Optional[str] = "default"
    sessionId: Optional[str] = None


class ResetMemoryRequest(BaseModel):
    provider: Optional[str] = None
    session_id: Optional[str] = None
    sessionId: Optional[str] = None


def _supports_history(manager) -> bool:
    return hasattr(manager, "get_history") and callable(getattr(manager, "get_history"))


def _supports_reset_memory(manager) -> bool:
    return hasattr(manager, "reset_memory") and callable(getattr(manager, "reset_memory"))


def _manager_provider_options(manager):
    custom_options = getattr(manager, "get_provider_options", None)
    if callable(custom_options):
        return custom_options()
    return get_provider_options(getattr(manager, "model_identifiers", None))


def _resolve_request_manager(manager, requested_model: Optional[str]):
    auto_selection_values = {
        None,
        "",
        getattr(manager, "auto_provider_option_name", None),
        getattr(manager, "active_model_identifier", None),
        getattr(manager, "model", None),
    }
    if requested_model in auto_selection_values:
        return manager, getattr(manager, "active_model_identifier", requested_model)

    available_identifiers = set(getattr(manager, "model_identifiers", []) or [])
    if available_identifiers and requested_model not in available_identifiers:
        raise HTTPException(status_code=400, detail=f"Unsupported provider/model selection '{requested_model}'")

    try:
        request_manager = manager.__class__(model=requested_model, stream=getattr(manager, "stream", False))
    except TypeError as exc:
        raise HTTPException(status_code=500, detail=f"Unable to construct manager for '{requested_model}': {exc}") from exc

    return request_manager, requested_model


def _build_query_response(result: dict, request: QueryRequest, manager, *, stream_override: Optional[bool] = None) -> dict:
    stream_mode = getattr(manager, "stream", False) if stream_override is None else stream_override
    normalized = normalize_agent_api_payload(result.get("response"), stream=stream_mode)
    effective_session_id = request.sessionId or request.session_id or "default"

    response_payload = {
        "success": True,
        "framework": getattr(manager, "framework", "agent"),
        "provider": result.get("provider"),
        "model": result.get("model"),
        "response": normalized["response"],
        "raw_answer": normalized.get("raw_answer"),
        "prompt": result.get("prompt", request.template.format(topic=request.topic)),
        "parameters": {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "template": request.template,
        },
        "session_id": result.get("session_id", effective_session_id),
    }

    if "tool_calls" in normalized:
        response_payload["tool_calls"] = normalized["tool_calls"]

    return response_payload


def create_web_api(manager, enable_streaming: bool = False) -> FastAPI:
    app = FastAPI(
        title="Volume 2 Agent API",
        version="1.0.0",
        description="HTTP bootstrap for chapter agent workflows",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def status():
        logger.debug("Status endpoint called")
        return {
            "framework": getattr(manager, "framework", "agent"),
            "available_providers": [getattr(manager, "provider", "unknown")],
            "total_available": 1,
            "initialization_status": {
                getattr(manager, "provider", "unknown"): "Ready"
            },
            "status": "healthy",
        }

    @app.get("/providers")
    async def providers():
        options = _manager_provider_options(manager)
        return {
            "framework": getattr(manager, "framework", "agent"),
            "providers": options,
            "count": len(options),
        }

    @app.get("/capabilities")
    async def capabilities():
        return {
            "framework": getattr(manager, "framework", "agent"),
            "streaming": True,
            "memory": _supports_history(manager) and _supports_reset_memory(manager),
            "memory_retrieval": False,
            "coagent": False,
        }

    @app.post("/query")
    async def query(payload: QueryRequest):
        logger.info("/query request | chars=%s", len(payload.topic))
        try:
            request_manager, active_identifier = _resolve_request_manager(manager, payload.provider)
            with redirect_stdout(io.StringIO()):
                result = request_manager.ask_question(payload.topic)
        except Exception as exc:  # noqa: PERF203
            logger.exception("/query raised exception")
            if isinstance(exc, HTTPException):
                raise exc
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Manager returned a non-dict response")
        if not result.get("success"):
            logger.error("/query failed | error=%s", result.get("error"))
            raise HTTPException(status_code=400, detail=result.get("error", "Query failed"))
        result["active_model_identifier"] = active_identifier
        return _build_query_response(result, payload, request_manager)

    async def _stream_query(payload: QueryRequest):
        try:
            request_manager, active_identifier = _resolve_request_manager(manager, payload.provider)
            with redirect_stdout(io.StringIO()):
                result = request_manager.ask_question(payload.topic)

            if not isinstance(result, dict):
                yield to_sse_line({"type": "error", "error": "Manager returned a non-dict response"})
                return

            if not result.get("success"):
                yield to_sse_line({"type": "error", "error": result.get("error", "Query failed")})
                return

            raw_response = result.get("response")
            stream_enabled = bool(getattr(request_manager, "stream", False))
            streamed_text = ""

            if stream_enabled:
                async for chunk in iter_stream_text_chunks(raw_response):
                    if chunk:
                        streamed_text += chunk
                        yield to_sse_line({"type": "chunk", "content": chunk})
            else:
                normalized = normalize_agent_api_payload(raw_response, stream=False)
                full_text = normalized.get("raw_answer") or ""
                for chunk in chunk_text(full_text):
                    if chunk:
                        streamed_text += chunk
                        yield to_sse_line({"type": "chunk", "content": chunk})

            normalized = normalize_agent_api_payload(streamed_text if streamed_text else raw_response, stream=False)
            done_payload = {
                "type": "done",
                "provider": result.get("provider"),
                "model": result.get("model"),
                "active_model_identifier": active_identifier,
                "response": normalized["response"] if isinstance(normalized.get("response"), dict) else None,
                "raw_answer": normalized.get("raw_answer"),
                "session_id": result.get("session_id", payload.sessionId or payload.session_id or "default"),
            }
            if "tool_calls" in normalized:
                done_payload["tool_calls"] = normalized["tool_calls"]
            yield to_sse_line(done_payload)
        except Exception as exc:  # noqa: PERF203
            logger.exception("/query-stream failed")
            if isinstance(exc, HTTPException):
                yield to_sse_line({"type": "error", "error": exc.detail})
            else:
                yield to_sse_line({"type": "error", "error": str(exc)})

    @app.post("/query-stream")
    async def query_stream(payload: QueryRequest):
        if not enable_streaming:
            raise HTTPException(status_code=404, detail="Streaming is disabled. Start with --stream.")
        logger.info("/query-stream request | chars=%s", len(payload.topic))
        return StreamingResponse(_stream_query(payload), media_type="text/event-stream")

    @app.post("/query/stream")
    async def query_stream_alias(payload: QueryRequest):
        return await query_stream(payload)

    @app.post("/query-all")
    async def query_all(payload: QueryAllRequest):
        raise HTTPException(status_code=400, detail="This agent server supports only single-provider query mode")

    @app.get("/history")
    async def history(provider: str = "openai", session_id: str = "default"):
        if not _supports_history(manager):
            raise HTTPException(status_code=400, detail="Session memory not supported by this manager")
        return manager.get_history(provider, session_id)

    @app.post("/reset-memory")
    async def reset_memory(
        request: Optional[ResetMemoryRequest] = Body(None),
        provider: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        if not _supports_reset_memory(manager):
            raise HTTPException(status_code=400, detail="Session memory not supported by this manager")
        effective_provider = request.provider if request and request.provider is not None else provider
        effective_session_id = (
            (request.sessionId or request.session_id) if request else None
        ) or session_id
        return manager.reset_memory(effective_provider, effective_session_id)

    return app


def run_web_server(manager, host: str = "0.0.0.0", port: int = 8000, enable_streaming: bool = False):
    import uvicorn

    app = create_web_api(manager, enable_streaming=enable_streaming)
    print(f"Starting web server for {getattr(manager, 'framework', 'Unknown')}")
    print(f"Docs: http://{host}:{port}/docs")
    print(f"Health: http://{host}:{port}/")
    uvicorn.run(app, host=host, port=port)
