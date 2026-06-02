"""Reusable web API for LangChain managers with an ``ask_question`` interface."""

from __future__ import annotations

import asyncio
from typing import Any, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from shared.web import model_payloads, normalize_model_identifier_input, normalize_response_text, stream_text_sse, to_sse_line


class QueryRequest(BaseModel):
    topic: str
    provider: Union[str, int, None] = None
    model_identifier: Union[str, int, None] = None
    modelIdentifier: Union[str, int, None] = None


def _extract_text(result: dict[str, Any]) -> str:
    return normalize_response_text(
        result.get("final_text")
        or result.get("finalText")
        or result.get("response")
        or ""
    )


def _request_model_identifier(manager: Any, request: QueryRequest) -> str | None:
    requested_model_identifier = request.modelIdentifier if request.modelIdentifier is not None else request.model_identifier
    return normalize_model_identifier_input(
        manager,
        requested_model_identifier or request.provider,
        require_available_provider=False,
    )


def _manager_for_model(manager: Any, model_identifier: str | None) -> Any:
    if not model_identifier or model_identifier == getattr(manager, "model_identifier", None):
        return manager

    manager_class = getattr(manager, "__class__", None)
    if callable(manager_class):
        try:
            return manager_class(model_identifier)
        except TypeError:
            pass

    set_model = getattr(manager, "set_model_identifier", None) or getattr(manager, "set_model", None)
    if callable(set_model):
        set_model(model_identifier)
    return manager


def create_web_api(manager: Any, stream_default: bool = False) -> FastAPI:
    app = FastAPI(title="LangChain Chapter LLM API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def status():
        return {
            "framework": getattr(manager, "framework", "unknown"),
            "tool_names": getattr(manager, "tool_names", []),
            "status": "healthy",
        }

    @app.get("/health")
    async def health():
        return {"status": "healthy", "framework": getattr(manager, "framework", "unknown")}

    @app.get("/providers")
    async def providers():
        payloads = model_payloads(manager, require_available_provider=False, default_status="✓ Initialized successfully")
        active = payloads[0]["name"] if payloads else None
        return {"providers": payloads, "count": len(payloads), "active_provider": active}

    @app.post("/query")
    async def query(request: QueryRequest):
        try:
            selected_model_identifier = _request_model_identifier(manager, request)
            selected_provider = selected_model_identifier or request.provider
            request_manager = _manager_for_model(manager, selected_model_identifier)
            result = await asyncio.to_thread(request_manager.ask_question, request.topic)
            if not isinstance(result, dict):
                raise HTTPException(status_code=500, detail="Manager returned invalid payload")
            if not result.get("success"):
                raise HTTPException(status_code=400, detail=result.get("error", "Query failed"))
            return {
                "success": True,
                "framework": getattr(manager, "framework", "unknown"),
                "topic": request.topic,
                "selected_provider": selected_provider,
                "response": _extract_text(result),
                "raw": result,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/query-stream")
    async def query_stream(request: QueryRequest):
        async def events():
            try:
                selected_model_identifier = _request_model_identifier(manager, request)
                request_manager = _manager_for_model(manager, selected_model_identifier)
                result = await asyncio.to_thread(request_manager.ask_question, request.topic)
                if not isinstance(result, dict) or not result.get("success"):
                    error = result.get("error", "Query failed") if isinstance(result, dict) else "Manager returned invalid payload"
                    yield to_sse_line({"type": "error", "error": str(error)})
                    return
                text = _extract_text(result)
                async for event in stream_text_sse(text, delay_seconds=0.03):
                    yield event
                yield to_sse_line({"type": "done"})
            except Exception as exc:
                yield to_sse_line({"type": "error", "error": str(exc)})

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/capabilities")
    async def capabilities():
        return {"streaming": True, "default_stream": stream_default, "single_provider": True}

    return app


def run_web_server(manager: Any, host: str = "0.0.0.0", port: int = 8000, stream_default: bool = False) -> None:
    app = create_web_api(manager, stream_default=stream_default)
    uvicorn.run(app, host=host, port=port)
