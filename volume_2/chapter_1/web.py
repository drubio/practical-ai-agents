"""Reusable web API for chapter managers with `ask_question` interface."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from shared.llm_models import ALL_MODEL_IDENTIFIERS, get_identifier_mappings


class QueryRequest(BaseModel):
    topic: str
    provider: str | None = None
    model_identifier: str | None = None


def _extract_text(result: dict[str, Any]) -> str:
    return str(result.get("final_text") or result.get("response") or "")


def _provider_payload(manager: Any, model_identifier: str, idx: int) -> dict[str, str]:
    mappings = get_identifier_mappings()
    config = mappings.get(model_identifier)
    provider_name = str(getattr(config, "provider", getattr(manager, "provider", "unknown")))
    model_name = str(getattr(config, "model", getattr(manager, "model", model_identifier)))
    canonical_name = f"{provider_name}:{model_name}"
    return {
        "id": str(idx),
        "name": canonical_name,
        "display_name": f"{provider_name.capitalize()} ({model_identifier})",
        "provider": provider_name,
        "default_model": model_name,
        "model": model_name,
        "model_identifier": model_identifier,
        "status": "✓ Initialized successfully",
        "framework": str(getattr(manager, "framework", "unknown")),
    }


def create_web_api(manager: Any, stream_default: bool = False) -> FastAPI:
    app = FastAPI(title="Chapter LLM API", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

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
        payloads = [_provider_payload(manager, mid, idx) for idx, mid in enumerate(ALL_MODEL_IDENTIFIERS, start=1)]
        active = payloads[0] if payloads else _provider_payload(manager, "unknown", 1)
        return {"providers": payloads, "count": len(payloads), "active_provider": active["name"]}

    @app.post("/query")
    async def query(request: QueryRequest):
        try:
            selected_provider = request.provider or request.model_identifier
            result = manager.ask_question(request.topic)
            if not isinstance(result, dict):
                raise HTTPException(status_code=500, detail="Manager returned invalid payload")
            if not result.get("success"):
                raise HTTPException(status_code=400, detail=result.get("error", "Query failed"))
            return {"success": True, "framework": getattr(manager, "framework", "unknown"), "topic": request.topic, "selected_provider": selected_provider, "response": _extract_text(result), "raw": result}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/query-stream")
    async def query_stream(request: QueryRequest):
        async def events():
            try:
                result = manager.ask_question(request.topic)
                if not isinstance(result, dict) or not result.get("success"):
                    error = result.get("error", "Query failed") if isinstance(result, dict) else "Manager returned invalid payload"
                    yield f"data: {{\"type\":\"error\",\"error\":\"{error}\"}}\n\n"
                    return
                text = _extract_text(result)
                for chunk in text.split():
                    yield f"data: {{\"type\":\"chunk\",\"content\":{chunk!r}}}\n\n"
                yield "data: {\"type\":\"done\"}\n\n"
            except Exception as exc:
                yield f"data: {{\"type\":\"error\",\"error\":\"{str(exc)}\"}}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/capabilities")
    async def capabilities():
        return {"streaming": True, "default_stream": stream_default, "single_provider": True}

    return app


def run_web_server(manager: Any, host: str = "0.0.0.0", port: int = 8000, stream_default: bool = False) -> None:
    app = create_web_api(manager, stream_default=stream_default)
    uvicorn.run(app, host=host, port=port)
