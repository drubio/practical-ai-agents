"""Reusable web API bootstrap for volume 2 chapter apps."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from stream import normalize_response_text, to_sse_line
from utils import get_chapter_logger


logger = get_chapter_logger("volume_2.chapter_1.web")


class QueryRequest(BaseModel):
    topic: str = Field(..., description="Prompt or question")


class QueryResponse(BaseModel):
    success: bool
    prompt: str
    response: Optional[str] = None
    error: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None


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
            "model": getattr(manager, "model", None),
            "streaming_enabled": enable_streaming,
            "status": "healthy",
        }

    @app.post("/query", response_model=QueryResponse)
    async def query(payload: QueryRequest):
        logger.info("/query request | chars=%s", len(payload.topic))
        result = manager.ask_question(payload.topic)
        if not result.get("success"):
            logger.error("/query failed | error=%s", result.get("error"))
            raise HTTPException(status_code=500, detail=result.get("error", "Query failed"))
        return result

    @app.post("/query/stream")
    async def query_stream(payload: QueryRequest):
        if not enable_streaming:
            raise HTTPException(status_code=404, detail="Streaming is disabled. Start with --stream.")
        logger.info("/query/stream request | chars=%s", len(payload.topic))

        async def event_generator():
            try:
                for chunk in manager.iter_answer_chunks(payload.topic):
                    text = normalize_response_text(chunk)
                    if text:
                        yield to_sse_line({"type": "token", "token": text})
                yield to_sse_line({"type": "done"})
            except Exception as exc:  # noqa: PERF203
                logger.exception("/query/stream failed")
                yield to_sse_line({"type": "error", "error": str(exc)})

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app


def run_web_server(manager, host: str = "0.0.0.0", port: int = 8000, enable_streaming: bool = False):
    import uvicorn

    app = create_web_api(manager, enable_streaming=enable_streaming)
    print(f"\nStarting API on http://{host}:{port} (streaming={'on' if enable_streaming else 'off'})")
    uvicorn.run(app, host=host, port=port)
