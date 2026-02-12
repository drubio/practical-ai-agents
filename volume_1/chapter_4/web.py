"""
web.py - Clean web interface for LLM testers
Now supports optional memory endpoints and session-based tracking.
"""

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import io
from contextlib import redirect_stdout
import json

from stream import iter_text_chunks, normalize_response_text


class QueryRequest(BaseModel):
    topic: str
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


def _supports_memory(manager) -> bool:
    return bool(
        getattr(manager, "memory_enabled", False)
        and hasattr(manager, "get_history")
        and hasattr(manager, "reset_memory")
    )


def _recover_structured_parse_error(result: dict) -> dict:
    """Downgrade structured-parse failures to usable plain-text responses."""
    if not isinstance(result, dict) or result.get("success"):
        return result

    error_message = str(result.get("error") or "")
    raw_response = result.get("raw_response")
    if "Failed to parse structured JSON response" not in error_message or not raw_response:
        return result

    fallback_text = normalize_response_text(raw_response)
    return {
        **result,
        "success": True,
        "error": None,
        "response": {
            "answer": fallback_text,
            "distilled": fallback_text,
            "metadata": {
                "confidence": "low",
                "notes": error_message,
            },
        },
        "raw_answer": fallback_text,
    }


def create_web_api(manager_class):
    app = FastAPI(
        title="LLM Service API",
        version="1.0.0",
        description="Universal API for LLM framework testing"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    manager = manager_class()

    @app.get("/")
    async def get_status():
        available = manager.get_available_providers()
        return {
            "framework": manager.framework,
            "available_providers": available,
            "total_available": len(available),
            "initialization_status": manager.initialization_messages,
            "status": "healthy" if available else "no_providers"
        }

    @app.get("/providers")
    async def get_providers():
        from utils import get_display_name, get_default_model
        providers = manager.get_available_providers()
        return {
            "framework": manager.framework,
            "providers": [
                {
                    "name": p,
                    "display_name": get_display_name(p),
                    "model": get_default_model(p),
                    "status": manager.initialization_messages.get(p, "Unknown")
                } for p in providers
            ],
            "count": len(providers)
        }

    @app.get("/capabilities")
    async def get_capabilities():
        return {
            "framework": manager.framework,
            "streaming": True,
            "memory": _supports_memory(manager),
        }

    @app.post("/query")
    async def query_single(request: QueryRequest):
        try:
            with redirect_stdout(io.StringIO()):
                args = {
                    "topic": request.topic,
                    "provider": request.provider,
                    "template": request.template,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature
                }
                effective_session_id = request.sessionId or request.session_id or "default"
                if _supports_memory(manager):
                    args["session_id"] = effective_session_id

                result = _recover_structured_parse_error(manager.ask_question(**args))

            if not result.get("success"):
                raise HTTPException(status_code=400, detail=result.get("error", "Query failed"))

            raw_response = result.get("response")
            content = raw_response if isinstance(raw_response, (dict, list)) else normalize_response_text(raw_response)

            return {
                "success": True,
                "framework": manager.framework,
                "provider": result["provider"],
                "model": result["model"],
                "response": content,
                "parameters": {
                    "temperature": result["temperature"],
                    "max_tokens": result["max_tokens"],
                    "template": request.template
                },
                "prompt": result["prompt"],
                "session_id": result.get("session_id", effective_session_id)
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query-stream")
    async def query_stream(request: QueryRequest):
        async def stream_events():
            try:
                with redirect_stdout(io.StringIO()):
                    args = {
                        "topic": request.topic,
                        "provider": request.provider,
                        "template": request.template,
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature,
                    }
                    effective_session_id = request.sessionId or request.session_id or "default"
                    if _supports_memory(manager):
                        args["session_id"] = effective_session_id

                    result = _recover_structured_parse_error(manager.ask_question(**args))

                if not result.get("success"):
                    error_payload = {"type": "error", "error": result.get("error", "Query failed")}
                    yield f"data: {json.dumps(error_payload)}\n\n"
                    return

                raw_response = result.get("response")
                response_text = normalize_response_text(raw_response)
                async for chunk in iter_text_chunks(response_text, delay_seconds=0.03):
                    payload = {"type": "chunk", "content": chunk}
                    yield f"data: {json.dumps(payload)}\n\n"

                done_payload = {
                    "type": "done",
                    "provider": result.get("provider"),
                    "model": result.get("model"),
                    "response": raw_response if isinstance(raw_response, dict) else None,
                    "token_usage": result.get("token_usage"),
                    "session_id": result.get("session_id") or effective_session_id,
                }
                yield f"data: {json.dumps(done_payload)}\n\n"
            except Exception as e:
                payload = {"type": "error", "error": str(e)}
                yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(stream_events(), media_type="text/event-stream")

    @app.post("/query-all")
    async def query_all(request: QueryAllRequest):
        try:
            with redirect_stdout(io.StringIO()):
                result = manager.query_all_providers(
                    topic=request.topic,
                    template=request.template,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )

            if not result.get("success"):
                raise HTTPException(status_code=400, detail=result.get("error", "Query failed"))

            clean_responses = {}
            for provider, res in result["responses"].items():
                content = res.get("response") if isinstance(res.get("response"), (dict, list)) else normalize_response_text(res.get("response"))
                clean_responses[provider] = {
                    "success": res["success"],
                    "model": res.get("model", ""),
                    "response": content,
                    "parameters": {
                        "temperature": res.get("temperature"),
                        "max_tokens": res.get("max_tokens")
                    }
                }

            return {
                "success": True,
                "framework": manager.framework,
                "prompt": result["prompt"],
                "responses": clean_responses
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/history")
    async def get_history(provider: str = "openai", session_id: str = "default"):
        if not _supports_memory(manager):
            raise HTTPException(status_code=400, detail="Memory not supported by this manager")
        return manager.get_history(provider, session_id)

    @app.post("/reset-memory")
    async def reset_memory(
        request: Optional[ResetMemoryRequest] = Body(None),
        provider: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        if not _supports_memory(manager):
            raise HTTPException(status_code=400, detail="Memory not supported by this manager")
        body_provider = request.provider if request else None
        body_session_id = (request.sessionId or request.session_id) if request else None
        effective_provider = body_provider if body_provider is not None else provider
        effective_session_id = body_session_id if body_session_id is not None else session_id
        return manager.reset_memory(effective_provider, effective_session_id)

    return app


def run_web_server(manager_class, host: str = "0.0.0.0", port: int = 8000):
    app = create_web_api(manager_class)
    try:
        framework_name = manager_class().framework
    except Exception:
        framework_name = "Unknown"

    print(f"Starting web server for {framework_name}")
    print(f"Docs: http://{host}:{port}/docs")
    print(f"Health: http://{host}:{port}/")
    uvicorn.run(app, host=host, port=port)


def main():
    print("Universal LLM Web API")
    print("Run using `run_web_server(manager_class)`")


if __name__ == "__main__":
    main()
