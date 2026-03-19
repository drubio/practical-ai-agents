"""
web.py - Clean web interface for LLM testers
Now supports optional memory endpoints and session-based tracking.
"""

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Union
import uvicorn
import io
from contextlib import redirect_stdout
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from stream import iter_text_chunks, normalize_response_text
from utils import parse_structured_json_response, get_all_providers, get_default_model, get_display_name


class QueryRequest(BaseModel):
    topic: str
    provider: Optional[Union[str, int]] = None
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
    provider: Optional[Union[str, int]] = None
    session_id: Optional[str] = None
    sessionId: Optional[str] = None


def _supports_memory(manager) -> bool:
    """True when manager replays full chat history into prompts."""
    return bool(
        getattr(manager, "memory_enabled", False)
        and hasattr(manager, "get_history")
        and hasattr(manager, "reset_memory")
    )


def _supports_memory_retrieval(manager) -> bool:
    """True when manager supports retrieval-based memory context."""
    return bool(
        getattr(manager, "retrieval_memory_enabled", False)
        and hasattr(manager, "get_history")
        and hasattr(manager, "reset_memory")
    )


def _supports_session_memory(manager) -> bool:
    """True when any session-based memory mode (full replay or retrieval) is available."""
    return _supports_memory(manager) or _supports_memory_retrieval(manager)


def _supports_coagent(manager) -> bool:
    """True when manager exposes coagent features."""
    return bool(getattr(manager, "coagent", False))


def _parse_structured_raw_response(raw_response):
    if raw_response is None:
        return None
    try:
        parsed = parse_structured_json_response(raw_response)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _extract_answer_text(response_obj, raw_response):
    if isinstance(response_obj, dict):
        answer = response_obj.get("answer") or response_obj.get("distilled") or response_obj.get("summary")
        if isinstance(answer, str) and answer.strip():
            return answer
    return normalize_response_text(raw_response)

def _recover_structured_parse_error(result: dict) -> dict:
    """Recover structured outputs without dropping fields like keywords."""
    if not isinstance(result, dict):
        return result

    raw_response = result.get("raw_response")
    parsed_response = _parse_structured_raw_response(raw_response)

    if result.get("success"):
        if parsed_response:
            response = result.get("response")
            metadata_notes = None
            if isinstance(response, dict):
                metadata_notes = (response.get("metadata") or {}).get("notes")
            if isinstance(metadata_notes, str) and metadata_notes.startswith("Failed to parse structured JSON response"):
                return {
                    **result,
                    "response": parsed_response,
                    "raw_answer": _extract_answer_text(parsed_response, raw_response),
                }
        return result

    error_message = str(result.get("error") or "")
    if "Failed to parse structured JSON response" not in error_message or not raw_response:
        return result

    recovered_response = parsed_response or {
        "answer": normalize_response_text(raw_response),
        "distilled": normalize_response_text(raw_response),
        "metadata": {
            "confidence": "low",
            "notes": error_message,
        },
    }

    return {
        **result,
        "success": True,
        "error": None,
        "response": recovered_response,
        "raw_answer": _extract_answer_text(recovered_response, raw_response),
    }


def _provider_selection_map(manager):
    """Build the same provider ordering used by the CLI selection prompt."""

    available = manager.get_available_providers()
    sorted_providers = sorted(available, key=lambda provider: (provider != "openai", get_display_name(provider)))
    return {str(index): provider for index, provider in enumerate(sorted_providers, start=1)}


def _normalize_provider_input(manager, provider: Optional[Union[str, int]]):
    """Accept provider numbers (CLI style) and provider keywords (web style)."""
    if provider is None:
        return None

    provider_map = _provider_selection_map(manager)
    available = {name.lower() for name in manager.get_available_providers()}
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




def _result_value(result: dict, *keys, default=None):
    for key in keys:
        if isinstance(result, dict) and key in result and result.get(key) is not None:
            return result.get(key)
    return default




def _is_corrupt_history_error(exc: Exception) -> bool:
    message = str(exc)
    signals = (
        "string indices must be integers",
        "Expecting value",
        "JSON",
    )
    return any(signal in message for signal in signals)


def _ask_question_with_recovery(manager, args: dict, session_id: Optional[str]):
    """Run manager.ask_question with a one-time session-memory recovery retry."""
    try:
        return manager.ask_question(**args)
    except Exception as exc:
        if not _supports_session_memory(manager) or not _is_corrupt_history_error(exc):
            raise

        provider = args.get("provider")
        try:
            manager.reset_memory(provider, session_id)
        except Exception:
            raise exc

        return manager.ask_question(**args)


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
            "memory_retrieval": _supports_memory_retrieval(manager),
            "coagent": _supports_coagent(manager),
        }

    @app.post("/query")
    async def query_single(request: QueryRequest):
        try:
            with redirect_stdout(io.StringIO()):
                args = {
                    "topic": request.topic,
                    "provider": _normalize_provider_input(manager, request.provider),
                    "template": request.template,
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature
                }
                effective_session_id = request.sessionId or request.session_id or "default"
                if _supports_session_memory(manager):
                    args["session_id"] = effective_session_id

                result = _recover_structured_parse_error(_ask_question_with_recovery(manager, args, effective_session_id))

            if not isinstance(result, dict):
                raise HTTPException(status_code=500, detail="Manager returned a non-dict response")

            if not result.get("success"):
                raise HTTPException(status_code=400, detail=result.get("error", "Query failed"))

            raw_response = result.get("response")
            content = raw_response if isinstance(raw_response, (dict, list)) else normalize_response_text(raw_response)

            return {
                "success": True,
                "framework": manager.framework,
                "provider": _result_value(result, "provider", default="unknown"),
                "model": _result_value(result, "model", default=""),
                "response": content,
                "parameters": {
                    "temperature": _result_value(result, "temperature"),
                    "max_tokens": _result_value(result, "max_tokens", "maxTokens"),
                    "template": request.template
                },
                "prompt": _result_value(result, "prompt", default=request.template.format(topic=request.topic)),
                "session_id": _result_value(result, "session_id", "sessionId", default=effective_session_id)
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
                        "provider": _normalize_provider_input(manager, request.provider),
                        "template": request.template,
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature,
                    }
                    effective_session_id = request.sessionId or request.session_id or "default"
                    if _supports_session_memory(manager):
                        args["session_id"] = effective_session_id

                    result = _recover_structured_parse_error(_ask_question_with_recovery(manager, args, effective_session_id))

                if not isinstance(result, dict):
                    error_payload = {"type": "error", "error": "Manager returned a non-dict response"}
                    yield f"data: {json.dumps(error_payload)}\n\n"
                    return

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
                    "provider": _result_value(result, "provider"),
                    "model": _result_value(result, "model"),
                    "response": raw_response if isinstance(raw_response, dict) else None,
                    "token_usage": _result_value(result, "token_usage", "tokenUsage"),
                    "session_id": _result_value(result, "session_id", "sessionId", default=effective_session_id),
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

            if not isinstance(result, dict):
                raise HTTPException(status_code=500, detail="Manager returned a non-dict response")

            if not result.get("success"):
                raise HTTPException(status_code=400, detail=result.get("error", "Query failed"))

            responses = result.get("responses")
            if not isinstance(responses, dict):
                raise HTTPException(status_code=500, detail="Manager returned invalid responses payload")

            clean_responses = {}
            for provider, res in responses.items():
                if not isinstance(res, dict):
                    clean_responses[provider] = {
                        "success": False,
                        "model": "",
                        "response": normalize_response_text(res),
                        "parameters": {
                            "temperature": None,
                            "max_tokens": None,
                        },
                    }
                    continue

                raw_content = _result_value(res, "response")
                content = raw_content if isinstance(raw_content, (dict, list)) else normalize_response_text(raw_content)
                clean_responses[provider] = {
                    "success": bool(_result_value(res, "success", default=False)),
                    "model": _result_value(res, "model", default=""),
                    "response": content,
                    "parameters": {
                        "temperature": _result_value(res, "temperature"),
                        "max_tokens": _result_value(res, "max_tokens", "maxTokens"),
                    }
                }

            return {
                "success": True,
                "framework": manager.framework,
                "prompt": _result_value(result, "prompt", default=request.template.format(topic=request.topic)),
                "responses": clean_responses
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/history")
    async def get_history(provider: str = "openai", session_id: str = "default"):
        if not _supports_session_memory(manager):
            raise HTTPException(status_code=400, detail="Session memory not supported by this manager")
        return manager.get_history(provider, session_id)

    @app.post("/reset-memory")
    async def reset_memory(
        request: Optional[ResetMemoryRequest] = Body(None),
        provider: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        if not _supports_session_memory(manager):
            raise HTTPException(status_code=400, detail="Session memory not supported by this manager")
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
