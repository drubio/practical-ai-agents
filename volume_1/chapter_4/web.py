"""web.py - Clean web interface for LLM testers
Now supports optional memory endpoints and session-based tracking.
"""

from __future__ import annotations

import os
import sys
from contextlib import redirect_stdout
import io
from typing import Optional, Union

from fastapi import Body, HTTPException

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from shared.web import (
    SharedQueryAllRequest as QueryAllRequest,
    SharedQueryRequest as QueryRequest,
    SharedResetMemoryRequest as ResetMemoryRequest,
    build_manager,
    iter_text_chunks,
    normalize_response_text,
    run_uvicorn_app,
    supports_coagent,
    supports_memory,
    supports_memory_retrieval,
    supports_session_memory,
    to_sse_line,
)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from utils import parse_structured_json_response, get_all_providers, get_default_model, get_display_name


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
                return {**result, "response": parsed_response, "raw_answer": _extract_answer_text(parsed_response, raw_response)}
        return result

    error_message = str(result.get("error") or "")
    if "Failed to parse structured JSON response" not in error_message or not raw_response:
        return result

    recovered_response = parsed_response or {
        "answer": normalize_response_text(raw_response),
        "distilled": normalize_response_text(raw_response),
        "metadata": {"confidence": "low", "notes": error_message},
    }

    return {**result, "success": True, "error": None, "response": recovered_response, "raw_answer": _extract_answer_text(recovered_response, raw_response)}


def _provider_selection_map(manager):
    available = manager.get_available_providers()
    sorted_providers = sorted(available, key=lambda provider: (provider != "openai", get_display_name(provider)))
    return {str(index): provider for index, provider in enumerate(sorted_providers, start=1)}


def _normalize_provider_input(manager, provider: Optional[Union[str, int]]):
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
    return any(signal in message for signal in ("string indices must be integers", "Expecting value", "JSON"))


def _ask_question_with_recovery(manager, args: dict, session_id: Optional[str]):
    try:
        return manager.ask_question(**args)
    except Exception as exc:
        if not supports_session_memory(manager) or not _is_corrupt_history_error(exc):
            raise
        provider = args.get("provider")
        try:
            manager.reset_memory(provider, session_id)
        except Exception:
            raise exc
        return manager.ask_question(**args)


def create_web_api(manager_class):
    app = FastAPI(title="LLM Service API", version="1.0.0", description="Universal API for LLM framework testing")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    manager = build_manager(manager_class)

    @app.get('/')
    async def get_status():
        available = manager.get_available_providers()
        return {"framework": manager.framework, "available_providers": available, "total_available": len(available), "initialization_status": manager.initialization_messages, "status": "healthy" if available else "no_providers"}

    @app.get('/providers')
    async def get_providers():
        providers = manager.get_available_providers()
        return {"framework": manager.framework, "providers": [{"name": p, "display_name": get_display_name(p), "model": get_default_model(p), "status": manager.initialization_messages.get(p, "Unknown")} for p in providers], "count": len(providers)}

    @app.get('/capabilities')
    async def get_capabilities():
        return {"framework": manager.framework, "streaming": True, "memory": supports_memory(manager), "memory_retrieval": supports_memory_retrieval(manager), "coagent": supports_coagent(manager)}

    @app.post('/query')
    async def query_single(request: QueryRequest):
        try:
            with redirect_stdout(io.StringIO()):
                args = {"topic": request.topic, "provider": _normalize_provider_input(manager, request.provider), "template": request.template, "max_tokens": request.max_tokens, "temperature": request.temperature}
                effective_session_id = request.sessionId or request.session_id or "default"
                if supports_session_memory(manager):
                    args["session_id"] = effective_session_id
                result = _recover_structured_parse_error(_ask_question_with_recovery(manager, args, effective_session_id))
            if not isinstance(result, dict):
                raise HTTPException(status_code=500, detail="Manager returned a non-dict response")
            if not result.get('success'):
                raise HTTPException(status_code=400, detail=result.get('error', 'Query failed'))
            raw_response = result.get('response')
            content = raw_response if isinstance(raw_response, (dict, list)) else normalize_response_text(raw_response)
            return {"success": True, "framework": manager.framework, "provider": _result_value(result, 'provider', default='unknown'), "model": _result_value(result, 'model', default=''), "response": content, "parameters": {"temperature": _result_value(result, 'temperature'), "max_tokens": _result_value(result, 'max_tokens', 'maxTokens'), "template": request.template}, "prompt": _result_value(result, 'prompt', default=request.template.format(topic=request.topic)), "session_id": _result_value(result, 'session_id', 'sessionId', default=effective_session_id)}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post('/query-stream')
    async def query_stream(request: QueryRequest):
        async def stream_events():
            try:
                with redirect_stdout(io.StringIO()):
                    args = {"topic": request.topic, "provider": _normalize_provider_input(manager, request.provider), "template": request.template, "max_tokens": request.max_tokens, "temperature": request.temperature}
                    effective_session_id = request.sessionId or request.session_id or "default"
                    if supports_session_memory(manager):
                        args["session_id"] = effective_session_id
                    result = _recover_structured_parse_error(_ask_question_with_recovery(manager, args, effective_session_id))
                if not isinstance(result, dict):
                    yield to_sse_line({"type": "error", "error": "Manager returned a non-dict response"})
                    return
                if not result.get('success'):
                    yield to_sse_line({"type": "error", "error": result.get('error', 'Query failed')})
                    return
                raw_response = result.get('response')
                async for chunk in iter_text_chunks(normalize_response_text(raw_response), delay_seconds=0.03):
                    yield to_sse_line({"type": "chunk", "content": chunk})
                yield to_sse_line({"type": "done", "provider": _result_value(result, 'provider'), "model": _result_value(result, 'model'), "response": raw_response if isinstance(raw_response, dict) else None, "token_usage": _result_value(result, 'token_usage', 'tokenUsage'), "session_id": _result_value(result, 'session_id', 'sessionId', default=effective_session_id)})
            except Exception as exc:
                yield to_sse_line({"type": "error", "error": str(exc)})
        return StreamingResponse(stream_events(), media_type='text/event-stream')

    @app.post('/query-all')
    async def query_all(request: QueryAllRequest):
        try:
            with redirect_stdout(io.StringIO()):
                result = manager.query_all_providers(topic=request.topic, template=request.template, max_tokens=request.max_tokens, temperature=request.temperature)
            if not isinstance(result, dict):
                raise HTTPException(status_code=500, detail='Manager returned a non-dict response')
            if not result.get('success'):
                raise HTTPException(status_code=400, detail=result.get('error', 'Query failed'))
            responses = result.get('responses')
            if not isinstance(responses, dict):
                raise HTTPException(status_code=500, detail='Manager returned invalid responses payload')
            clean_responses = {}
            for provider, res in responses.items():
                if not isinstance(res, dict):
                    clean_responses[provider] = {"success": False, "model": '', "response": normalize_response_text(res), "parameters": {"temperature": None, "max_tokens": None}}
                    continue
                raw_content = _result_value(res, 'response')
                content = raw_content if isinstance(raw_content, (dict, list)) else normalize_response_text(raw_content)
                clean_responses[provider] = {"success": bool(_result_value(res, 'success', default=False)), "model": _result_value(res, 'model', default=''), "response": content, "parameters": {"temperature": _result_value(res, 'temperature'), "max_tokens": _result_value(res, 'max_tokens', 'maxTokens')}}
            return {"success": True, "framework": manager.framework, "prompt": _result_value(result, 'prompt', default=request.template.format(topic=request.topic)), "responses": clean_responses}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get('/history')
    async def get_history(provider: str = 'openai', session_id: str = 'default'):
        if not supports_session_memory(manager):
            raise HTTPException(status_code=400, detail='Session memory not supported by this manager')
        return manager.get_history(provider, session_id)

    @app.post('/reset-memory')
    async def reset_memory(request: Optional[ResetMemoryRequest] = Body(None), provider: Optional[str] = None, session_id: Optional[str] = None):
        if not supports_session_memory(manager):
            raise HTTPException(status_code=400, detail='Session memory not supported by this manager')
        body_provider = request.provider if request else None
        body_session_id = (request.sessionId or request.session_id) if request else None
        effective_provider = body_provider if body_provider is not None else provider
        effective_session_id = body_session_id if body_session_id is not None else session_id
        return manager.reset_memory(effective_provider, effective_session_id)

    return app


def run_web_server(manager_class, host: str = '0.0.0.0', port: int = 8000):
    app = create_web_api(manager_class)
    try:
        framework_name = build_manager(manager_class).framework
    except Exception:
        framework_name = 'Unknown'
    run_uvicorn_app(app, framework_name, host=host, port=port)


def main():
    print('Universal LLM Web API')
    print('Run using `run_web_server(manager_class)`')


if __name__ == '__main__':
    main()
