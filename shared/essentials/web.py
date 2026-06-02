"""Essentials-book web API helpers built on shared web primitives."""

from __future__ import annotations

from typing import Optional, Union

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from shared.utils import (
    normalize_response_text,
    parse_structured_json_response,
)
from shared.llm_models import get_identifier_mappings
from shared.essentials.utils import selected_model_context


class SharedQueryRequest(BaseModel):
    topic: str = Field(..., description="Prompt or question")
    provider: Optional[Union[str, int]] = None
    model_identifier: Optional[Union[str, int]] = None
    modelIdentifier: Optional[Union[str, int]] = None
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


from shared.web import (
    available_model_identifiers,
    build_manager,
    resolve_session_id,
    model_payload,
    model_payloads,
    model_identifiers_for_provider,
    normalize_model_identifier_input,
    normalize_provider_input,
    provider_payload,
    provider_selection_map,
    result_is_success,
    run_manager_in_thread,
    run_uvicorn_app,
    stream_text_sse,
    supports_coagent,
    supports_memory,
    supports_memory_retrieval,
    supports_session_memory,
    to_sse_line,
)


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


def recover_structured_parse_error(result: dict) -> dict:
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



def result_value(result: dict, *keys, default=None):
    for key in keys:
        if isinstance(result, dict) and key in result and result.get(key) is not None:
            return result.get(key)
    return default


def is_corrupt_history_error(exc: Exception) -> bool:
    message = str(exc)
    return any(signal in message for signal in ("string indices must be integers", "Expecting value", "JSON"))


def ask_question_with_recovery(manager, args: dict, session_id: Optional[str]):
    try:
        return manager.ask_question(**args)
    except Exception as exc:
        if not supports_session_memory(manager) or not is_corrupt_history_error(exc):
            raise
        provider = args.get("provider")
        try:
            manager.reset_memory(provider, session_id)
        except Exception:
            raise exc
        return manager.ask_question(**args)


def build_query_args(manager, request: SharedQueryRequest) -> tuple[dict, str, Optional[str]]:
    effective_session_id = resolve_session_id(request.sessionId, request.session_id)
    requested_model_identifier = request.modelIdentifier if request.modelIdentifier is not None else request.model_identifier
    model_identifier = normalize_model_identifier_input(manager, requested_model_identifier)
    provider_input = request.provider
    if model_identifier is None:
        model_identifier = normalize_model_identifier_input(manager, provider_input)
    if model_identifier:
        provider = get_identifier_mappings()[model_identifier].provider
    else:
        provider = normalize_provider_input(manager, provider_input)
    args = {
        "topic": request.topic,
        "provider": provider,
        "template": request.template,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }
    if supports_session_memory(manager):
        args["session_id"] = effective_session_id
    return args, effective_session_id, model_identifier


async def execute_manager_query(manager, args: dict, session_id: Optional[str], model_identifier: Optional[str] = None) -> dict:
    def run_query():
        with selected_model_context(model_identifier):
            return ask_question_with_recovery(manager, args, session_id)

    return recover_structured_parse_error(await run_manager_in_thread(run_query))


def query_error_message(result: dict) -> str:
    if not isinstance(result, dict):
        return "Manager returned a non-dict response"
    return result.get("error", "Query failed")


def serialize_query_response(manager, request: SharedQueryRequest, result: dict, session_id: str, model_identifier: Optional[str] = None) -> dict:
    raw_response = result.get("response")
    content = raw_response if isinstance(raw_response, (dict, list)) else normalize_response_text(raw_response)
    return {
        "success": True,
        "framework": manager.framework,
        "topic": request.topic,
        "selected_provider": request.provider or model_identifier,
        "provider": result_value(result, "provider", default="unknown"),
        "model": result_value(result, "model", default=""),
        "model_identifier": model_identifier,
        "response": content,
        "parameters": {
            "temperature": result_value(result, "temperature"),
            "max_tokens": result_value(result, "max_tokens", "maxTokens"),
            "template": request.template,
        },
        "prompt": result_value(result, "prompt", default=request.template.format(topic=request.topic)),
        "session_id": result_value(result, "session_id", "sessionId", default=session_id),
    }


def serialize_done_event(result: dict, raw_response, session_id: str) -> dict:
    return {
        "type": "done",
        "provider": result_value(result, "provider"),
        "model": result_value(result, "model"),
        "model_identifier": result_value(result, "model_identifier"),
        "response": raw_response if isinstance(raw_response, dict) else None,
        "token_usage": result_value(result, "token_usage", "tokenUsage"),
        "session_id": result_value(result, "session_id", "sessionId", default=session_id),
    }


def create_web_api(manager_class):
    app = FastAPI(title="LLM Service API", version="1.0.0", description="Universal API for LLM framework testing")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    manager = build_manager(manager_class)

    @app.get('/')
    async def get_status():
        available = manager.get_available_providers()
        return {
            "framework": manager.framework,
            "available_providers": available,
            "available_provider_details": [provider_payload(provider, manager) for provider in available],
            "available_model_details": model_payloads(manager),
            "total_available": len(available),
            "total_models_available": len(model_payloads(manager)),
            "initialization_status": manager.initialization_messages,
            "status": "healthy" if available else "no_providers",
        }

    @app.get('/providers')
    async def get_providers():
        providers = manager.get_available_providers()
        models = model_payloads(manager)
        active = models[0]["name"] if models else None
        return {
            "framework": manager.framework,
            "providers": models,
            "provider_groups": [provider_payload(provider, manager) for provider in providers],
            "available_providers": providers,
            "models": models,
            "count": len(models),
            "provider_count": len(providers),
            "model_count": len(models),
            "active_provider": active,
        }

    @app.get('/capabilities')
    async def get_capabilities():
        return {"framework": manager.framework, "streaming": True, "memory": supports_memory(manager), "memory_retrieval": supports_memory_retrieval(manager), "coagent": supports_coagent(manager)}

    @app.post('/query')
    async def query_single(request: SharedQueryRequest):
        try:
            args, effective_session_id, model_identifier = build_query_args(manager, request)
            result = await execute_manager_query(manager, args, effective_session_id, model_identifier)
            if isinstance(result, dict) and model_identifier:
                result["model_identifier"] = model_identifier
            if not isinstance(result, dict):
                raise HTTPException(status_code=500, detail=query_error_message(result))
            if not result_is_success(result):
                raise HTTPException(status_code=400, detail=query_error_message(result))
            return serialize_query_response(manager, request, result, effective_session_id, model_identifier)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post('/query-stream')
    async def query_stream(request: SharedQueryRequest):
        async def stream_events():
            try:
                args, effective_session_id, model_identifier = build_query_args(manager, request)
                result = await execute_manager_query(manager, args, effective_session_id, model_identifier)
                if isinstance(result, dict) and model_identifier:
                    result["model_identifier"] = model_identifier
                if not isinstance(result, dict) or not result_is_success(result):
                    yield to_sse_line({"type": "error", "error": query_error_message(result)})
                    return
                raw_response = result.get('response')
                async for event in stream_text_sse(normalize_response_text(raw_response), delay_seconds=0.03):
                    yield event
                yield to_sse_line(serialize_done_event(result, raw_response, effective_session_id))
            except Exception as exc:
                yield to_sse_line({"type": "error", "error": str(exc)})
        return StreamingResponse(stream_events(), media_type='text/event-stream')

    @app.get('/history')
    async def get_history(provider: str = 'openai', session_id: str = 'default'):
        if not supports_session_memory(manager):
            raise HTTPException(status_code=400, detail='Session memory not supported by this manager')
        return manager.get_history(provider, session_id)

    @app.post('/reset-memory')
    async def reset_memory(request: Optional[SharedResetMemoryRequest] = Body(None), provider: Optional[str] = None, session_id: Optional[str] = None):
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
