from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, PlainTextResponse

from .config import resolve_gemini_config, settings
from .gemini_client import GeminiClient
from .models import (
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    GenerateResponse,
    GeminiGenerateContentRequest,
    GeminiContent,
    GeminiContentPart,
    GeminiResponse,
    Message,
)


router = APIRouter()


def _now_iso() -> str:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()


def ollama_messages_to_gemini_contents(messages: List[Message]) -> List[GeminiContent]:
    contents: List[GeminiContent] = []
    import json as _json
    for m in messages:
        if m.role in ("user", "system"):
            contents.append(
                GeminiContent(role="user", parts=[GeminiContentPart(text=m.content)])
            )
        elif m.role == "assistant":
            contents.append(
                GeminiContent(role="model", parts=[GeminiContentPart(text=m.content)])
            )
        elif m.role == "tool":
            # Map tool outputs to functionResponse in a user message
            response_obj: Any
            try:
                response_obj = _json.loads(m.content)
            except Exception:
                response_obj = {"result": m.content}
            contents.append(
                GeminiContent(
                    role="user",
                    parts=[
                        GeminiContentPart(
                            functionResponse={
                                "name": m.name or "tool",
                                "response": response_obj,
                            }
                        )
                    ],
                )
            )
        else:
            contents.append(
                GeminiContent(role="user", parts=[GeminiContentPart(text=m.content)])
            )
    return contents


def options_to_generation_config(options: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not options:
        return None
    mapping = {
        "temperature": "temperature",
        "top_p": "topP",
        "top_k": "topK",
        "max_tokens": "maxOutputTokens",
    }
    cfg: Dict[str, Any] = {}
    for k, v in options.items():
        if k in mapping:
            cfg[mapping[k]] = v
    return cfg or None


def normalize_tools(tools: Optional[List[dict]]) -> Optional[List[dict]]:
    if not tools:
        return None
    out: List[dict] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        t2 = dict(t)
        if "function_declarations" in t2 and "functionDeclarations" not in t2:
            t2["functionDeclarations"] = t2.pop("function_declarations")
        out.append(t2)
    return out or None


async def call_gemini(model: str, payload: GeminiGenerateContentRequest, request: Request, *, stream: bool = False):
    base_url, version, api_key, model_id = resolve_gemini_config(
        dict(request.headers), model_alias=model
    )
    client = GeminiClient(base_url=base_url, version=version, api_key=api_key)

    if stream:
        async def _gen():
            try:
                async for line in client.stream_generate_content(
                    model=model_id, payload=payload.model_dump(exclude_none=True)
                ):
                    yield line
            finally:
                await client.aclose()

        return _gen()

    resp = await client.generate_content(
        model=model_id, payload=payload.model_dump(exclude_none=True)
    )
    try:
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise HTTPException(status_code=resp.status_code, detail=detail)
        return resp.json()
    finally:
        await client.aclose()


def _parse_gemini_stream_line(line: str) -> tuple[str, Optional[dict]]:
    """Extract text delta or functionCall from a Gemini stream line.

    Returns (text, function_call_dict_or_None).
    """
    import json as _json

    # Some servers may prepend 'data: ' for SSE framing; strip it if present.
    if line.startswith("data: "):
        line = line[len("data: "):]

    try:
        obj = _json.loads(line)
    except Exception:
        return "", None

    text = ""
    fn = None
    try:
        cands = obj.get("candidates") or []
        if cands:
            parts = ((cands[0].get("content") or {}).get("parts") or [])
            if parts:
                p0 = parts[0]
                if isinstance(p0, dict):
                    text = p0.get("text") or ""
                    fn = p0.get("functionCall")
    except Exception:
        pass
    return text or "", fn


@router.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request):
    contents = ollama_messages_to_gemini_contents(req.messages)
    generation_config = options_to_generation_config(req.options)
    payload = GeminiGenerateContentRequest(
        contents=contents, generationConfig=generation_config, tools=normalize_tools(req.tools)
    )

    if req.stream:
        async def sse_gen():
            stream_iter = call_gemini(
                model=req.model, payload=payload, request=request, stream=True
            )
            async for line in stream_iter:
                text, fn = _parse_gemini_stream_line(line)
                if not text and not fn:
                    continue
                import json as _json

                event = {
                    "model": req.model,
                    "created_at": _now_iso(),
                    "message": {"role": "assistant", "content": text or ""},
                    "done": False,
                }
                if fn:
                    event["tool_calls"] = [{"type": "function", **fn}]
                yield ("data: " + _json.dumps(event) + "\n\n").encode()

            # done event
            import json as _json
            done_obj = {
                "model": req.model,
                "created_at": _now_iso(),
                "message": {"role": "assistant", "content": ""},
                "done": True,
            }
            yield ("data: " + _json.dumps(done_obj) + "\n\n").encode()

        return StreamingResponse(sse_gen(), media_type="text/event-stream")

    data = await call_gemini(model=req.model, payload=payload, request=request)
    gem = GeminiResponse(**data)

    text = ""
    tool_calls = []
    if gem.candidates:
        parts = gem.candidates[0].content.parts
        if parts:
            first = parts[0]
            if first.text:
                text = first.text
            if first.functionCall:
                tool_calls.append({"type": "function", **first.functionCall})

    return {
        "model": req.model,
        "created_at": _now_iso(),
        "message": {"role": "assistant", "content": text},
        "done": True,
        "tool_calls": tool_calls or None,
    }


@router.post("/api/generate")
async def api_generate(req: GenerateRequest, request: Request):
    contents = ollama_messages_to_gemini_contents(
        [Message(role="user", content=req.prompt)]
    )
    generation_config = options_to_generation_config(req.options)
    payload = GeminiGenerateContentRequest(
        contents=contents, generationConfig=generation_config, tools=normalize_tools(req.tools)
    )

    if req.stream:
        async def sse_gen():
            stream_iter = call_gemini(
                model=req.model, payload=payload, request=request, stream=True
            )
            async for line in stream_iter:
                text, fn = _parse_gemini_stream_line(line)
                if not text and not fn:
                    continue
                import json as _json

                event = {
                    "model": req.model,
                    "created_at": _now_iso(),
                    "response": text or "",
                    "done": False,
                }
                if fn:
                    event["tool_calls"] = [{"type": "function", **fn}]
                yield ("data: " + _json.dumps(event) + "\n\n").encode()

            # done event
            import json as _json
            done_obj = {
                "model": req.model,
                "created_at": _now_iso(),
                "response": "",
                "done": True,
            }
            yield ("data: " + _json.dumps(done_obj) + "\n\n").encode()

        return StreamingResponse(sse_gen(), media_type="text/event-stream")

    data = await call_gemini(model=req.model, payload=payload, request=request)
    gem = GeminiResponse(**data)

    text = ""
    tool_calls = []
    if gem.candidates:
        parts = gem.candidates[0].content.parts
        if parts:
            first = parts[0]
            if first.text:
                text = first.text
            if first.functionCall:
                tool_calls.append({"type": "function", **first.functionCall})

    return {
        "model": req.model,
        "created_at": _now_iso(),
        "response": text,
        "done": True,
        "tool_calls": tool_calls or None,
    }


@router.get("/api/tags")
async def api_tags():
    # Emulate Ollama /api/tags listing
    # When registry is empty, return the passthrough default model/version
    items = []
    now = _now_iso()
    if settings.models:
        for alias, mc in settings.models.items():
            items.append(
                {
                    "name": alias,
                    "model": mc.model_id,
                    "modified_at": mc.modified or now,
                    "details": {
                        "display_name": mc.display_name or alias,
                        "description": mc.description or "",
                        "api_version": mc.api_version,
                        "base_url": mc.base_url,
                    },
                }
            )
    else:
        items.append(
            {
                "name": "gemini-default",
                "model": "gemini-1.5-pro",
                "modified_at": now,
                "details": {
                    "api_version": settings.gemini_api_version,
                    "base_url": settings.gemini_base_url,
                },
            }
        )
    return {"models": items}
