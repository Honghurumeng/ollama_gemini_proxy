from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Query
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


def _parse_gemini_stream_response(lines: list[str]) -> tuple[str, Optional[dict]]:
    """Parse accumulated lines to extract text delta or functionCall from a Gemini response.

    Returns (text, function_call_dict_or_None).
    """
    import json as _json
    from .logger import get_logger
    
    log = get_logger("gemini_parser")

    # Join all lines to form complete JSON
    full_response = ''.join(lines)
    log.info(f"Parsing complete Gemini response: {full_response[:300]}{'...' if len(full_response) > 300 else ''}")

    try:
        obj = _json.loads(full_response)
        log.info(f"Successfully parsed JSON object, type: {type(obj)}")
    except Exception as e:
        log.warning(f"Failed to parse JSON: {e}")
        return "", None

    text = ""
    fn = None
    try:
        # Handle JSON array of streaming blocks (Gemini's actual format)
        if isinstance(obj, list):
            log.info(f"Processing {len(obj)} streaming blocks")
            
            # Collect text from ALL streaming blocks, not just the first one  
            all_text_parts = []
            
            for i, block in enumerate(obj):
                if not isinstance(block, dict):
                    continue
                    
                cands = block.get("candidates") or []
                log.debug(f"Block {i}: {len(cands)} candidates")
                
                for candidate in cands:
                    parts = ((candidate.get("content") or {}).get("parts") or [])
                    
                    for part in parts:
                        if isinstance(part, dict):
                            part_text = part.get("text")
                            if part_text:
                                all_text_parts.append(part_text)
                                log.debug(f"Block {i}: extracted '{part_text[:100]}{'...' if len(part_text) > 100 else ''}'")
                            
                            # Check for function call in any part
                            part_fn = part.get("functionCall")
                            if part_fn:
                                fn = part_fn
                                log.info(f"Block {i}: found functionCall")
            
            # Combine all text parts from all blocks
            text = ''.join(all_text_parts)
            log.info(f"Combined {len(all_text_parts)} text parts from {len(obj)} blocks: total {len(text)} chars")
        else:
            # Handle single object (fallback for non-streaming responses)
            log.info("Processing single object response")
            cands = obj.get("candidates") or []
            
            text_parts = []
            for candidate in cands:
                parts = ((candidate.get("content") or {}).get("parts") or [])
                
                for part in parts:
                    if isinstance(part, dict):
                        part_text = part.get("text")
                        if part_text:
                            text_parts.append(part_text)
                        
                        part_fn = part.get("functionCall")
                        if part_fn:
                            fn = part_fn
            
            text = ''.join(text_parts)
    except Exception as e:
        log.error(f"Error extracting content from Gemini response: {e}")
        pass
    
    result_text = text or ""
    log.info(f"Final result: {len(result_text)} chars - '{result_text[:200]}{'...' if len(result_text) > 200 else ''}'")
    return result_text, fn


def _parse_gemini_stream_line(line: str) -> tuple[str, Optional[dict]]:
    """Extract text delta or functionCall from a Gemini stream line.

    Returns (text, function_call_dict_or_None).
    """
    import json as _json
    from .logger import get_logger
    
    log = get_logger("stream_line_parser")

    # Some servers may prepend 'data: ' for SSE framing; strip it if present.
    if line.startswith("data: "):
        line = line[len("data: "):]

    if not line.strip():
        return "", None
        
    log.debug(f"Parsing line: {line[:200]}{'...' if len(line) > 200 else ''}")

    try:
        obj = _json.loads(line)
        log.debug(f"Successfully parsed JSON: {type(obj)}")
    except Exception as e:
        log.debug(f"Failed to parse JSON: {e}")
        return "", None

    text = ""
    fn = None
    try:
        cands = obj.get("candidates") or []
        log.debug(f"Found {len(cands)} candidates")
        if cands:
            parts = ((cands[0].get("content") or {}).get("parts") or [])
            log.debug(f"Found {len(parts)} parts")
            
            # Process ALL parts, not just the first one
            text_parts = []
            for i, part in enumerate(parts):
                if isinstance(part, dict):
                    part_text = part.get("text")
                    if part_text:
                        text_parts.append(part_text)
                        log.debug(f"Part {i}: found text '{part_text[:50]}{'...' if len(part_text) > 50 else ''}'")
                    
                    # Check for function call in any part
                    part_fn = part.get("functionCall")
                    if part_fn:
                        fn = part_fn
                        log.debug(f"Part {i}: found functionCall")
            
            # Combine all text parts
            text = ''.join(text_parts)
            if text_parts:
                log.debug(f"Combined {len(text_parts)} parts: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    except Exception as e:
        log.error(f"Error processing candidates: {e}")
        pass
    
    result = (text or "", fn)
    log.debug(f"Returning: text_len={len(result[0])}, has_fn={result[1] is not None}")
    return result


@router.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request):
    contents = ollama_messages_to_gemini_contents(req.messages)
    generation_config = options_to_generation_config(req.options)
    payload = GeminiGenerateContentRequest(
        contents=contents, generationConfig=generation_config, tools=normalize_tools(req.tools)
    )

    if req.stream and settings.stream_format == "ndjson":
        async def ndjson_gen():
            try:
                stream_iter = await call_gemini(
                    model=req.model, payload=payload, request=request, stream=True
                )
            except Exception as e:
                import json as _json
                err = {"model": req.model, "created_at": _now_iso(), "error": str(e), "done": True, "success": False}
                yield (_json.dumps(err) + "\n").encode()
                return
            
            # Process stream with enhanced buffering and fallback
            accumulated_lines = []
            has_streaming_content = False
            
            from .logger import get_logger
            log = get_logger("stream_processor")
            
            async for line in stream_iter:
                accumulated_lines.append(line)
                log.debug(f"Received stream line: {len(line)} chars - {line[:100]}{'...' if len(line) > 100 else ''}")
                
                # Try real-time parsing first
                text, fn = _parse_gemini_stream_line(line)
                if text or fn:
                    has_streaming_content = True
                    log.info(f"Real-time parsed content: text_len={len(text)}, has_fn={fn is not None}")
                    import json as _json
                    event = {
                        "model": req.model,
                        "created_at": _now_iso(),
                        "message": {"role": "assistant", "content": text or ""},
                        "done": False,
                    }
                    if fn:
                        event["message"]["tool_calls"] = [{"type": "function", **fn}]
                    yield (_json.dumps(event) + "\n").encode()
            
            log.info(f"Stream processing complete: {len(accumulated_lines)} lines, has_streaming_content={has_streaming_content}")
            
            # Enhanced fallback: always try accumulated parsing for complete responses
            if accumulated_lines:
                log.info("Attempting enhanced parsing of all accumulated content")
                text, fn = _parse_gemini_stream_response(accumulated_lines)
                
                # Only yield if we haven't already yielded content, or if this gives us more/better content
                should_yield = not has_streaming_content or (text and len(text) > 50)  # Threshold for "significant" content
                
                if should_yield and (text or fn):
                    log.info(f"Enhanced parsing successful: text_len={len(text)}, has_fn={fn is not None}")
                    import json as _json
                    event = {
                        "model": req.model,
                        "created_at": _now_iso(),
                        "message": {"role": "assistant", "content": text or ""},
                        "done": False,
                    }
                    if fn:
                        event["message"]["tool_calls"] = [{"type": "function", **fn}]
                    yield (_json.dumps(event) + "\n").encode()
                elif not has_streaming_content:
                    log.warning("No content extracted from stream - this may cause Cline errors!")

            # done event
            import json as _json
            done_obj = {
                "model": req.model, 
                "created_at": _now_iso(), 
                "message": {"role": "assistant", "content": ""}, 
                "done": True, 
                "success": True
            }
            yield (_json.dumps(done_obj) + "\n").encode()

        return StreamingResponse(ndjson_gen(), media_type="application/x-ndjson")
    elif req.stream:
        async def sse_gen():
            try:
                stream_iter = await call_gemini(
                    model=req.model, payload=payload, request=request, stream=True
                )
            except Exception as e:
                import json as _json
                err = {"model": req.model, "created_at": _now_iso(), "error": str(e), "done": True, "success": False}
                yield ("data: " + _json.dumps(err) + "\n\n").encode()
                return
            
            # Process stream in real-time for better streaming experience
            async for line in stream_iter:
                text, fn = _parse_gemini_stream_line(line)
                if text or fn:
                    import json as _json
                    event = {
                        "model": req.model,
                        "created_at": _now_iso(),
                        "message": {"role": "assistant", "content": text or ""},
                        "done": False,
                    }
                    if fn:
                        event["message"]["tool_calls"] = [{"type": "function", **fn}]
                    yield ("data: " + _json.dumps(event) + "\n\n").encode()

            # done event
            import json as _json
            done_obj = {
                "model": req.model, 
                "created_at": _now_iso(), 
                "message": {"role": "assistant", "content": ""}, 
                "done": True, 
                "success": True
            }
            yield ("data: " + _json.dumps(done_obj) + "\n\n").encode()

        return StreamingResponse(sse_gen(), media_type="text/event-stream")

    data = await call_gemini(model=req.model, payload=payload, request=request)
    gem = GeminiResponse(**data)

    text = ""
    tool_calls = []
    if gem.candidates:
        parts = gem.candidates[0].content.parts
        # Process ALL parts, not just the first one
        text_parts = []
        for part in parts:
            if part.text:
                text_parts.append(part.text)
            if part.functionCall:
                tool_calls.append({"type": "function", **part.functionCall})
        
        # Combine all text parts
        text = ''.join(text_parts)

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

    if req.stream and settings.stream_format == "ndjson":
        async def ndjson_gen():
            try:
                stream_iter = await call_gemini(
                    model=req.model, payload=payload, request=request, stream=True
                )
            except Exception as e:
                import json as _json
                err = {"model": req.model, "created_at": _now_iso(), "error": str(e), "done": True, "success": False}
                yield (_json.dumps(err) + "\n").encode()
                return
            
            # Process stream with fallback mechanism for better compatibility
            accumulated_lines = []
            has_streaming_content = False
            
            async for line in stream_iter:
                accumulated_lines.append(line)
                
                # Try real-time parsing first
                text, fn = _parse_gemini_stream_line(line)
                if text or fn:
                    has_streaming_content = True
                    import json as _json
                    event = {
                        "model": req.model,
                        "created_at": _now_iso(),
                        "response": text or "",
                        "done": False,
                    }
                    if fn:
                        event["tool_calls"] = [{"type": "function", **fn}]
                    yield (_json.dumps(event) + "\n").encode()
            
            # Fallback: if real-time parsing failed, try accumulated parsing
            if not has_streaming_content and accumulated_lines:
                text, fn = _parse_gemini_stream_response(accumulated_lines)
                if text or fn:
                    import json as _json
                    event = {
                        "model": req.model,
                        "created_at": _now_iso(),
                        "response": text or "",
                        "done": False,
                    }
                    if fn:
                        event["tool_calls"] = [{"type": "function", **fn}]
                    yield (_json.dumps(event) + "\n").encode()

            # done event
            import json as _json
            done_obj = {
                "model": req.model, 
                "created_at": _now_iso(), 
                "response": "", 
                "done": True, 
                "success": True
            }
            yield (_json.dumps(done_obj) + "\n").encode()

        return StreamingResponse(ndjson_gen(), media_type="application/x-ndjson")
    elif req.stream:
        async def sse_gen():
            try:
                stream_iter = await call_gemini(
                    model=req.model, payload=payload, request=request, stream=True
                )
            except Exception as e:
                import json as _json
                err = {"model": req.model, "created_at": _now_iso(), "error": str(e), "done": True, "success": False}
                yield ("data: " + _json.dumps(err) + "\n\n").encode()
                return
            
            # Process stream in real-time for better streaming experience
            async for line in stream_iter:
                text, fn = _parse_gemini_stream_line(line)
                if text or fn:
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
                "success": True
            }
            yield ("data: " + _json.dumps(done_obj) + "\n\n").encode()

        return StreamingResponse(sse_gen(), media_type="text/event-stream")

    data = await call_gemini(model=req.model, payload=payload, request=request)
    gem = GeminiResponse(**data)

    text = ""
    tool_calls = []
    if gem.candidates:
        parts = gem.candidates[0].content.parts
        # Process ALL parts, not just the first one
        text_parts = []
        for part in parts:
            if part.text:
                text_parts.append(part.text)
            if part.functionCall:
                tool_calls.append({"type": "function", **part.functionCall})
        
        # Combine all text parts
        text = ''.join(text_parts)

    return {
        "model": req.model,
        "created_at": _now_iso(),
        "response": text,
        "done": True,
        "tool_calls": tool_calls or None,
    }


@router.get("/api/tags")
async def api_tags():
    # Emulate Ollama /api/tags listing as closely as possible.
    from .logger import get_logger
    log = get_logger("api_tags")
    
    log.info("GET /api/tags called - VS Code Copilot requesting model list")
    
    items = []
    now = _now_iso()
    if settings.models:
        for alias, mc in settings.models.items():
            # Ensure size and digest are properly set for VS Code compatibility
            size = mc.size if mc.size and mc.size > 0 else 815319791  # Use same default as real Ollama
            digest = mc.digest if mc.digest else "sha256:8648f39daa8fbf5b18c7b4e6a8fb4990c692751d49917417b8842ca5758e7ffc"
            
            # Format name with :latest if no tag specified
            model_name = mc.name or f"{alias}:latest"
            
            items.append(
                {
                    "name": model_name,
                    "model": model_name,  # Match real Ollama: model = name
                    "modified_at": mc.modified or now,
                    "size": size,
                    "digest": digest,
                    "details": {
                        "parent_model": mc.parent_model or "",
                        "format": mc.format or "gguf",
                        "family": mc.family or "gemma3",  # Use gemma3 as default like real Ollama
                        "families": mc.families or [mc.family or "gemma3"],
                        "parameter_size": mc.parameter_size or "999.89M",  # Match real Ollama default
                        "quantization_level": mc.quantization_level or "Q4_K_M",
                        # Remove non-standard fields that real Ollama doesn't have
                        # Removed: display_name, description, capabilities, api_version, base_url
                    },
                }
            )
    else:
        items.append(
            {
                "name": "gemini-default:latest",
                "model": "gemini-default:latest",  # Match real Ollama format
                "modified_at": now,
                "size": 815319791,  # Match real Ollama default
                "digest": "sha256:8648f39daa8fbf5b18c7b4e6a8fb4990c692751d49917417b8842ca5758e7ffc",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "gemma3",
                    "families": ["gemma3"],
                    "parameter_size": "999.89M",
                    "quantization_level": "Q4_K_M",
                },
            }
        )
    
    log.info(f"Returning {len(items)} models in /api/tags response")
    return {"models": items}


@router.get("/api/version")
async def api_version():
    # Minimal Ollama compatibility endpoint
    return {"version": settings.ollama_version}


@router.get("/")
async def health_check():
    """Health check endpoint for VS Code Copilot compatibility."""
    return {"status": "ok", "message": "Ollama-compatible server is running"}


@router.get("/v1/models")
async def openai_models():
    """OpenAI-compatible models endpoint for better compatibility."""
    models = []
    if settings.models:
        for alias, mc in settings.models.items():
            model_name = mc.name or f"{alias}:latest"
            models.append({
                "id": model_name,  # Use the full model name with tag
                "object": "model", 
                "created": int(dt.datetime.now().timestamp()),
                "owned_by": "library",  # Match real Ollama
            })
    else:
        models.append({
            "id": "gemini-default:latest",
            "object": "model",
            "created": int(dt.datetime.now().timestamp()), 
            "owned_by": "library",  # Match real Ollama
        })
    
    return {
        "object": "list",
        "data": models
    }


def _build_show_info(alias: str) -> dict:
    now = _now_iso()
    
    # Handle cases where alias might already include :tag (e.g., "gemini25pro:latest")
    # Extract the base alias name for lookup
    base_alias = alias.split(':')[0]
    
    if base_alias in settings.models:
        mc = settings.models[base_alias]
        # Use the original alias (which might already have :tag) or fall back to configured name
        model_name = alias if ':' in alias else (mc.name or f"{base_alias}:latest")
        size = mc.size if mc.size and mc.size > 0 else 815319791  # Match real Ollama default
        digest = mc.digest if mc.digest else "sha256:8648f39daa8fbf5b18c7b4e6a8fb4990c692751d49917417b8842ca5758e7ffc"
        
        # Match real Ollama /api/show response structure exactly
        info = {
            "license": mc.license or "MIT License",
            "modelfile": mc.modelfile or f"# Modelfile for {model_name}\nFROM {model_name}",
            "parameters": mc.parameters or "top_p                          0.95\nstop                           \"<end_of_turn>\"\ntemperature                    1\ntop_k                          64",
            "template": mc.template or "{{- range $i, $_ := .Messages }}\n{{- $last := eq (len (slice $.Messages $i)) 1 }}\n{{- if or (eq .Role \"user\") (eq .Role \"system\") }}<start_of_turn>user\n{{ .Content }}<end_of_turn>\n{{ if $last }}<start_of_turn>model\n{{ end }}\n{{- else if eq .Role \"assistant\" }}<start_of_turn>model\n{{ .Content }}{{ if not $last }}<end_of_turn>\n{{ end }}\n{{- end }}\n{{- end }}",
            "details": {
                "parent_model": mc.parent_model or "",
                "format": mc.format or "gguf",
                "family": mc.family or "gemma3",  # Match real Ollama default
                "families": mc.families or [mc.family or "gemma3"],
                "parameter_size": mc.parameter_size or "999.89M",  # Match real Ollama default
                "quantization_level": mc.quantization_level or "Q4_K_M",
            },
            "model_info": {
                "general.architecture": "gemma3",
                "general.file_type": 15,
                "general.parameter_count": 999885952,
                "general.quantization_version": 2,
                "tokenizer.ggml.model": "llama",
                f"{mc.family or 'gemma3'}.context_length": 32768,
                f"{mc.family or 'gemma3'}.embedding_length": 1152,
                f"{mc.family or 'gemma3'}.block_count": 26,
                f"{mc.family or 'gemma3'}.attention.head_count": 4,
                f"{mc.family or 'gemma3'}.attention.head_count_kv": 1,
            },
            "tensors": [],  # Empty for simplicity, VS Code probably doesn't need this
            "capabilities": ["completion"],  # This is crucial for VS Code!
            "modified_at": mc.modified or now,
        }
        return info
    
    # Fallback: treat alias as a direct model using real Ollama defaults
    # Make sure not to double-add :latest if it's already there
    final_name = alias if ':' in alias else f"{alias}:latest"
    return {
        "license": "MIT License",
        "modelfile": f"# Modelfile for {final_name}\nFROM {final_name}",
        "parameters": "top_p                          0.95\nstop                           \"<end_of_turn>\"\ntemperature                    1\ntop_k                          64",
        "template": "{{- range $i, $_ := .Messages }}\n{{- $last := eq (len (slice $.Messages $i)) 1 }}\n{{- if or (eq .Role \"user\") (eq .Role \"system\") }}<start_of_turn>user\n{{ .Content }}<end_of_turn>\n{{ if $last }}<start_of_turn>model\n{{ end }}\n{{- else if eq .Role \"assistant\" }}<start_of_turn>model\n{{ .Content }}{{ if not $last }}<end_of_turn>\n{{ end }}\n{{- end }}\n{{- end }}",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "gemma3",
            "families": ["gemma3"],
            "parameter_size": "999.89M",
            "quantization_level": "Q4_K_M",
        },
        "model_info": {
            "general.architecture": "gemma3",
            "general.file_type": 15,
            "general.parameter_count": 999885952,
            "general.quantization_version": 2,
            "tokenizer.ggml.model": "llama",
            "gemma3.context_length": 32768,
            "gemma3.embedding_length": 1152,
            "gemma3.block_count": 26,
            "gemma3.attention.head_count": 4,
            "gemma3.attention.head_count_kv": 1,
        },
        "tensors": [],
        "capabilities": ["completion"],  # This is crucial for VS Code!
        "modified_at": now,
    }


@router.get("/api/show")
async def api_show_get(model: str | None = Query(default=None)):
    """Compatibility: show model info. Requires model parameter."""
    from .logger import get_logger
    log = get_logger("api_show")
    
    log.info(f"GET /api/show called with model parameter: '{model}'")
    
    if not model:
        # Match real Ollama behavior: return plain text 405
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse("405 method not allowed", status_code=405)
    
    result = _build_show_info(model)
    log.info(f"Returning show info for model '{model}': {str(result)[:200]}...")
    return result


@router.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint for VS Code Copilot."""
    from .logger import get_logger
    log = get_logger("openai_chat")
    
    try:
        payload = await request.json()
    except Exception as e:
        log.error(f"Failed to parse JSON payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    log.info(f"POST /v1/chat/completions called with payload keys: {list(payload.keys())}")
    
    # Extract OpenAI format parameters
    model = payload.get("model", "gemini25pro")
    messages = payload.get("messages", [])
    stream = payload.get("stream", False)
    temperature = payload.get("temperature", 0.7)
    max_tokens = payload.get("max_tokens")
    top_p = payload.get("top_p")
    
    # Convert OpenAI messages to our format
    ollama_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        ollama_messages.append(Message(role=role, content=content))
    
    log.info(f"Converted {len(messages)} OpenAI messages to Ollama format for model '{model}'")
    
    # Convert to Gemini format
    contents = ollama_messages_to_gemini_contents(ollama_messages)
    
    # Build generation config from OpenAI parameters
    generation_config = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if max_tokens is not None:
        generation_config["maxOutputTokens"] = max_tokens
    if top_p is not None:
        generation_config["topP"] = top_p
    
    gemini_payload = GeminiGenerateContentRequest(
        contents=contents, 
        generationConfig=generation_config or None,
        tools=None  # VS Code Copilot usually doesn't use tools in chat
    )
    
    if stream:
        # Return streaming response in OpenAI format
        async def openai_stream():
            import json as _json
            import uuid
            
            try:
                stream_iter = await call_gemini(
                    model=model, payload=gemini_payload, request=request, stream=True
                )
            except Exception as e:
                log.error(f"Error calling Gemini: {e}")
                error_response = {
                    "error": {
                        "message": str(e),
                        "type": "gemini_error",
                        "code": "service_unavailable"
                    }
                }
                yield f"data: {_json.dumps(error_response)}\n\n"
                return
            
            # Accumulate lines to form complete JSON (handle formatted response)
            accumulated_lines = []
            async for line in stream_iter:
                accumulated_lines.append(line)
            
            # Parse the complete response
            text, fn = _parse_gemini_stream_response(accumulated_lines)
            
            if text:
                # Generate OpenAI-compatible streaming response
                completion_id = f"chatcmpl-{uuid.uuid4().hex}"
                
                # Send the text content
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk", 
                    "created": int(dt.datetime.now().timestamp()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None
                    }]
                }
                yield f"data: {_json.dumps(chunk)}\n\n"
                
                # Send final chunk
                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(dt.datetime.now().timestamp()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {_json.dumps(final_chunk)}\n\n"
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(openai_stream(), media_type="text/event-stream")
    
    else:
        # Non-streaming response
        import uuid
        
        try:
            data = await call_gemini(model=model, payload=gemini_payload, request=request)
            gem = GeminiResponse(**data)
            
            text = ""
            if gem.candidates:
                parts = gem.candidates[0].content.parts
                if parts and parts[0].text:
                    text = parts[0].text
            
            log.info(f"Returning OpenAI chat completion with {len(text)} characters")
            
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(dt.datetime.now().timestamp()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,  # Gemini doesn't provide token counts
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            
        except Exception as e:
            log.error(f"Error in non-streaming chat completion: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/show")
async def api_show_post(payload: dict):
    from .logger import get_logger
    log = get_logger("api_show")
    
    log.info(f"POST /api/show called with payload: {payload}")
    
    model = payload.get("model") or payload.get("name")
    if not model:
        log.warning("POST /api/show called without model/name in payload")
        raise HTTPException(status_code=400, detail="model is required")
    
    result = _build_show_info(model)
    log.info(f"Returning show info for model '{model}': {str(result)[:200]}...")
    return result
