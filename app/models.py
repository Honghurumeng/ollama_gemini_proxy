from __future__ import annotations

from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field


# --- Ollama-compatible request/response models (subset) ---


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    format: Optional[str] = None
    options: Optional[dict[str, Any]] = None
    # Tool calling (Gemini function declarations)
    tools: Optional[List[dict[str, Any]]] = None
    # Ollama context passthrough (unused by Gemini)
    context: Optional[List[int]] = None


class ChatResponse(BaseModel):
    model: str
    created_at: str
    message: Message
    done: bool = True
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    context: Optional[List[int]] = None


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: Optional[bool] = False
    format: Optional[str] = None
    options: Optional[dict[str, Any]] = None
    tools: Optional[List[dict[str, Any]]] = None
    context: Optional[List[int]] = None


class GenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool = True
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    context: Optional[List[int]] = None


# --- Gemini payloads (simplified) ---


class GeminiContentPart(BaseModel):
    text: Optional[str] = None
    # Tool calling fields
    functionCall: Optional[dict[str, Any]] = None
    functionResponse: Optional[dict[str, Any]] = None


class GeminiContent(BaseModel):
    role: Optional[str] = None
    parts: List[GeminiContentPart]


class GeminiGenerateContentRequest(BaseModel):
    contents: List[GeminiContent]
    generationConfig: Optional[dict[str, Any]] = None
    safetySettings: Optional[List[dict[str, Any]]] = None
    tools: Optional[List[dict[str, Any]]] = None
    toolConfig: Optional[dict[str, Any]] = None


class GeminiCandidate(BaseModel):
    content: GeminiContent
    finishReason: Optional[str] = None


class GeminiResponse(BaseModel):
    candidates: List[GeminiCandidate] = Field(default_factory=list)
