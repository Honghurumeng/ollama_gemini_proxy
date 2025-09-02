from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import router
from .logger import get_logger, truncate_text, mask_secret


app = FastAPI(title="Ollama-to-Gemini Proxy")
log = get_logger("app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(router)


@app.get("/")
async def root():
    return {"status": "ok", "service": "ollama->gemini proxy"}


def run():
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        body = await request.body()
    except Exception:
        body = b""

    # Mask secrets in headers
    headers = dict(request.headers)
    for k in list(headers.keys()):
        if k.lower() in {"authorization", "x-gemini-api-key"}:
            headers[k] = mask_secret(headers.get(k))

    # Query params (mask potential secret-like keys) and build a safe query string
    secret_q_keys = {"key", "api_key", "x-gemini-api-key", "authorization"}
    query_params = dict(request.query_params)
    for k in list(query_params.keys()):
        if k.lower() in secret_q_keys:
            query_params[k] = mask_secret(query_params.get(k))
    try:
        from urllib.parse import urlencode

        safe_items = []
        for k, v in request.query_params.multi_items():
            safe_items.append((k, mask_secret(v) if k.lower() in secret_q_keys else v))
        query_string_safe = urlencode(safe_items)
    except Exception:
        query_string_safe = ""

    path_with_query = request.url.path + ("?" + query_string_safe if query_string_safe else "")

    log.info(
        f"REQ {request.method} {path_with_query}",
        extra={
            "extra_data": {
                "url": str(request.url),
                "query": query_params,
                "headers": headers,
                "body": truncate_text(body.decode(errors="ignore")),
            }
        },
    )

    response = await call_next(request)

    # 不读取 StreamingResponse 的 body，避免消费掉迭代器影响实际响应。
    log.info(
        f"RES {request.method} {path_with_query} {response.status_code}",
        extra={
            "extra_data": {
                "url": str(request.url),
                "query": query_params,
                "headers": dict(response.headers),
                "content_type": response.headers.get("content-type"),
            }
        },
    )
    return response
