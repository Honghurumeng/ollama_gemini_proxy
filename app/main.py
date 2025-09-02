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

    log.info(
        f"REQ {request.method} {request.url.path}",
        extra={
            "extra_data": {
                "headers": headers,
                "body": truncate_text(body.decode(errors="ignore")),
            }
        },
    )

    response = await call_next(request)

    # 不读取 StreamingResponse 的 body，避免消费掉迭代器影响实际响应。
    log.info(
        f"RES {request.method} {request.url.path} {response.status_code}",
        extra={
            "extra_data": {
                "headers": dict(response.headers),
                "content_type": response.headers.get("content-type"),
            }
        },
    )
    return response
