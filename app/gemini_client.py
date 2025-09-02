from __future__ import annotations

import httpx
from typing import AsyncIterator, Optional
from .logger import get_logger, truncate_text, mask_secret


class GeminiClient:
    def __init__(self, base_url: str, version: str, api_key: Optional[str]) -> None:
        self.base_url = base_url.rstrip("/")
        self.version = version
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
        self._log = get_logger("gemini")

    def _endpoint_for_model(self, model: str, *, stream: bool = False) -> str:
        # Gemini path: /{version}/models/{model}:generateContent or streamGenerateContent
        action = "streamGenerateContent" if stream else "generateContent"
        return f"{self.base_url}/{self.version}/models/{model}:{action}"

    async def generate_content(self, model: str, payload: dict) -> httpx.Response:
        params = {}
        headers = {}
        if self.api_key:
            params["key"] = self.api_key

        url = self._endpoint_for_model(model, stream=False)
        self._log.info(
            f"POST {url}",
            extra={
                "extra_data": {
                    "params": {**params, "key": mask_secret(params.get("key")) if "key" in params else None},
                    "payload": truncate_text(payload),
                }
            },
        )
        resp = await self._client.post(url, json=payload, params=params, headers=headers)
        try:
            self._log.info(
                f"RESP {resp.status_code} {url}",
                extra={
                    "extra_data": {
                        "headers": dict(resp.headers),
                        "body": truncate_text(resp.text),
                    }
                },
            )
        except Exception:
            self._log.info(f"RESP {resp.status_code} {url} <unreadable body>")
        return resp

    async def stream_generate_content(self, model: str, payload: dict) -> AsyncIterator[str]:
        params = {}
        headers = {}
        if self.api_key:
            params["key"] = self.api_key

        url = self._endpoint_for_model(model, stream=True)
        self._log.info(
            f"STREAM POST {url}",
            extra={
                "extra_data": {
                    "params": {**params, "key": mask_secret(params.get("key")) if "key" in params else None},
                    "payload": truncate_text(payload),
                }
            },
        )
        async with self._client.stream("POST", url, json=payload, params=params, headers=headers) as resp:
            resp.raise_for_status()
            self._log.info(f"STREAM OPEN {url} status={resp.status_code}")
            async for line in resp.aiter_lines():
                if line:
                    yield line
        self._log.info(f"STREAM END {url}")

    async def aclose(self) -> None:
        await self._client.aclose()
