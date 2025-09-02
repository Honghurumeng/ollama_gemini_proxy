from __future__ import annotations

import httpx
import os
import json
from datetime import datetime
from typing import AsyncIterator, Optional
from .logger import get_logger, truncate_text, mask_secret


class GeminiClient:
    def __init__(self, base_url: str, version: str, api_key: Optional[str]) -> None:
        self.base_url = base_url.rstrip("/")
        self.version = version
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
        self._log = get_logger("gemini")
        self._debug_save = os.getenv("DEBUG_SAVE_RESPONSES", "").lower() in {"1", "true", "yes", "on"}
        
    def _save_debug_response(self, response_text: str, is_stream: bool = False) -> None:
        """保存调试响应到文件"""
        if not self._debug_save:
            return
            
        try:
            # 创建debug目录
            debug_dir = "debug_responses"
            os.makedirs(debug_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 包含毫秒
            stream_suffix = "_stream" if is_stream else "_non_stream"
            filename = f"{debug_dir}/gemini_response_{timestamp}{stream_suffix}.json"
            
            # 保存响应
            with open(filename, "w", encoding="utf-8") as f:
                f.write(response_text)
            
            self._log.info(f"DEBUG: Saved Gemini response to {filename} ({len(response_text)} characters)")
            
        except Exception as e:
            self._log.error(f"Failed to save debug response: {e}")

    def _endpoint_for_model(self, model: str, *, stream: bool = False) -> str:
        # Gemini path: /{version}/models/{model}:generateContent or streamGenerateContent
        # Some gateways expect raw model id without alias-style tags like ":latest".
        safe_model = model.split(":", 1)[0].split("@", 1)[0]
        action = "streamGenerateContent" if stream else "generateContent"
        return f"{self.base_url}/{self.version}/models/{safe_model}:{action}"

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
        
        # 保存调试响应
        try:
            response_text = resp.text
            self._save_debug_response(response_text, is_stream=False)
        except Exception as e:
            self._log.error(f"Failed to access response text for debug: {e}")
        
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
            
            # 收集完整响应用于调试
            full_response_parts = []
            
            # Accumulate partial data
            buffer = ""
            async for chunk in resp.aiter_text():
                buffer += chunk
                # Split by lines but keep incomplete lines in buffer
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        full_response_parts.append(line)
                        yield line
            
            # Yield any remaining content in buffer
            if buffer.strip():
                self._log.info(f"Yielding final buffer content: {len(buffer)} chars")
                full_response_parts.append(buffer)
                yield buffer
                
            # 保存完整的调试响应
            if self._debug_save and full_response_parts:
                full_response = '\n'.join(full_response_parts)
                self._save_debug_response(full_response, is_stream=True)
                
        self._log.info(f"STREAM END {url}")

    async def aclose(self) -> None:
        await self._client.aclose()
