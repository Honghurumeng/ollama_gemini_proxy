from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional


def _bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_JSON = _bool_env("LOG_JSON", False)
LOG_MAX_BODY = int(os.getenv("LOG_MAX_BODY", "1000"))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload: Dict[str, Any] = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, "extra_data"):
            try:
                payload.update(getattr(record, "extra_data"))
            except Exception:
                payload["extra_data_error"] = "failed to merge extra_data"
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    if LOG_JSON:
        handler.setFormatter(JsonFormatter())
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(LOG_LEVEL)


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def mask_secret(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    v = str(value)
    if len(v) <= 8:
        return "***" + v[-2:]
    return v[:6] + "..." + v[-2:]


def truncate_text(text: Any, max_len: Optional[int] = None) -> str:
    try:
        s = text if isinstance(text, str) else str(text)
    except Exception:
        s = "<non-str>"
    lim = LOG_MAX_BODY if max_len is None else max_len
    if len(s) > lim:
        return s[:lim] + "…(truncated)"
    return s

