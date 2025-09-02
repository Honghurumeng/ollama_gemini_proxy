from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from dotenv import load_dotenv


DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_GEMINI_API_VERSION = "v1beta"


def _slug_env(alias: str) -> str:
    return (
        alias.upper()
        .replace("-", "_")
        .replace(".", "_")
        .replace(":", "_")
        .replace("/", "_")
    )


@dataclass
class ModelConfig:
    alias: str
    model_id: str
    base_url: str
    api_version: str
    api_key: Optional[str]
    display_name: Optional[str] = None
    description: Optional[str] = None
    modified: Optional[str] = None


class Settings:
    """Runtime configuration loaded from .env and environment variables.

    Priority order for API base URL and key:
    - Request headers (handled in request-time resolver)
    - Per-model env overrides (MODEL_<ALIAS>_*)
    - Global env vars (GEMINI_*)
    - Defaults (for base url/version)
    """

    def __init__(self) -> None:
        load_dotenv(override=False)

        self.gemini_base_url: str = os.getenv(
            "GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL
        ).rstrip("/")
        self.gemini_api_version: str = os.getenv(
            "GEMINI_API_VERSION", DEFAULT_GEMINI_API_VERSION
        ).strip()
        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")

        # Server config
        self.port: int = int(os.getenv("PORT", "8000"))
        self.host: str = os.getenv("HOST", "0.0.0.0")
        # Ollama compatibility version for /api/version
        self.ollama_version: str = os.getenv("OLLAMA_VERSION", "0.1.32")

        # Model registry
        self.models: Dict[str, ModelConfig] = self._load_models_from_env()

    def _load_models_from_env(self) -> Dict[str, ModelConfig]:
        registry: Dict[str, ModelConfig] = {}
        raw = os.getenv("MODELS", "").strip()
        if raw:
            aliases = [a.strip() for a in raw.split(",") if a.strip()]
            for alias in aliases:
                key = _slug_env(alias)
                model_id = os.getenv(f"MODEL_{key}_ID", alias)
                base_url = os.getenv(
                    f"MODEL_{key}_BASE_URL", self.gemini_base_url
                ).rstrip("/")
                api_version = os.getenv(
                    f"MODEL_{key}_API_VERSION", self.gemini_api_version
                ).strip()
                api_key = os.getenv(f"MODEL_{key}_API_KEY", self.gemini_api_key)
                display_name = os.getenv(f"MODEL_{key}_DISPLAY_NAME")
                description = os.getenv(f"MODEL_{key}_DESCRIPTION")
                modified = os.getenv(f"MODEL_{key}_MODIFIED")

                registry[alias] = ModelConfig(
                    alias=alias,
                    model_id=model_id,
                    base_url=base_url,
                    api_version=api_version,
                    api_key=api_key,
                    display_name=display_name,
                    description=description,
                    modified=modified,
                )
        return registry


settings = Settings()


def resolve_gemini_config(
    headers: dict[str, str], model_alias: Optional[str] = None
) -> Tuple[str, str, Optional[str], str]:
    """Resolve Gemini endpoint base url, version, api key, and model id.

    Header overrides (highest priority):
    - X-Gemini-Base-Url
    - X-Gemini-Api-Version
    - X-Gemini-Api-Key

    Then per-model env (MODEL_<ALIAS>_*) if alias is provided.
    Then global env (GEMINI_*), then defaults.
    The model id defaults to the provided alias if not specified.
    """

    base_url_hdr = headers.get("x-gemini-base-url") or headers.get("X-Gemini-Base-Url")
    api_version_hdr = headers.get("x-gemini-api-version") or headers.get(
        "X-Gemini-Api-Version"
    )
    api_key_hdr = headers.get("x-gemini-api-key") or headers.get("X-Gemini-Api-Key")

    model_id = model_alias or ""
    base_url = settings.gemini_base_url
    api_version = settings.gemini_api_version
    api_key = settings.gemini_api_key

    if model_alias and model_alias in settings.models:
        mc = settings.models[model_alias]
        model_id = mc.model_id or model_alias
        base_url = mc.base_url or base_url
        api_version = mc.api_version or api_version
        api_key = mc.api_key or api_key
    else:
        model_id = model_alias or model_id

    base_url = (base_url_hdr or base_url).rstrip("/")
    api_version = (api_version_hdr or api_version).strip()
    api_key = api_key_hdr or api_key

    return base_url, api_version, api_key, model_id
