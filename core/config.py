from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "LitmusAI Backend"
    app_env: str = "development"

    # Use a per-environment secret in production.
    secrets_encryption_key: str = "change-me-in-production"

    mongodb_url: str = "mongodb://mongo:27017"
    mongodb_db: str = "litmusai"
    redis_url: str = "redis://redis:6379/0"

    public_api_base_url: str = "http://localhost:8000"

    bootstrap_api_key: str = "lmai_dev_key"
    auth_exempt_paths: list[str] = Field(
        default_factory=lambda: ["/v1/health", "/docs", "/openapi.json", "/redoc"]
    )

    # ── LLM routing ──────────────────────────────────────────────────────────
    # llm_provider controls which provider branch is used for caching logic.
    # llm_model is the full litellm model string passed to acompletion().
    # Examples:
    #   gemini    → "gemini/gemini-1.5-flash"
    #   anthropic → "anthropic/claude-haiku-4-5-20251001"
    #   openai    → "openai/gpt-4o-mini"
    #   lmstudio  → uses lmstudio_base_url + lmstudio_model instead of llm_model
    llm_provider: str = "openai"
    llm_model: str = "openai/gpt-4o-mini"

    # ── Provider API keys ─────────────────────────────────────────────────────
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # ── LM Studio local endpoint ──────────────────────────────────────────────
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_model: str = "gemma4"

    # ── Retry strategy ────────────────────────────────────────────────────────
    retry_max_attempts: int = 3
    retry_min_wait_s: float = 4.0
    retry_max_wait_s: float = 10.0

    # ── Persona profiles directory ────────────────────────────────────────────
    persona_profiles_dir: str = "personas/profiles"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @field_validator("auth_exempt_paths", mode="before")
    @classmethod
    def _parse_auth_exempt_paths(cls, value):
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("secrets_encryption_key")
    @classmethod
    def _validate_secrets_key(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("secrets_encryption_key must not be empty")
        return value


settings = Settings()

# Apply litellm global config after settings are loaded.
# Wrapped in try/except so tests that don't install litellm still pass.
try:
    import os

    import litellm

    litellm.openai_key = settings.openai_api_key
    litellm.anthropic_key = settings.anthropic_api_key
    if settings.gemini_api_key:
        # LiteLLM's gemini/ provider reads GEMINI_API_KEY from the environment.
        # Setting litellm.api_key is a generic field and is NOT picked up by
        # the Gemini provider — os.environ is the correct injection point.
        os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
except ImportError:
    pass
