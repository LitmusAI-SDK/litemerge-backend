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

    api_key_header: str = "x-api-key"
    bootstrap_api_key: str = "lmai_dev_key"
    auth_exempt_paths: list[str] = Field(
        default_factory=lambda: ["/v1/health", "/docs", "/openapi.json", "/redoc"]
    )

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
