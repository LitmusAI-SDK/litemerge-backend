from datetime import datetime
from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, Field, model_validator

AMBER_LATENCY_MS: float = 2000.0

AuthType = Literal["bearer", "apikey", "basic", "none"]


class AuthConfigInput(BaseModel):
    type: AuthType
    value: str | None = Field(default=None, min_length=1, max_length=4096)
    header_name: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def _validate_shape(self):
        if self.type == "none":
            if self.value is not None:
                raise ValueError("value must be omitted when auth type is 'none'")
            if self.header_name is not None:
                raise ValueError("header_name must be omitted when auth type is 'none'")
            return self

        if not self.value:
            raise ValueError("value is required for auth type bearer|apikey|basic")

        if self.type == "apikey" and not self.header_name:
            raise ValueError("header_name is required when auth type is 'apikey'")

        if self.type in {"bearer", "basic"} and self.header_name is not None:
            raise ValueError("header_name is only valid for auth type 'apikey'")

        return self


class AuthConfigPublic(BaseModel):
    type: AuthType
    header_name: str | None = None
    has_value: bool = False


class ProjectCreateRequest(BaseModel):
    id: str | None = Field(default=None, min_length=5, max_length=72)
    name: str = Field(min_length=2, max_length=128)
    agent_endpoint: AnyHttpUrl
    auth_config: AuthConfigInput
    owner_id: str = Field(min_length=2, max_length=128)
    schema_hints: dict[str, str] | None = None


class ProjectPatchRequest(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=128)
    agent_endpoint: AnyHttpUrl | None = None
    auth_config: AuthConfigInput | None = None
    owner_id: str | None = Field(default=None, min_length=2, max_length=128)
    schema_hints: dict[str, str] | None = None

    @model_validator(mode="after")
    def _validate_non_empty(self):
        if all(
            value is None
            for value in [
                self.name,
                self.agent_endpoint,
                self.auth_config,
                self.owner_id,
                self.schema_hints,
            ]
        ):
            raise ValueError("At least one updatable field must be provided")
        return self


class ProjectResponse(BaseModel):
    id: str
    name: str
    agent_endpoint: str
    auth_config: AuthConfigPublic
    owner_id: str
    schema_hints: dict[str, str] | None = None
    created_at: datetime
    updated_at: datetime


class ProjectListResponse(BaseModel):
    items: list[ProjectResponse]


class PreflightResponse(BaseModel):
    status: Literal["green", "amber", "red"]
    latency_ms: float
    error: str | None = None
