import pytest
from pydantic import ValidationError

from api.schemas.projects import AuthConfigInput, ProjectPatchRequest


def test_auth_config_rejects_missing_value_for_bearer() -> None:
    with pytest.raises(ValidationError):
        AuthConfigInput(type="bearer")


def test_auth_config_rejects_missing_header_for_apikey() -> None:
    with pytest.raises(ValidationError):
        AuthConfigInput(type="apikey", value="secret")


def test_auth_config_accepts_none_type_without_value() -> None:
    config = AuthConfigInput(type="none")
    assert config.type == "none"
    assert config.value is None


def test_project_patch_requires_at_least_one_field() -> None:
    with pytest.raises(ValidationError):
        ProjectPatchRequest()
