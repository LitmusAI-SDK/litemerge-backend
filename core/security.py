import hashlib
import secrets


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def generate_run_id() -> str:
    return f"run_{secrets.token_hex(6)}"
