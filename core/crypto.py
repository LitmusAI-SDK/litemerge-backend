import base64
import hashlib
import secrets

from core.config import settings


def _derived_key() -> bytes:
    return hashlib.sha256(settings.secrets_encryption_key.encode("utf-8")).digest()


def _keystream(seed: bytes, length: int) -> bytes:
    blocks: list[bytes] = []
    counter = 0

    while sum(len(block) for block in blocks) < length:
        blocks.append(hashlib.sha256(seed + counter.to_bytes(4, "big")).digest())
        counter += 1

    return b"".join(blocks)[:length]


def encrypt_secret(value: str) -> str:
    plaintext = value.encode("utf-8")
    nonce = secrets.token_bytes(16)
    stream = _keystream(_derived_key() + nonce, len(plaintext))
    ciphertext = bytes(p ^ s for p, s in zip(plaintext, stream))
    return base64.urlsafe_b64encode(nonce + ciphertext).decode("utf-8")


def decrypt_secret(token: str) -> str:
    raw = base64.urlsafe_b64decode(token.encode("utf-8"))
    if len(raw) < 16:
        raise ValueError("Invalid encrypted token")

    nonce = raw[:16]
    ciphertext = raw[16:]
    stream = _keystream(_derived_key() + nonce, len(ciphertext))
    plaintext = bytes(c ^ s for c, s in zip(ciphertext, stream))
    return plaintext.decode("utf-8")
