from core.crypto import decrypt_secret, encrypt_secret


def test_encrypt_decrypt_round_trip() -> None:
    raw = "super-secret-token"
    encrypted = encrypt_secret(raw)

    assert encrypted != raw
    assert decrypt_secret(encrypted) == raw
