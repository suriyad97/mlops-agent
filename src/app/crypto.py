"""PAT encryption at rest (Fernet). Key comes from PLATFORM_SECRET_KEY in .env."""
import os

from cryptography.fernet import Fernet

from src.shared.exceptions import ConfigurationError

_fernet: Fernet | None = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        key = os.getenv("PLATFORM_SECRET_KEY", "")
        if not key:
            raise ConfigurationError(
                "PLATFORM_SECRET_KEY missing — generate one with "
                "python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\" "
                "and add it to .env"
            )
        _fernet = Fernet(key.encode())
    return _fernet


def encrypt(plaintext: str) -> str:
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    return _get_fernet().decrypt(ciphertext.encode()).decode()
