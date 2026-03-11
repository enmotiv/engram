"""Hashing utilities."""

import hashlib


def hash_api_key(raw_key: str) -> str:
    """SHA-256 hash of a plaintext API key. Never store or log the raw key."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
