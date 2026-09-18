"""Tiny TTL cache (stdlib) to avoid repeat Sentinel Hub spend."""
from __future__ import annotations

import time


class TTLCache:
    def __init__(self, ttl_s: float = 86_400.0):
        self.ttl_s = ttl_s
        self._store: dict[str, tuple[float, object]] = {}

    def get(self, key: str):
        hit = self._store.get(key)
        if hit is None:
            return None
        exp, value = hit
        if time.monotonic() >= exp:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: object) -> None:
        self._store[key] = (time.monotonic() + self.ttl_s, value)

    def clear(self) -> None:
        self._store.clear()
