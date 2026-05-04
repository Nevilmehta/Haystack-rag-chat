import hashlib
import time
from typing import Any

class InMemoryCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self.store: dict[str, dict[str, Any]] = {}

    def _make_key(self, text: str):
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, text: str):
        key = self._make_key(text)
        item = self.store.get(key)

        if not item:
            return None

        if time.time() - item["created_at"] > self.ttl_seconds:
            del self.store[key]
            return None

        return item["value"]

    def set(self, text: str, value: Any):
        key = self._make_key(text)

        self.store[key] = {
            "value": value,
            "created_at": time.time()
        }        

    def clear(self):
        self.store.clear()