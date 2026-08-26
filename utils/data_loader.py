"""
utils/data_loader.py

Thread-safe JSON loader with in-memory caching.
Prevents repeated disk reads across the session.
"""

import json
import threading
from typing import Any, Dict

_CACHE: Dict[str, Any] = {}
_LOCK = threading.Lock()


def load_json(path: str) -> Any:
    """
    Load a JSON file and cache it in memory.

    First call reads from disk; subsequent calls return the cached copy.
    """
    with _LOCK:
        if path not in _CACHE:
            with open(path, "r", encoding="utf-8") as fh:
                _CACHE[path] = json.load(fh)
        return _CACHE[path]


def clear_cache() -> None:
    """Flush the in-memory cache (useful in tests or after data reload)."""
    with _LOCK:
        _CACHE.clear()
