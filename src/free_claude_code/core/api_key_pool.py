"""Thread-safe round-robin API key pool for 429 failover."""

import logging
import threading

logger = logging.getLogger(__name__)


class ApiKeyPool:
    """Thread-safe round-robin pool of API keys with 429-aware rotation."""

    def __init__(self, raw: str) -> None:
        """Parses comma-separated keys, strips whitespace, and removes empty values."""
        if not raw:
            raise ValueError("API key credential cannot be empty")

        # Split by comma and strip whitespace
        self._keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not self._keys:
            raise ValueError("API key pool must contain at least one non-empty key")

        self._lock = threading.Lock()
        # Round-robin index: start at 0
        self._index = 0
        # Tracks which indices have been tried in the current request cycle
        self._tried: set[int] = {0}

    @property
    def size(self) -> int:
        """Total number of keys in the pool."""
        return len(self._keys)

    def current_key(self) -> str:
        """Returns the current API key without rotating the index."""
        with self._lock:
            return self._keys[self._index]

    def rotate(self) -> str | None:
        """Advances the index to the next key.

        Returns:
            The next key, or None if all keys have been tried in the current cycle.
        """
        with self._lock:
            # Advance index in round-robin fashion
            next_idx = (self._index + 1) % len(self._keys)

            # If we've already tried this index in this cycle, we've exhausted all keys
            if next_idx in self._tried:
                return None

            self._index = next_idx
            self._tried.add(next_idx)
            return self._keys[next_idx]

    def reset(self) -> None:
        """Resets the tried set for a new request cycle, starting with the currently active index."""
        with self._lock:
            self._tried = {self._index}
