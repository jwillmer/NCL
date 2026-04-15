"""Tracks API call counts by service type during ingest."""

from __future__ import annotations

import threading


class ServiceCounter:
    """Thread-safe counter for API calls by service type."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def add(self, service: str, count: int = 1):
        with self._lock:
            self._counts[service] = self._counts.get(service, 0) + count

    def reset(self):
        with self._lock:
            self._counts.clear()

    @property
    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def to_dict(self) -> dict:
        with self._lock:
            return {"service_calls": dict(self._counts)}
