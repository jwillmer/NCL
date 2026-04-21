"""Unit tests for the native-async archive upload helper.

``storage.async_upload`` is the replacement for the sync ``ArchiveStorage.
upload_file`` + ``asyncio.to_thread`` pattern that triggered
``WSAEWOULDBLOCK`` on Windows under concurrent large-body POSTs. These
tests cover the retry / success / failure / semaphore contract without
spinning up a real ``httpx.AsyncClient`` — the helper splits the
per-file retry loop into ``_upload_one`` exactly so it can be exercised
against a mock bucket.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mtss.storage import async_upload
from mtss.storage.async_upload import UploadItem, _upload_one, upload_many


def _mk_item(tmp_path: Path, name: str = "file.pdf") -> UploadItem:
    p = tmp_path / name
    p.write_bytes(b"payload")
    return UploadItem(local_path=p, remote_key=name, content_type="application/pdf")


class TestUploadOne:
    """Direct exercise of the retry loop."""

    async def test_success_on_first_attempt(self, tmp_path, monkeypatch):
        bucket = MagicMock()
        bucket.upload = AsyncMock(return_value=None)
        monkeypatch.setattr(async_upload.asyncio, "sleep", AsyncMock())

        item = _mk_item(tmp_path)
        outcome: list[tuple[UploadItem, bool]] = []

        ok = await _upload_one(
            bucket,
            item,
            max_attempts=3,
            backoff_base=0.01,
            jitter=0.0,
            on_retry=None,
            on_progress=lambda it, s: outcome.append((it, s)),
        )

        assert ok is True
        assert bucket.upload.await_count == 1
        # on_progress invoked exactly once with success.
        assert outcome == [(item, True)]
        # No retry → no sleeps.
        assert async_upload.asyncio.sleep.await_count == 0

    async def test_retries_then_succeeds(self, tmp_path, monkeypatch):
        bucket = MagicMock()
        bucket.upload = AsyncMock(
            side_effect=[RuntimeError("transient"), RuntimeError("again"), None]
        )
        fake_sleep = AsyncMock()
        monkeypatch.setattr(async_upload.asyncio, "sleep", fake_sleep)
        # Pin jitter so the delay assertions are deterministic.
        monkeypatch.setattr(async_upload.random, "uniform", lambda a, b: 1.0)

        item = _mk_item(tmp_path)
        retries: list[tuple[int, BaseException, float, str]] = []

        ok = await _upload_one(
            bucket,
            item,
            max_attempts=5,
            backoff_base=0.5,
            jitter=0.5,
            on_retry=lambda attempt, exc, delay, key: retries.append(
                (attempt, exc, delay, key)
            ),
            on_progress=None,
        )

        assert ok is True
        assert bucket.upload.await_count == 3
        # Two retries → two sleeps at 0.5*2^0 and 0.5*2^1.
        delays = [c.args[0] for c in fake_sleep.await_args_list]
        assert delays == [pytest.approx(0.5), pytest.approx(1.0)]
        assert [r[0] for r in retries] == [1, 2]
        assert all(r[3] == item.remote_key for r in retries)

    async def test_returns_false_after_budget_exhausted(
        self, tmp_path, monkeypatch
    ):
        bucket = MagicMock()
        bucket.upload = AsyncMock(side_effect=RuntimeError("always fails"))
        monkeypatch.setattr(async_upload.asyncio, "sleep", AsyncMock())

        item = _mk_item(tmp_path)
        outcome: list[tuple[UploadItem, bool]] = []

        ok = await _upload_one(
            bucket,
            item,
            max_attempts=3,
            backoff_base=0.01,
            jitter=0.0,
            on_retry=None,
            on_progress=lambda it, s: outcome.append((it, s)),
        )

        assert ok is False
        assert bucket.upload.await_count == 3
        assert outcome == [(item, False)]

    async def test_passes_path_for_streaming_upload(self, tmp_path):
        """Regression: the helper must pass the ``Path`` directly, not
        pre-read bytes. storage3 streams from the handle only when given
        a ``Path``/``str``/file-object; passing bytes defeats the whole
        point of the async migration."""
        bucket = MagicMock()
        bucket.upload = AsyncMock(return_value=None)

        item = _mk_item(tmp_path)
        await _upload_one(
            bucket,
            item,
            max_attempts=1,
            backoff_base=0.01,
            jitter=0.0,
            on_retry=None,
            on_progress=None,
        )

        call = bucket.upload.await_args
        # (path, file, file_options)
        assert call.args[0] == item.remote_key
        assert call.args[1] is item.local_path
        assert isinstance(call.args[1], Path)
        assert call.args[2] == {
            "content-type": item.content_type,
            "upsert": "true",
        }


class TestUploadManySemaphore:
    """The semaphore must actually bound in-flight uploads."""

    async def test_semaphore_caps_in_flight_uploads(self, tmp_path, monkeypatch):
        """With ``max_concurrency=3``, at most 3 uploads run at once
        even with 10 queued."""
        # Keep the real ``_upload_one`` — we only want to wrap the
        # per-bucket call. Replace ``AsyncStorageClient`` construction
        # with a no-op and drive a mock bucket.
        items = [_mk_item(tmp_path, f"f{i}.bin") for i in range(10)]

        in_flight = 0
        peak = 0
        lock = asyncio.Lock()

        async def fake_upload(path, file, opts):
            nonlocal in_flight, peak
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            # Yield so the semaphore can let other coroutines into the
            # critical section up to the configured cap.
            await asyncio.sleep(0.01)
            async with lock:
                in_flight -= 1

        bucket = MagicMock()
        bucket.upload = fake_upload

        class FakeClient:
            def __init__(self, *a, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

        class FakeStorageClient:
            def __init__(self, *a, **kw):
                pass

            def from_(self, name):
                return bucket

        monkeypatch.setattr(async_upload.httpx, "AsyncClient", FakeClient)
        monkeypatch.setattr(async_upload, "AsyncStorageClient", FakeStorageClient)

        # Minimal settings stub — the helper only reads supabase_url + key.
        fake_settings = MagicMock()
        fake_settings.supabase_url = "https://example.supabase.co"
        fake_settings.supabase_key = "fake-key"
        monkeypatch.setattr(async_upload, "get_settings", lambda: fake_settings)

        results = await upload_many(
            items,
            bucket_name="test-bucket",
            max_concurrency=3,
            max_attempts=1,
            backoff_base=0.01,
            jitter=0.0,
        )

        assert results == [True] * 10
        assert peak <= 3, f"semaphore breach: saw {peak} concurrent"
        assert peak == 3, f"expected to saturate cap of 3, saw {peak}"


class TestUploadManyArgValidation:
    async def test_empty_items_returns_empty_list_without_client_setup(
        self, monkeypatch
    ):
        # If the helper built a client for an empty batch it would waste a
        # TLS handshake on nothing. Assert short-circuit.
        constructed = []

        class FakeClient:
            def __init__(self, *a, **kw):
                constructed.append(kw)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

        monkeypatch.setattr(async_upload.httpx, "AsyncClient", FakeClient)

        result = await upload_many(
            [], bucket_name="x", max_concurrency=4, max_attempts=3,
            backoff_base=0.5, jitter=0.0,
        )

        assert result == []
        assert constructed == []

    async def test_rejects_zero_concurrency(self, tmp_path):
        with pytest.raises(ValueError, match="max_concurrency"):
            await upload_many(
                [_mk_item(tmp_path)],
                bucket_name="x",
                max_concurrency=0,
                max_attempts=1,
                backoff_base=0.5,
                jitter=0.0,
            )

    async def test_rejects_zero_attempts(self, tmp_path):
        with pytest.raises(ValueError, match="max_attempts"):
            await upload_many(
                [_mk_item(tmp_path)],
                bucket_name="x",
                max_concurrency=1,
                max_attempts=0,
                backoff_base=0.5,
                jitter=0.0,
            )
