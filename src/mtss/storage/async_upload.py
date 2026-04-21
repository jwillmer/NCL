"""Native async archive upload path.

The sync ``ArchiveStorage.upload_file`` route runs on threads dispatched
via ``asyncio.to_thread`` and talks to Supabase Storage through a
``httpx.Client`` whose internal socket is a Python *timeout* socket —
which is internally non-blocking and drives I/O through a ``select()``
loop. On Windows this surfaces ``WSAEWOULDBLOCK`` (WinError 10035) back
to user code whenever the per-socket send buffer fills faster than the
wire drains, which happens routinely under concurrent large-body POSTs
(10–50 MB PDFs × 8 workers).

``httpx.AsyncClient`` on Windows runs on asyncio's Proactor loop → IOCP,
which absorbs that exact backpressure at the OS layer instead of
surfacing it as a socket error. Combined with passing a ``Path`` (so
storage3 streams the file handle into httpx's multipart encoder instead
of materialising the full body) this eliminates the WSAEWOULDBLOCK
class, not just retries around it.

The sync ``ArchiveStorage`` remains the canonical client for the small
operations (list/remove/delete_folder). This module is only the
high-concurrency upload fan-out.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import httpx
from storage3 import AsyncStorageClient
from storage3.utils import StorageException

from ..config import get_settings

logger = logging.getLogger(__name__)


# Supabase Storage validates object keys against this character set
# (see storage-api/src/http/routes/object/util.ts). A key containing
# anything outside this set is rejected with ``{"error": "InvalidKey",
# "statusCode": 400}`` and no amount of retrying will help. Keys must
# be sanitized before the upload is dispatched.
_SAFE_KEY_RE = re.compile(r"[\w/!\-.*'() &$@=;:+,?]")


def sanitize_storage_key(key: str) -> str:
    """Replace characters Supabase Storage's key validator rejects with ``_``.

    The real-world offender is ``%`` appearing in attachment filenames
    like ``STHAMEX_SV-HT_2%_F-10_9273_SDS.pdf``. The sanitizer is
    deterministic so repeated runs collapse to the same remote key.
    """
    if all(_SAFE_KEY_RE.fullmatch(c) for c in key):
        return key
    return "".join(c if _SAFE_KEY_RE.fullmatch(c) else "_" for c in key)


def _is_terminal_upload_error(exc: BaseException) -> bool:
    """True for errors no retry can recover: 4xx client errors from the
    storage API. Retrying them wastes the full backoff budget (~7.5s on
    the default 5-attempt ladder) per bad file."""
    if not isinstance(exc, StorageException):
        return False
    payload = exc.args[0] if exc.args else None
    if not isinstance(payload, dict):
        return False
    status = payload.get("statusCode")
    # statusCode arrives as str on some storage3 paths, int on others.
    try:
        status_int = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_int = None
    return status_int is not None and 400 <= status_int < 500


@dataclass(frozen=True)
class UploadItem:
    """One file to upload: local path + remote key + content type."""

    local_path: Path
    remote_key: str
    content_type: str


# HTTP/1.1 over multiple TCP connections (one per active upload, up to
# ``max_keepalive_connections``) is deliberately preferred over HTTP/2.
# HTTP/2 multiplexes every upload onto a single TCP socket; under 8
# concurrent large-body POSTs that concentrates all the send-buffer
# pressure on one socket and reintroduces the exact problem we're trying
# to escape. With HTTP/1.1 + a tuned keep-alive pool, each upload gets
# its own socket and kernel-level backpressure works per-connection.
_DEFAULT_HTTP2 = False

# Per-upload HTTP timeouts. ``write`` is generous because a 50 MB PDF
# on a slow uplink can genuinely take minutes; ``read`` covers the
# gateway's acknowledgement latency after the body is on the wire;
# ``connect`` and ``pool`` stay short so a stuck DNS lookup or an
# exhausted connection pool fails fast instead of masquerading as a
# slow upload.
_DEFAULT_CONNECT_TIMEOUT = 10.0
_DEFAULT_READ_TIMEOUT = 120.0
_DEFAULT_WRITE_TIMEOUT = 300.0
_DEFAULT_POOL_TIMEOUT = 10.0


async def upload_many(
    items: Sequence[UploadItem],
    *,
    bucket_name: str,
    max_concurrency: int = 8,
    max_attempts: int = 5,
    backoff_base: float = 0.5,
    jitter: float = 0.5,
    on_retry: Callable[[int, BaseException, float, str], None] | None = None,
    on_progress: Callable[[UploadItem, bool], None] | None = None,
) -> list[bool]:
    """Upload every item in ``items`` concurrently via one async client.

    Returns a list aligned with ``items`` — ``True`` on success, ``False``
    after all attempts exhausted or the upload raised a non-retriable
    error. Per-file failures are caught so one bad file never kills the
    batch. The caller is expected to tally the ``False`` entries into
    whatever summary they're building.

    Args:
        items: Files to upload.
        bucket_name: Supabase Storage bucket (e.g. ``"archive"``).
        max_concurrency: Maximum in-flight uploads. Bounded both by the
            semaphore here and by ``httpx.Limits`` on the shared client.
        max_attempts: Per-file retry budget. ``1`` disables retry.
        backoff_base: Base delay in seconds. Delay for attempt ``i``
            (0-indexed) is ``backoff_base * 2**i``.
        jitter: Fractional jitter in ``[0, 1]``. ``0.5`` spreads each
            retry uniformly over ``[0.5*d, 1.5*d]`` to decouple
            concurrent retry waves.
        on_retry: Optional callback invoked as
            ``on_retry(attempt, exc, delay_seconds, remote_key)`` between
            attempts. Useful for logging without coupling to this module.
        on_progress: Optional callback invoked once per item after its
            final outcome, as ``on_progress(item, success)``. Wire this
            to a progress bar or counter in the caller.
    """
    if not items:
        return []
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    settings = get_settings()
    # storage3 tolerates with or without trailing slash, but matches
    # supabase-py's own wiring when it has the /storage/v1 suffix.
    storage_url = f"{settings.supabase_url.rstrip('/')}/storage/v1"
    headers = {
        "apikey": settings.supabase_key,
        "Authorization": f"Bearer {settings.supabase_key}",
    }

    limits = httpx.Limits(
        max_connections=max_concurrency * 2,
        max_keepalive_connections=max_concurrency,
    )
    timeout = httpx.Timeout(
        connect=_DEFAULT_CONNECT_TIMEOUT,
        read=_DEFAULT_READ_TIMEOUT,
        write=_DEFAULT_WRITE_TIMEOUT,
        pool=_DEFAULT_POOL_TIMEOUT,
    )

    async with httpx.AsyncClient(
        headers=headers,
        limits=limits,
        timeout=timeout,
        follow_redirects=True,
        http2=_DEFAULT_HTTP2,
    ) as http_client:
        storage = AsyncStorageClient(
            url=storage_url, headers=headers, http_client=http_client
        )
        bucket = storage.from_(bucket_name)
        sem = asyncio.Semaphore(max_concurrency)

        async def _one(item: UploadItem) -> bool:
            async with sem:
                return await _upload_one(
                    bucket,
                    item,
                    max_attempts=max_attempts,
                    backoff_base=backoff_base,
                    jitter=jitter,
                    on_retry=on_retry,
                    on_progress=on_progress,
                )

        return await asyncio.gather(*(_one(it) for it in items))


async def _upload_one(
    bucket,
    item: UploadItem,
    *,
    max_attempts: int,
    backoff_base: float,
    jitter: float,
    on_retry: Callable[[int, BaseException, float, str], None] | None,
    on_progress: Callable[[UploadItem, bool], None] | None,
) -> bool:
    """Upload one item with exponential-backoff retry.

    Split out of ``upload_many`` so tests can drive the retry loop
    against a mock bucket without spinning up an ``AsyncClient``.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            # Pass the Path directly so storage3 streams via
            # ``open(path, "rb")`` into httpx's multipart encoder
            # rather than preloading the whole file into memory.
            await bucket.upload(
                item.remote_key,
                item.local_path,
                {"content-type": item.content_type, "upsert": "true"},
            )
            if on_progress is not None:
                on_progress(item, True)
            return True
        except Exception as exc:
            last_exc = exc
            if _is_terminal_upload_error(exc):
                # 4xx from the storage API — the request is structurally
                # wrong (e.g. InvalidKey). Retrying will re-fail identically.
                break
            if attempt == max_attempts:
                break
            delay = backoff_base * (2 ** (attempt - 1))
            if jitter > 0:
                delay *= random.uniform(1 - jitter, 1 + jitter)
            if on_retry is not None:
                on_retry(attempt, exc, delay, item.remote_key)
            await asyncio.sleep(delay)
    logger.warning(
        "Failed to upload %s after %d attempts: %s",
        item.remote_key,
        max_attempts,
        last_exc,
    )
    if on_progress is not None:
        on_progress(item, False)
    return False
