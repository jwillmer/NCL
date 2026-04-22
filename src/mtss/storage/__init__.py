"""Storage layer for mtss.

Only API-runtime-safe modules are re-exported here. Ingest-side storage
(``SqliteStorageClient``, ``ProgressTracker``, ``LocalBucketStorage``,
``FailureReport*``, ``UnsupportedFileLogger``) depend on numpy/sklearn/etc.
which live under the ``ingest`` extras and are absent in the API image —
importing them eagerly from this package would break API startup.

Ingest-side code imports them directly via their full module path
(``mtss.storage.sqlite_client`` etc.), so nothing breaks there either.
"""

from .archive_storage import ArchiveStorage, ArchiveStorageError
from .supabase_client import SupabaseClient

__all__ = [
    "ArchiveStorage",
    "ArchiveStorageError",
    "SupabaseClient",
]
