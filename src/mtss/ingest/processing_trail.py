"""In-memory per-email processing trail.

Records which model/parser produced which artifact (parse, context,
embed, vision, summary, decider, topics) at which time. Current-state
only — re-running a step overwrites the previous entry. At the end of
``process_email`` the trail is merged into the email's
``archive/<folder_id>/metadata.json`` under a ``processing`` key.

Not imported by the Supabase side — this data lives only in local
ingest output.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

_EMAIL_KEY = "email"
_ATTACHMENTS_KEY = "attachments"


# Canonical step names. Use these constants at every stamp site so a typo
# becomes an ImportError instead of a silently-created new key.
STEP_PARSE = "parse"
STEP_CONTEXT = "context"
STEP_CONTENT_CLEANUP = "content_cleanup"
STEP_TOPICS = "topics"
STEP_THREAD_DIGEST = "thread_digest"
STEP_DECIDER = "decider"
STEP_SUMMARY = "summary"
STEP_VISION = "vision"
STEP_EMBED = "embed"


def _utcnow_iso() -> str:
    """UTC timestamp with second precision and explicit ``Z`` suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _clean_details(details: Dict[str, Any]) -> Dict[str, Any]:
    """Drop ``None`` values from a details dict so entries stay compact."""
    return {k: v for k, v in details.items() if v is not None}


class ProcessingTrail:
    """Accumulator for per-step model + timestamp records.

    Two entry-point axes: email-level (single slot) and attachment-level
    (one slot per filename). Each slot holds the latest stamp per step.
    """

    def __init__(self) -> None:
        self._email: Dict[str, Dict[str, Any]] = {}
        self._attachments: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._doc_id_to_filename: Dict[str, str] = {}

    # ── registration ────────────────────────────────────────────────

    def register_email_document(self, document_id: str) -> None:
        """Link the email-document's UUID to the email slot.

        Enables ``stamp_by_document_id`` to route email-document stamps
        (e.g. post-embed chunk counts) to the email slot.
        """
        self._doc_id_to_filename[str(document_id)] = _EMAIL_KEY

    def register_attachment_document(
        self, document_id: str, filename: str
    ) -> None:
        """Link an attachment-document's UUID to its filename slot."""
        self._doc_id_to_filename[str(document_id)] = filename

    # ── stamping ────────────────────────────────────────────────────

    def stamp_email(
        self,
        step: str,
        *,
        model: Optional[str] = None,
        **details: Any,
    ) -> None:
        """Overwrite the email-level record for ``step``."""
        self._email[step] = self._build_entry(model, details)

    def stamp_attachment(
        self,
        filename: str,
        step: str,
        *,
        model: Optional[str] = None,
        **details: Any,
    ) -> None:
        """Overwrite the per-attachment record for ``(filename, step)``."""
        slot = self._attachments.setdefault(filename, {})
        slot[step] = self._build_entry(model, details)

    def stamp_by_document_id(
        self,
        document_id: str,
        step: str,
        *,
        model: Optional[str] = None,
        **details: Any,
    ) -> None:
        """Route a stamp to the email or attachment slot by UUID.

        No-op when the document_id has not been registered — lets callers
        fan out over chunk.document_id without pre-filtering.
        """
        key = self._doc_id_to_filename.get(str(document_id))
        if key is None:
            return
        if key == _EMAIL_KEY:
            self.stamp_email(step, model=model, **details)
        else:
            self.stamp_attachment(key, step, model=model, **details)

    # ── serialization ───────────────────────────────────────────────

    def to_json(self) -> Dict[str, Any]:
        """Serialize to the shape written into ``metadata.json``.

        Returns an empty dict ``{email: {}, attachments: {}}`` rather than
        ``None`` so the key is always present once the trail was created.
        """
        return {
            _EMAIL_KEY: dict(self._email),
            _ATTACHMENTS_KEY: {fn: dict(slot) for fn, slot in self._attachments.items()},
        }

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _build_entry(model: Optional[str], details: Dict[str, Any]) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"ran_at": _utcnow_iso()}
        if model is not None:
            entry["model"] = model
        cleaned = _clean_details(details)
        if cleaned:
            entry.update(cleaned)
        return entry
