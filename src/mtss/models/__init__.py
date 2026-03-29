"""Data models for mtss."""

from .chunk import Chunk, ChunkWithContext, RetrievalResult
from .document import (
    AttachmentMetadata,
    Document,
    DocumentType,
    EmailMessage,
    EmailMetadata,
    ParsedAttachment,
    ParsedEmail,
    ProcessingStatus,
)
from .vessel import Vessel, VesselSummary

__all__ = [
    "Document",
    "DocumentType",
    "ProcessingStatus",
    "EmailMetadata",
    "EmailMessage",
    "AttachmentMetadata",
    "ParsedEmail",
    "ParsedAttachment",
    "Chunk",
    "ChunkWithContext",
    "RetrievalResult",
    "Vessel",
    "VesselSummary",
]
