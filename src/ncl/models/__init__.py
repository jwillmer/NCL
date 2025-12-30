"""Data models for NCL."""

from .document import (
    Document,
    DocumentType,
    ProcessingStatus,
    EmailMetadata,
    EmailMessage,
    AttachmentMetadata,
    ParsedEmail,
    ParsedAttachment,
)
from .chunk import Chunk, ChunkWithContext, SourceReference
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
    "SourceReference",
    "Vessel",
    "VesselSummary",
]
