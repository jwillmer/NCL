"""Document and email-related data models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Types of documents in the hierarchy."""

    EMAIL = "email"
    ATTACHMENT_PDF = "attachment_pdf"
    ATTACHMENT_IMAGE = "attachment_image"
    ATTACHMENT_DOCX = "attachment_docx"
    ATTACHMENT_PPTX = "attachment_pptx"
    ATTACHMENT_XLSX = "attachment_xlsx"
    # Legacy formats (via LlamaParse)
    ATTACHMENT_DOC = "attachment_doc"
    ATTACHMENT_XLS = "attachment_xls"
    ATTACHMENT_PPT = "attachment_ppt"
    ATTACHMENT_CSV = "attachment_csv"
    ATTACHMENT_RTF = "attachment_rtf"
    ATTACHMENT_OTHER = "attachment_other"


class ProcessingStatus(str, Enum):
    """Processing status for documents."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EmailMessage(BaseModel):
    """A single message within an email conversation."""

    from_address: str
    to_addresses: List[str] = Field(default_factory=list)
    cc_addresses: List[str] = Field(default_factory=list)
    date: Optional[datetime] = None
    content: str = ""


class EmailMetadata(BaseModel):
    """Email conversation metadata extracted from EML files.

    Designed for email threads/conversations rather than single emails.
    Tracks all participants and the conversation initiator.
    """

    # Conversation-level metadata
    subject: Optional[str] = None
    participants: List[str] = Field(default_factory=list)  # All unique email addresses
    initiator: Optional[str] = None  # First sender in the thread
    date_start: Optional[datetime] = None  # Earliest message date
    date_end: Optional[datetime] = None  # Latest message date
    message_count: int = 1  # Number of messages in conversation

    # For reference/threading
    message_ids: List[str] = Field(default_factory=list)
    in_reply_to: Optional[str] = None
    references: List[str] = Field(default_factory=list)


class AttachmentMetadata(BaseModel):
    """Attachment-specific metadata."""

    content_type: str
    size_bytes: int
    original_filename: str


class Document(BaseModel):
    """Represents a document in the hierarchy (email or attachment)."""

    id: UUID = Field(default_factory=uuid4)
    parent_id: Optional[UUID] = None
    root_id: Optional[UUID] = None
    depth: int = 0
    path: List[str] = Field(default_factory=list)

    document_type: DocumentType
    file_path: str
    file_name: str
    file_hash: Optional[str] = None

    # Type-specific metadata (one will be set based on document_type)
    email_metadata: Optional[EmailMetadata] = None
    attachment_metadata: Optional[AttachmentMetadata] = None

    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    processed_at: Optional[datetime] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ParsedAttachment(BaseModel):
    """Extracted attachment from email."""

    filename: str
    content_type: str
    size_bytes: int
    saved_path: str  # Path where attachment was saved to disk


class ParsedEmail(BaseModel):
    """Result of parsing an EML file (conversation).

    Handles both single emails and threaded conversations.
    """

    metadata: EmailMetadata
    messages: List[EmailMessage] = Field(default_factory=list)  # Individual messages in thread
    full_text: str = ""  # Combined conversation text for chunking
    attachments: List[ParsedAttachment] = Field(default_factory=list)
