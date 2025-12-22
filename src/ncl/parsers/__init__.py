"""Document parsers for NCL."""

from .eml_parser import EMLParser
from .attachment_processor import AttachmentProcessor

__all__ = ["EMLParser", "AttachmentProcessor"]
