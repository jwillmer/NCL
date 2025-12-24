"""Document parsers for NCL."""

from .attachment_processor import AttachmentProcessor
from .base import BaseParser
from .chunker import DocumentChunker
from .eml_parser import EMLParser
from .llamaparse_parser import LlamaParseParser
from .preprocessor import DocumentPreprocessor, PreprocessResult
from .registry import ParserRegistry

# Register built-in parsers
ParserRegistry.register(LlamaParseParser)

__all__ = [
    "AttachmentProcessor",
    "BaseParser",
    "DocumentChunker",
    "DocumentPreprocessor",
    "EMLParser",
    "LlamaParseParser",
    "ParserRegistry",
    "PreprocessResult",
]
