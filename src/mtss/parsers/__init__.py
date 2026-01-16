"""Document parsers for mtss."""

from .attachment_processor import AttachmentProcessor
from .base import BaseParser
from .chunker import ContextGenerator, DocumentChunker
from .eml_parser import EMLParser
from .llamaparse_parser import LlamaParseParser
from .preprocessor import DocumentPreprocessor, PreprocessResult
from .registry import ParserRegistry
from .text_parser import TextParser

# Register built-in parsers
ParserRegistry.register(LlamaParseParser)
ParserRegistry.register(TextParser)

__all__ = [
    "AttachmentProcessor",
    "BaseParser",
    "ContextGenerator",
    "DocumentChunker",
    "DocumentPreprocessor",
    "EMLParser",
    "LlamaParseParser",
    "ParserRegistry",
    "PreprocessResult",
    "TextParser",
]
