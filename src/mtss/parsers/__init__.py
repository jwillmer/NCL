"""Document parsers for mtss."""

from .attachment_processor import AttachmentProcessor
from .base import BaseParser
from .chunker import ContextGenerator, DocumentChunker
from .eml_parser import EMLParser
from .llamaparse_parser import LlamaParseParser
from .msg_parser import MsgParser
from .preprocessor import DocumentPreprocessor, PreprocessResult
from .registry import ParserRegistry
from .rfc822_parser import RFC822Parser
from .text_parser import TextParser

# Register built-in parsers
ParserRegistry.register(LlamaParseParser)
ParserRegistry.register(TextParser)
ParserRegistry.register(RFC822Parser)
ParserRegistry.register(MsgParser)

__all__ = [
    "AttachmentProcessor",
    "BaseParser",
    "ContextGenerator",
    "DocumentChunker",
    "DocumentPreprocessor",
    "EMLParser",
    "LlamaParseParser",
    "MsgParser",
    "ParserRegistry",
    "PreprocessResult",
    "RFC822Parser",
    "TextParser",
]
