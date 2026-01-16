"""Parser registry for plugin management."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Type

if TYPE_CHECKING:
    from .base import BaseParser

logger = logging.getLogger(__name__)


class ParserRegistry:
    """Registry for document parser plugins.

    Provides simple dict-based registration and lookup of parsers.
    """

    _parsers: Dict[str, "BaseParser"] = {}
    _mimetype_map: Dict[str, str] = {}  # mimetype -> parser_name
    _extension_map: Dict[str, str] = {}  # extension -> parser_name

    @classmethod
    def register(cls, parser_class: Type["BaseParser"]) -> None:
        """Register a parser class.

        Args:
            parser_class: The parser class to register.
        """
        parser = parser_class()
        cls._parsers[parser.name] = parser

        # Build lookup maps
        for mimetype in parser.supported_mimetypes:
            cls._mimetype_map[mimetype] = parser.name

        for ext in parser.supported_extensions:
            cls._extension_map[ext.lower()] = parser.name

        logger.debug(f"Registered parser: {parser.name}")

    @classmethod
    def get_parser(cls, name: str) -> Optional["BaseParser"]:
        """Get parser by name.

        Args:
            name: Parser name.

        Returns:
            Parser instance or None if not found.
        """
        return cls._parsers.get(name)

    @classmethod
    def get_parser_for_file(
        cls,
        file_path: Path,
        content_type: Optional[str] = None,
    ) -> Optional["BaseParser"]:
        """Get appropriate parser for a file.

        Args:
            file_path: Path to the file.
            content_type: Optional MIME type.

        Returns:
            Best matching parser or None.
        """
        # Check by MIME type first
        if content_type and content_type in cls._mimetype_map:
            parser_name = cls._mimetype_map[content_type]
            return cls._parsers.get(parser_name)

        # Check by extension
        ext = file_path.suffix.lower()
        if ext in cls._extension_map:
            parser_name = cls._extension_map[ext]
            return cls._parsers.get(parser_name)

        return None

    @classmethod
    def list_parsers(cls) -> Dict[str, "BaseParser"]:
        """Get all registered parsers."""
        return cls._parsers.copy()

    @classmethod
    def clear(cls) -> None:
        """Clear all registered parsers. Useful for testing."""
        cls._parsers.clear()
        cls._mimetype_map.clear()
        cls._extension_map.clear()
