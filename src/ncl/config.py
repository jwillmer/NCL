"""Configuration management using Pydantic settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Supabase Configuration
    supabase_url: str = Field(..., validation_alias="SUPABASE_URL")
    supabase_key: str = Field(..., validation_alias="SUPABASE_KEY")
    supabase_db_url: str = Field(..., validation_alias="SUPABASE_DB_URL")

    # OpenAI Configuration
    openai_api_key: str = Field(..., validation_alias="OPENAI_API_KEY")

    # Model Configuration
    embedding_model: str = Field(
        default="text-embedding-3-small", validation_alias="EMBEDDING_MODEL"
    )
    embedding_dimensions: int = Field(default=1536, validation_alias="EMBEDDING_DIMENSIONS")
    llm_model: str = Field(default="gpt-4o-mini", validation_alias="LLM_MODEL")

    # Chunking Configuration
    chunk_size_tokens: int = Field(default=512, validation_alias="CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens: int = Field(default=50, validation_alias="CHUNK_OVERLAP_TOKENS")

    # File Storage Paths
    attachments_dir: Path = Field(
        default=Path("./data/attachments"), validation_alias="ATTACHMENTS_DIR"
    )
    eml_source_dir: Path = Field(default=Path("./data/emails"), validation_alias="EML_SOURCE_DIR")

    # Processing Options
    batch_size: int = Field(default=10, validation_alias="BATCH_SIZE")
    max_concurrent_embeddings: int = Field(
        default=5, validation_alias="MAX_CONCURRENT_EMBEDDINGS"
    )
    enable_ocr: bool = Field(default=True, validation_alias="ENABLE_OCR")
    enable_picture_description: bool = Field(
        default=True, validation_alias="ENABLE_PICTURE_DESCRIPTION"
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
