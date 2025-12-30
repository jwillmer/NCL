"""Configuration management using Pydantic settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    Data Folder Structure:
        data/
        ├── source/           # User-provided data (read-only)
        │   ├── emails/       # EML files (can have subdirectories)
        │   └── ...           # Users can organize as they wish
        └── processed/        # NCL-generated data
            ├── attachments/  # Extracted email attachments
            └── extracted/    # Extracted ZIP contents
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Supabase Configuration
    supabase_url: str = Field(..., validation_alias="SUPABASE_URL")
    supabase_key: str = Field(..., validation_alias="SUPABASE_KEY")
    supabase_db_url: str = Field(..., validation_alias="SUPABASE_DB_URL")
    supabase_jwt_secret: Optional[str] = Field(
        default=None, validation_alias="SUPABASE_JWT_SECRET"
    )

    # OpenAI Configuration
    openai_api_key: str = Field(..., validation_alias="OPENAI_API_KEY")

    # Model Configuration
    embedding_model: str = Field(
        default="text-embedding-3-small", validation_alias="EMBEDDING_MODEL"
    )
    embedding_dimensions: int = Field(default=1536, validation_alias="EMBEDDING_DIMENSIONS")
    llm_model: str = Field(default="gpt-4o-mini", validation_alias="LLM_MODEL")

    # Contextual Chunking Configuration
    context_llm_model: str = Field(
        default="gpt-5-nano", validation_alias="CONTEXT_LLM_MODEL"
    )

    # Chunking Configuration
    chunk_size_tokens: int = Field(default=512, validation_alias="CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens: int = Field(default=50, validation_alias="CHUNK_OVERLAP_TOKENS")

    # Base data directory
    data_dir: Path = Field(
        default=Path("./data"), validation_alias="DATA_DIR"
    )

    # File Storage Paths - Source Data (user-provided, read-only)
    data_source_dir: Path = Field(
        default=Path("./data/source"), validation_alias="DATA_SOURCE_DIR"
    )

    # File Storage Paths - Processed Data (NCL-generated)
    data_processed_dir: Path = Field(
        default=Path("./data/processed"), validation_alias="DATA_PROCESSED_DIR"
    )

    # Derived paths (computed from base directories)
    @computed_field
    @property
    def eml_source_dir(self) -> Path:
        """Directory containing source EML files (supports subdirectories)."""
        return self.data_source_dir

    @computed_field
    @property
    def attachments_dir(self) -> Path:
        """Directory for extracted email attachments."""
        return self.data_processed_dir / "attachments"

    @computed_field
    @property
    def extracted_dir(self) -> Path:
        """Directory for extracted ZIP contents."""
        return self.data_processed_dir / "extracted"

    # Archive Configuration (for browsable content)
    archive_dir: Path = Field(
        default=Path("./data/archive"), validation_alias="ARCHIVE_DIR"
    )
    archive_base_url: str = Field(
        default="", validation_alias="ARCHIVE_BASE_URL"
    )

    # Ingest Versioning
    current_ingest_version: int = Field(
        default=1, validation_alias="CURRENT_INGEST_VERSION"
    )

    # Processing Options
    batch_size: int = Field(default=10, validation_alias="BATCH_SIZE")
    max_concurrent_embeddings: int = Field(
        default=5, validation_alias="MAX_CONCURRENT_EMBEDDINGS"
    )
    max_concurrent_files: int = Field(
        default=5, validation_alias="MAX_CONCURRENT_FILES"
    )
    enable_ocr: bool = Field(default=True, validation_alias="ENABLE_OCR")
    enable_picture_description: bool = Field(
        default=True, validation_alias="ENABLE_PICTURE_DESCRIPTION"
    )

    # Reranker Configuration
    rerank_enabled: bool = Field(default=True, validation_alias="RERANK_ENABLED")
    rerank_model: str = Field(
        default="cohere/rerank-english-v3.0", validation_alias="RERANK_MODEL"
    )
    rerank_top_n: int = Field(default=5, validation_alias="RERANK_TOP_N")
    cohere_api_key: Optional[str] = Field(default=None, validation_alias="COHERE_API_KEY")

    # LlamaParse Configuration (for legacy formats like .doc, .xls, .ppt)
    llama_cloud_api_key: Optional[str] = Field(
        default=None, validation_alias="LLAMA_CLOUD_API_KEY"
    )

    @computed_field
    @property
    def llamaparse_enabled(self) -> bool:
        """LlamaParse is auto-enabled when API key is set."""
        return self.llama_cloud_api_key is not None

    # ZIP Extraction Limits (DoS protection)
    zip_max_files: int = Field(default=100, validation_alias="ZIP_MAX_FILES")
    zip_max_depth: int = Field(default=3, validation_alias="ZIP_MAX_DEPTH")
    zip_max_total_size_mb: int = Field(default=500, validation_alias="ZIP_MAX_TOTAL_SIZE_MB")

    # Embedding batch size (OpenAI allows up to 2048)
    embedding_batch_size: int = Field(default=100, validation_alias="EMBEDDING_BATCH_SIZE")

    # Error truncation length for database storage
    error_message_max_length: int = Field(
        default=1000, validation_alias="ERROR_MESSAGE_MAX_LENGTH"
    )

    # Chunk content display truncation (for source references in responses)
    chunk_display_max_chars: int = Field(
        default=500, validation_alias="CHUNK_DISPLAY_MAX_CHARS"
    )

    # API Configuration
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173", validation_alias="CORS_ORIGINS"
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
