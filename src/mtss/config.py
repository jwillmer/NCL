"""Configuration management using Pydantic settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    Data Folder Structure:
        data/
        ├── source/           # User-provided data (read-only)
        │   ├── emails/       # EML files (can have subdirectories)
        │   └── ...           # Users can organize as they wish
        └── processed/        # MTSS-generated data
            ├── attachments/  # Extracted email attachments
            └── extracted/    # Extracted ZIP contents
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Supabase Configuration (required for import/query/maintenance commands)
    supabase_url: str | None = Field(default=None, validation_alias="SUPABASE_URL")
    supabase_key: str | None = Field(default=None, validation_alias="SUPABASE_KEY")
    supabase_db_url: str | None = Field(default=None, validation_alias="SUPABASE_DB_URL")
    supabase_jwt_secret: str | None = Field(
        default=None, validation_alias="SUPABASE_JWT_SECRET"
    )

    # OpenRouter Configuration
    openrouter_api_key: str = Field(..., validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", validation_alias="OPENROUTER_BASE_URL"
    )

    # Model Configuration
    embedding_model: str = Field(
        default="openrouter/openai/text-embedding-3-small", validation_alias="EMBEDDING_MODEL"
    )
    embedding_dimensions: int = Field(default=1536, validation_alias="EMBEDDING_DIMENSIONS")
    embedding_max_tokens: int = Field(default=8000, validation_alias="EMBEDDING_MAX_TOKENS")

    # Default LLM model (fallback when specific model not set)
    llm_model: str = Field(default="openrouter/openai/gpt-5-nano", validation_alias="LLM_MODEL")

    # Dedicated models per functionality (None = fallback to llm_model)
    context_llm_model: str | None = Field(
        default=None, validation_alias="CONTEXT_LLM_MODEL"
    )
    context_llm_fallback: str | None = Field(
        default=None, validation_alias="CONTEXT_LLM_FALLBACK"
    )
    email_cleaner_model: str | None = Field(
        default="openrouter/openai/gpt-5-nano", validation_alias="EMAIL_CLEANER_MODEL"
    )
    thread_digest_model: str | None = Field(
        default=None, validation_alias="THREAD_DIGEST_MODEL"
    )
    image_llm_model: str | None = Field(
        default="openrouter/openai/gpt-5-nano", validation_alias="IMAGE_LLM_MODEL"
    )
    rag_llm_model: str | None = Field(
        default=None, validation_alias="RAG_LLM_MODEL"
    )

    def get_model(self, specific_model: str | None) -> str:
        """Return specific model if set, otherwise fallback to llm_model."""
        return specific_model or self.llm_model

    # Chunking Configuration
    chunk_size_tokens: int = Field(default=1024, validation_alias="CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens: int = Field(default=100, validation_alias="CHUNK_OVERLAP_TOKENS")

    # Base data directory
    data_dir: Path = Field(
        default=Path("./data"), validation_alias="DATA_DIR"
    )

    # File Storage Paths - Source Data (user-provided, read-only)
    data_source_dir: Path = Field(
        default=Path("./data/source"), validation_alias="DATA_SOURCE_DIR"
    )

    # File Storage Paths - Processed Data (MTSS-generated)
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

    # Archive Configuration (Supabase Storage)
    archive_bucket: str = Field(
        default="archive", validation_alias="ARCHIVE_BUCKET"
    )
    archive_base_url: str = Field(
        default="", validation_alias="ARCHIVE_BASE_URL"
    )

    # Ingest Versioning
    # Version history:
    #   1 - Initial ingest
    #   2 - Added line numbers
    #   3 - Added context summaries
    #   4 - Added topic extraction
    #   5 - Sanitized archive keys (underscores), stripped LlamaParse image refs
    current_ingest_version: int = Field(
        default=5, validation_alias="CURRENT_INGEST_VERSION"
    )

    # Processing Options
    max_concurrent_files: int = Field(
        default=8, validation_alias="MAX_CONCURRENT_FILES"
    )
    max_concurrent_llamaparse: int = Field(
        default=5, validation_alias="MAX_CONCURRENT_LLAMAPARSE"
    )
    # Reranker Configuration
    rerank_enabled: bool = Field(default=True, validation_alias="RERANK_ENABLED")
    rerank_model: str = Field(
        default="cohere/rerank-v3.5", validation_alias="RERANK_MODEL"
    )
    rerank_top_n: int = Field(default=8, validation_alias="RERANK_TOP_N")
    rerank_score_floor: float = Field(default=0.2, validation_alias="RERANK_SCORE_FLOOR")

    # Retrieval Configuration
    retrieval_top_k: int = Field(default=40, validation_alias="RETRIEVAL_TOP_K")

    # Hybrid Search (vector + BM25)
    hybrid_search_enabled: bool = Field(
        default=True, validation_alias="HYBRID_SEARCH_ENABLED"
    )

    # Topic Loosening (query-time)
    topic_match_threshold_loose: float = Field(
        default=0.55, ge=0.0, le=1.0, validation_alias="TOPIC_MATCH_THRESHOLD_LOOSE"
    )
    topic_loosening_min_chunks: int = Field(
        default=3, ge=1, le=50, validation_alias="TOPIC_LOOSENING_MIN_CHUNKS"
    )

    # LlamaParse Configuration (for legacy formats like .doc, .xls, .ppt)
    llama_cloud_api_key: str | None = Field(
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

    # Embedding batch size
    embedding_batch_size: int = Field(default=100, validation_alias="EMBEDDING_BATCH_SIZE")

    # Error truncation length for database storage
    error_message_max_length: int = Field(
        default=1000, validation_alias="ERROR_MESSAGE_MAX_LENGTH"
    )

    # Per-file processing timeout (seconds) - prevents stalled API calls from hanging
    per_file_timeout_seconds: int = Field(
        default=300, validation_alias="PER_FILE_TIMEOUT_SECONDS"
    )

    # Failure Report Configuration
    failure_reports_dir: Path = Field(
        default=Path("./data/reports"), validation_alias="FAILURE_REPORTS_DIR"
    )
    failure_reports_keep_count: int = Field(
        default=30, validation_alias="FAILURE_REPORTS_KEEP_COUNT"
    )

    @field_validator(
        "data_dir",
        "data_source_dir",
        "data_processed_dir",
        "failure_reports_dir",
        mode="after",
    )
    @classmethod
    def validate_no_path_traversal(cls, v: Path) -> Path:
        """Validate that paths don't contain path traversal sequences.

        Prevents security vulnerabilities from malicious path configurations.

        Args:
            v: Path to validate.

        Returns:
            The validated path.

        Raises:
            ValueError: If path contains '..' traversal sequence.
        """
        if ".." in str(v):
            raise ValueError(f"Path traversal detected in path: {v}")
        return v

    # API Configuration
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173", validation_alias="CORS_ORIGINS"
    )

    # Langfuse Observability (optional)
    langfuse_enabled: bool = Field(default=False, validation_alias="LANGFUSE_ENABLED")
    langfuse_public_key: str | None = Field(
        default=None, validation_alias="LANGFUSE_PUBLIC_KEY"
    )
    langfuse_secret_key: str | None = Field(
        default=None, validation_alias="LANGFUSE_SECRET_KEY"
    )
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com", validation_alias="LANGFUSE_BASE_URL"
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
