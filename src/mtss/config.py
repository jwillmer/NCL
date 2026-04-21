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
    #   6 - Two-tier PDF parser (PyMuPDF -> Gemini) + per-doc embedding_mode
    current_ingest_version: int = Field(
        default=6, validation_alias="CURRENT_INGEST_VERSION"
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

    # Topic dedup (ingest-time). Tighter than the earlier 0.85 default to
    # reduce long-tail fragmentation — raise if you start seeing false merges.
    topic_dedup_threshold: float = Field(
        default=0.80, ge=0.0, le=1.0, validation_alias="TOPIC_DEDUP_THRESHOLD"
    )
    # End-of-ingest auto-merge pairwise threshold (pairs above this merge).
    topic_auto_merge_threshold: float = Field(
        default=0.76, ge=0.0, le=1.0, validation_alias="TOPIC_AUTO_MERGE_THRESHOLD"
    )
    # Query-time topic match threshold. Tuned via scripts/tune_topic_filter.py
    # against the golden question set: 0.55 Pareto-dominates the old 0.70
    # default (match rate 59.5% → 97.3% at the same per-match precision) on
    # this ontology (~3k rows, median 1 chunk per topic, heavy near-synonym
    # fragmentation like the 7 "spare parts" topics). Re-tune if the topic
    # ontology changes significantly.
    topic_query_match_threshold: float = Field(
        default=0.55, ge=0.0, le=1.0, validation_alias="TOPIC_QUERY_MATCH_THRESHOLD"
    )
    # Top-K per extracted name. K=3 covers the common cluster size without
    # meaningful precision loss (judged 53% vs 59% at K=1) and roughly
    # doubles the chunk pool (13.6 vs 7.2 chunks/query). Reranker handles
    # residual noise.
    topic_query_top_k: int = Field(
        default=3, ge=1, le=20, validation_alias="TOPIC_QUERY_TOP_K"
    )

    # Topic Loosening (query-time). Rarely fires with the tuned primary
    # threshold of 0.55 + top-K=3; retained as a safety belt for operators
    # who tighten the primary threshold via env overrides.
    topic_match_threshold_loose: float = Field(
        default=0.50, ge=0.0, le=1.0, validation_alias="TOPIC_MATCH_THRESHOLD_LOOSE"
    )
    topic_loosening_min_chunks: int = Field(
        default=1, ge=1, le=50, validation_alias="TOPIC_LOOSENING_MIN_CHUNKS"
    )

    # LlamaParse Configuration (legacy binary Office only: .doc, .xls, .ppt).
    # Modern formats and PDFs route to PyMuPDF4LLM -> Gemini instead.
    llama_cloud_api_key: str | None = Field(
        default=None, validation_alias="LLAMA_CLOUD_API_KEY"
    )

    @computed_field
    @property
    def llamaparse_enabled(self) -> bool:
        """LlamaParse is auto-enabled when API key is set."""
        return self.llama_cloud_api_key is not None

    # Gemini PDF parser (via OpenRouter). Used when a PDF is scanned,
    # produces empty text via PyMuPDF, or otherwise classifies COMPLEX.
    gemini_pdf_model: str = Field(
        default="openrouter/google/gemini-2.5-flash",
        validation_alias="GEMINI_PDF_MODEL",
    )
    max_concurrent_gemini_pdf: int = Field(
        default=4, validation_alias="MAX_CONCURRENT_GEMINI_PDF"
    )
    gemini_pdf_page_batch_size: int = Field(
        default=25, validation_alias="GEMINI_PDF_PAGE_BATCH_SIZE"
    )
    gemini_pdf_hard_page_ceiling: int = Field(
        default=200, validation_alias="GEMINI_PDF_HARD_PAGE_CEILING"
    )
    gemini_pdf_max_cost_usd_per_doc: float = Field(
        default=0.50, validation_alias="GEMINI_PDF_MAX_COST_USD_PER_DOC"
    )
    # Wall-clock ceiling per Gemini batch call. Dense scanned PDFs can make a
    # single batch run for minutes while Gemini emits unbounded output, and
    # LiteLLM's built-in ``timeout`` kwarg doesn't reliably cancel. We wrap
    # the call in asyncio.wait_for using this value so the adaptive halving
    # path can trigger in bounded time instead of blocking the ingest loop.
    gemini_pdf_call_timeout_seconds: float = Field(
        default=90.0, validation_alias="GEMINI_PDF_CALL_TIMEOUT_SECONDS"
    )
    # Hard wall-clock cap for parsing a single PDF end-to-end. Independent of
    # per-batch timeout — guards against halving + retries stacking into
    # unbounded total time. On timeout the parser returns partial output and
    # the decider routes the doc to METADATA_ONLY.
    gemini_pdf_doc_timeout_seconds: float = Field(
        default=240.0, validation_alias="GEMINI_PDF_DOC_TIMEOUT_SECONDS"
    )
    # Kill-switch: a Gemini batch that returns more than this many chars per
    # input page is almost always hallucinated repetition on a scanned form.
    # Treat as truncation so the halving path triggers immediately.
    gemini_pdf_max_chars_per_page: int = Field(
        default=20000, validation_alias="GEMINI_PDF_MAX_CHARS_PER_PAGE"
    )
    # Give up on a doc after N total halving events across all batches.
    # Protects against scanned-form PDFs where every batch triggers halving
    # and the recursion burns minutes before hitting the doc timeout.
    gemini_pdf_max_halvings_per_doc: int = Field(
        default=6, validation_alias="GEMINI_PDF_MAX_HALVINGS_PER_DOC"
    )

    @computed_field
    @property
    def gemini_pdf_enabled(self) -> bool:
        """Gemini PDF parser auto-enabled when OpenRouter key is present."""
        return bool(self.openrouter_api_key)

    # Reject attachments above this byte size before loading into memory.
    attachment_max_bytes: int = Field(
        default=100 * 1024 * 1024, validation_alias="ATTACHMENT_MAX_BYTES"
    )

    # Embedding-mode decider thresholds (tune via env; no code change needed).
    # See src/mtss/ingest/embedding_decider.py for the decision tree.
    decider_short_token_threshold: int = Field(
        default=50, validation_alias="DECIDER_SHORT_TOKEN_THRESHOLD"
    )
    decider_no_prose_ratio: float = Field(
        default=0.15, validation_alias="DECIDER_NO_PROSE_RATIO"
    )
    decider_bulk_token_threshold: int = Field(
        default=20_000, validation_alias="DECIDER_BULK_TOKEN_THRESHOLD"
    )
    decider_digit_ratio: float = Field(
        default=0.40, validation_alias="DECIDER_DIGIT_RATIO"
    )
    decider_table_char_pct: float = Field(
        default=0.50, validation_alias="DECIDER_TABLE_CHAR_PCT"
    )
    # Tuned 2026-04-19 from a 150-doc dry-run inventory: prose vessel-status
    # reports were getting demoted to SUMMARY because section headers/footers
    # repeat across hundreds of pages, inflating repetition_score even though
    # the per-section content is meaningful. Sensor logs (the intended target)
    # have *unique* per-row values so this rule never caught them anyway —
    # digit_ratio + table_char_pct are the right discriminators for those.
    decider_repetition_score: float = Field(
        default=0.92, validation_alias="DECIDER_REPETITION_SCORE"
    )
    decider_short_line_ratio: float = Field(
        default=0.95, validation_alias="DECIDER_SHORT_LINE_RATIO"
    )
    decider_medium_token_threshold: int = Field(
        default=50_000, validation_alias="DECIDER_MEDIUM_TOKEN_THRESHOLD"
    )
    decider_medium_prose_ratio: float = Field(
        default=0.50, validation_alias="DECIDER_MEDIUM_PROSE_RATIO"
    )
    embedding_triage_llm_model: str | None = Field(
        default=None, validation_alias="EMBEDDING_TRIAGE_LLM_MODEL"
    )

    # ZIP Extraction Limits (DoS protection)
    zip_max_files: int = Field(default=100, validation_alias="ZIP_MAX_FILES")
    zip_max_depth: int = Field(default=3, validation_alias="ZIP_MAX_DEPTH")
    zip_max_total_size_mb: int = Field(default=500, validation_alias="ZIP_MAX_TOTAL_SIZE_MB")

    # Skip PDFs above this page count during ingest. Large operational logs
    # (800+ page sensor dumps, etc.) cost orders of magnitude more at LlamaParse
    # than they return in useful RAG signal; better to skip and log.
    pdf_max_pages: int = Field(default=40, validation_alias="PDF_MAX_PAGES")

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

    # Concurrency for processing members of a single ZIP attachment.
    # Each member can trigger a vision/LLM call (images) or LlamaParse (docs).
    # Image-heavy ZIPs (e.g. 48 inspection photos) serialise badly at 1 and
    # easily exceed the per-file timeout.
    zip_member_concurrency: int = Field(
        default=5, validation_alias="ZIP_MEMBER_CONCURRENCY"
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
