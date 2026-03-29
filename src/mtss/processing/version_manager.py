"""Re-export from new location for backward compatibility."""

# Re-export all public names
from ..ingest.version_manager import IngestDecision, VersionManager  # noqa: F401

# Re-export the config dependency so patches like
# `patch("mtss.processing.version_manager.get_settings")` still work
from ..config import get_settings  # noqa: F401
