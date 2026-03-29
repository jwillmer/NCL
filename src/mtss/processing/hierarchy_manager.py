"""Re-export from new location for backward compatibility."""

from ..ingest.hierarchy_manager import HierarchyManager  # noqa: F401

# Re-export dependencies so patches still work
from ..config import get_settings  # noqa: F401
