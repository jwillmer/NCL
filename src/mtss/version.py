"""Version information for the MTSS application.

The Git SHA + build time are injected at image build via Docker build args
(GIT_SHA, BUILD_TIME) or can be set via environment variable for local runs.
"""

import os

# Read from environment variable, default to "development" if not set
GIT_SHA: str = os.getenv("GIT_SHA", "development")
GIT_SHA_SHORT: str = GIT_SHA[:8] if len(GIT_SHA) >= 8 else GIT_SHA

# ISO-8601 UTC timestamp stamped into the image at build time. Empty string
# in local dev — the frontend falls back to "dev" in that case.
BUILD_TIME: str = os.getenv("BUILD_TIME", "")

# Application version (should match pyproject.toml)
APP_VERSION: str = "1.0.0"


def get_version_info() -> dict:
    """Return version information as a dictionary."""
    return {
        "version": APP_VERSION,
        "git_sha": GIT_SHA,
        "git_sha_short": GIT_SHA_SHORT,
        "build_time": BUILD_TIME,
    }
