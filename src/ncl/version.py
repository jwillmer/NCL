"""Version information for the NCL application.

The Git SHA is injected at build time via Docker build args
or can be set via environment variable.
"""

import os

# Read from environment variable, default to "development" if not set
GIT_SHA: str = os.getenv("GIT_SHA", "development")
GIT_SHA_SHORT: str = GIT_SHA[:8] if len(GIT_SHA) >= 8 else GIT_SHA

# Application version (should match pyproject.toml)
APP_VERSION: str = "0.1.0"


def get_version_info() -> dict:
    """Return version information as a dictionary."""
    return {
        "version": APP_VERSION,
        "git_sha": GIT_SHA,
        "git_sha_short": GIT_SHA_SHORT,
    }
