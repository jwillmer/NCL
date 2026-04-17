"""OpenRouter privacy guard.

Forwards ``provider.data_collection = "deny"`` on every LLM / embedding /
rerank call so OpenRouter refuses to route to upstream providers that log
prompts or retain data. Fails closed: if no compliant provider is available,
the request errors rather than silently falling back to a logging endpoint.

See: https://openrouter.ai/docs/features/privacy-and-logging
"""

from __future__ import annotations

from typing import Any, Dict

OPENROUTER_PRIVACY_EXTRA_BODY: Dict[str, Any] = {
    "provider": {"data_collection": "deny"},
}
