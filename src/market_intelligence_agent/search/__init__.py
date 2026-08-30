"""Search backends and the factory that picks one from settings."""

from __future__ import annotations

from ..config import Settings
from .base import SearchError, SearchProvider, SearchResult, canonical_url, domain_of
from .mock import MockSearchProvider
from .tavily import TavilySearchProvider

__all__ = [
    "MockSearchProvider",
    "SearchError",
    "SearchProvider",
    "SearchResult",
    "TavilySearchProvider",
    "build_provider",
    "canonical_url",
    "domain_of",
]


def build_provider(settings: Settings) -> SearchProvider:
    """Instantiate the provider named in settings."""
    provider = settings.search_provider.lower()
    if provider == "tavily":
        return TavilySearchProvider(
            settings.tavily_api_key or "",
            timeout=settings.per_request_timeout_seconds,
        )
    if provider == "mock":
        return MockSearchProvider(results_per_query=settings.results_per_subquestion)
    raise SearchError(f"Unknown search provider: {settings.search_provider!r}")
