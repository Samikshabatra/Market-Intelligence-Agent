from __future__ import annotations

import pytest

from market_intelligence_agent.config import Settings
from market_intelligence_agent.search.mock import MockSearchProvider


@pytest.fixture()
def settings() -> Settings:
    """Offline settings: mock provider, no credentials, tight budgets for fast tests."""
    return Settings(
        anthropic_api_key=None,
        tavily_api_key=None,
        search_provider="mock",
        total_budget_seconds=10.0,
        search_budget_seconds=5.0,
    )


@pytest.fixture()
def provider() -> MockSearchProvider:
    return MockSearchProvider()
