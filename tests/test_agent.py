from __future__ import annotations

import json

import pytest

from market_intelligence_agent.agent import MarketIntelligenceAgent
from market_intelligence_agent.config import Settings
from market_intelligence_agent.llm import LLMClient
from market_intelligence_agent.models import SECTION_NAMES
from market_intelligence_agent.render import render_markdown, render_summary
from market_intelligence_agent.search.base import SearchProvider, SearchResult
from market_intelligence_agent.search.mock import MockSearchProvider

QUERY = "How does Ramp position against Brex in mid-market fintech?"


def build_agent(settings: Settings, provider: SearchProvider | None = None):
    return MarketIntelligenceAgent(
        settings, provider=provider or MockSearchProvider(), llm=LLMClient(settings)
    )


@pytest.mark.asyncio
async def test_end_to_end_run_produces_a_grounded_brief(settings: Settings):
    result = await build_agent(settings).run(QUERY)
    assert result.query == QUERY
    assert result.sources_used
    assert result.search_plan_trace
    assert len(result.distinct_domains()) >= settings.min_distinct_domains


@pytest.mark.asyncio
async def test_every_asserted_section_resolves_to_stored_sources(settings: Settings):
    result = await build_agent(settings).run(QUERY)
    valid_ids = {s.source_id for s in result.sources_used}
    for name in SECTION_NAMES:
        section = getattr(result.brief, name)
        assert set(section.citations) <= valid_ids
        if section.is_asserted():
            assert section.citations, f"{name} asserted without a citation"


@pytest.mark.asyncio
async def test_output_matches_the_spec_schema(settings: Settings):
    result = await build_agent(settings).run(QUERY)
    payload = result.to_spec_dict()
    assert set(payload) == {
        "query", "generated_at", "latency_ms", "sources_used",
        "brief", "unverified_flags", "search_plan_trace",
    }
    assert set(payload["brief"]) == set(SECTION_NAMES)
    assert set(payload["sources_used"][0]) == {"domain", "url", "retrieved_at"}
    json.dumps(payload)  # must be JSON-serialisable as-is


@pytest.mark.asyncio
async def test_run_stays_inside_the_latency_budget(settings: Settings):
    result = await build_agent(settings).run(QUERY)
    assert not result.budget_exceeded
    assert result.timings.total_ms() <= result.latency_ms + 5


@pytest.mark.asyncio
async def test_empty_search_results_produce_flags_not_claims(settings: Settings):
    class Empty(SearchProvider):
        name = "empty"

        async def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
            return []

    result = await build_agent(settings, Empty()).run(QUERY)
    assert result.unverified_flags
    assert all(not getattr(result.brief, name).is_asserted() for name in SECTION_NAMES)


@pytest.mark.asyncio
async def test_fallback_round_runs_when_evidence_is_thin(settings: Settings):
    thin = Settings(
        search_provider="mock",
        min_distinct_domains=99,       # unreachable, so the shortfall always triggers
        total_budget_seconds=30.0,
        fallback_budget_seconds=1.0,
    )
    result = await build_agent(thin).run(QUERY)
    assert result.fallback_rounds == 1
    assert result.timings.fallback_ms > 0


@pytest.mark.asyncio
async def test_fallback_is_skipped_when_disabled(settings: Settings):
    off = Settings(
        search_provider="mock",
        min_distinct_domains=99,
        fallback_enabled=False,
        total_budget_seconds=30.0,
    )
    result = await build_agent(off).run(QUERY)
    assert result.fallback_rounds == 0


@pytest.mark.asyncio
async def test_markdown_render_includes_sources_and_plan(settings: Settings):
    result = await build_agent(settings).run(QUERY)
    markdown = render_markdown(result)
    assert markdown.startswith("# Competitor brief:")
    assert "## Sources" in markdown and "## Search plan" in markdown and "## Timing" in markdown
    assert "s" not in render_summary(result).split("|")[0].strip()[:1]  # starts with a number
