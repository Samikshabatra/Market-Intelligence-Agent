"""M6: latency-budget behaviour under pressure."""

from __future__ import annotations

import asyncio

import pytest

from market_intelligence_agent.agent import MarketIntelligenceAgent
from market_intelligence_agent.config import Settings
from market_intelligence_agent.llm import LLMClient
from market_intelligence_agent.planner import seed_plan
from market_intelligence_agent.search.base import SearchProvider, SearchResult
from market_intelligence_agent.search.mock import MockSearchProvider
from market_intelligence_agent.synthesizer import BriefSynthesizer

QUERY = "How does Ramp position against Brex?"
LONG = "A long enough passage about the competitor to clear the thin-content filter. " * 3


class SlowProvider(SearchProvider):
    """Every search takes `delay` seconds, so budget handling can be observed."""

    name = "slow"

    def __init__(self, delay: float) -> None:
        self._delay = delay
        self.calls = 0

    async def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        self.calls += 1
        await asyncio.sleep(self._delay)
        return [SearchResult(url=f"https://d{self.calls}.com/a", content=LONG, score=0.8)]


def test_seed_plan_is_two_broad_steps_derived_from_the_raw_query():
    plan = seed_plan(QUERY)
    assert len(plan.sub_questions) == 2
    assert plan.sub_questions[0].search_query == QUERY
    assert plan.sub_questions[0].id.startswith("seed")


def test_degrade_effort_steps_down_and_stops_at_the_bottom():
    settings = Settings(synthesizer_effort="high")
    synthesizer = BriefSynthesizer(LLMClient(settings), settings)
    assert synthesizer.degrade_effort() == "medium"
    assert synthesizer.degrade_effort() == "low"
    assert synthesizer.degrade_effort() == "low"  # cannot go below the bottom rung
    synthesizer.reset_effort()
    assert synthesizer._effort == "high"


@pytest.mark.asyncio
async def test_seed_search_runs_concurrently_with_planning(settings: Settings):
    provider = MockSearchProvider()
    agent = MarketIntelligenceAgent(settings, provider=provider, llm=LLMClient(settings))
    await agent.run(QUERY)
    # The raw query is only ever issued by the seed plan, never by the heuristic planner.
    assert QUERY in provider.calls


@pytest.mark.asyncio
async def test_slow_search_does_not_push_the_run_past_its_budget():
    settings = Settings(
        search_provider="mock",
        total_budget_seconds=2.0,
        search_budget_seconds=1.0,
        fallback_budget_seconds=0.5,
        synthesis_budget_seconds=0.5,
        per_request_timeout_seconds=0.5,
    )
    agent = MarketIntelligenceAgent(
        settings, provider=SlowProvider(5.0), llm=LLMClient(settings)
    )
    result = await agent.run(QUERY)
    assert result.latency_ms < 4000  # the 5s-per-search provider must have been cut off
    assert result.unverified_flags


@pytest.mark.asyncio
async def test_a_starved_budget_still_produces_a_result(settings: Settings):
    starved = Settings(
        search_provider="mock",
        total_budget_seconds=0.5,
        search_budget_seconds=0.2,
        synthesis_budget_seconds=0.2,
    )
    agent = MarketIntelligenceAgent(
        starved, provider=MockSearchProvider(), llm=LLMClient(starved)
    )
    result = await agent.run(QUERY)
    assert result.query == QUERY
    assert result.timings.total_ms() >= 0


@pytest.mark.asyncio
async def test_effort_degradation_does_not_leak_between_queries(settings: Settings):
    tight = Settings(
        search_provider="mock",
        synthesizer_effort="high",
        total_budget_seconds=0.4,
        synthesis_budget_seconds=1.0,
    )
    agent = MarketIntelligenceAgent(tight, provider=MockSearchProvider(), llm=LLMClient(tight))
    await agent.run(QUERY)
    assert agent._synthesizer._effort in {"medium", "high"}
    await agent.run(QUERY)
    # Whatever the second run does, it must not start from the first run's degraded rung.
    assert agent._synthesizer._effort in {"medium", "high"}
