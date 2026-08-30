from __future__ import annotations

import pytest

from market_intelligence_agent.config import Settings
from market_intelligence_agent.executor import SearchExecutor, classify_source
from market_intelligence_agent.models import SearchPlan, SubQuestion
from market_intelligence_agent.planner import heuristic_plan
from market_intelligence_agent.search.base import SearchProvider, SearchResult
from market_intelligence_agent.search.mock import MockSearchProvider

LONG = "A sufficiently long passage about the competitor to survive the thin-content filter. " * 3


class StubProvider(SearchProvider):
    """Returns a fixed result list for every query."""

    name = "stub"

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    async def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        return list(self._results)


def one_question_plan(query: str = "ramp pricing") -> SearchPlan:
    return SearchPlan(
        query=query,
        subject="Ramp",
        sub_questions=[SubQuestion(id="r0q1", question=query, search_query=query)],
    )


def test_classify_source_maps_known_platforms():
    assert classify_source("g2.com") == "review_platform"
    assert classify_source("blog.crunchbase.com") == "funding_database"
    assert classify_source("techcrunch.com") == "news"
    assert classify_source("ramp.com", subject="Ramp") == "company_site"
    assert classify_source("some-random-blog.net") == "other"


@pytest.mark.asyncio
async def test_executor_drops_denylisted_domains(settings: Settings):
    provider = StubProvider([
        SearchResult(url="https://quora.com/a", title="q", content=LONG, score=0.9),
        SearchResult(url="https://g2.com/b", title="g", content=LONG, score=0.8),
    ])
    report = await SearchExecutor(provider, settings).run(one_question_plan())
    assert report.dropped_denylist == 1
    assert report.distinct_domains() == {"g2.com"}


@pytest.mark.asyncio
async def test_executor_dedupes_urls_and_syndicated_copies(settings: Settings):
    provider = StubProvider([
        SearchResult(url="https://g2.com/a?utm_source=x", content=LONG, score=0.9),
        SearchResult(url="https://g2.com/a", content=LONG, score=0.8),
        SearchResult(url="https://other.com/a", content=LONG, score=0.7),
    ])
    report = await SearchExecutor(provider, settings).run(one_question_plan())
    assert len(report.records) == 1
    assert report.dropped_duplicate == 2


@pytest.mark.asyncio
async def test_executor_drops_thin_content(settings: Settings):
    provider = StubProvider([SearchResult(url="https://g2.com/a", content="too short", score=0.9)])
    report = await SearchExecutor(provider, settings).run(one_question_plan())
    assert report.records == []
    assert report.dropped_thin == 1


@pytest.mark.asyncio
async def test_executor_survives_provider_failure(settings: Settings):
    class Broken(SearchProvider):
        name = "broken"

        async def search(self, query: str, *, max_results: int = 5):
            raise RuntimeError("upstream exploded")

    report = await SearchExecutor(Broken(), settings).run(one_question_plan())
    assert report.records == []
    assert report.errors


@pytest.mark.asyncio
async def test_executor_dedupe_persists_across_rounds(settings: Settings):
    provider = StubProvider([SearchResult(url="https://g2.com/a", content=LONG, score=0.9)])
    executor = SearchExecutor(provider, settings)
    first = await executor.run(one_question_plan(), round_index=0)
    second = await executor.run(one_question_plan(), round_index=1)
    assert len(first.records) == 1
    assert second.records == []


@pytest.mark.asyncio
async def test_full_mock_round_clears_the_five_domain_floor(settings: Settings):
    plan = heuristic_plan("How does Ramp compare to Brex?", settings)
    report = await SearchExecutor(MockSearchProvider(), settings).run(plan)
    assert len(report.distinct_domains()) >= settings.min_distinct_domains
    assert all(r.source_id.startswith("s") for r in report.records)
