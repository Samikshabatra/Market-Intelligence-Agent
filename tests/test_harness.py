from __future__ import annotations

from pathlib import Path

import pytest

from eval.harness import (
    QuerySpec,
    load_query_set,
    percentile,
    rubric_rows,
    score_result,
    summarise,
)
from market_intelligence_agent.agent import MarketIntelligenceAgent
from market_intelligence_agent.config import Settings
from market_intelligence_agent.llm import LLMClient
from market_intelligence_agent.models import AgentResult, BriefSection, SourceRecord
from market_intelligence_agent.search.mock import MockSearchProvider

QUERY_SET = Path(__file__).resolve().parents[1] / "eval" / "queries.yaml"
CORPUS = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "offline_corpus.json"


def test_query_set_has_twenty_five_queries_across_all_categories():
    specs = load_query_set(QUERY_SET)
    assert len(specs) == 25
    assert {s.category for s in specs} == {
        "comparison", "pricing", "recent", "sparse", "conflicting"
    }


def test_query_set_ids_are_unique():
    specs = load_query_set(QUERY_SET)
    assert len({s.id for s in specs}) == 25


def test_sparse_and_conflicting_queries_expect_a_flag():
    specs = load_query_set(QUERY_SET)
    for spec in specs:
        if spec.category in {"sparse", "conflicting"}:
            assert spec.expect_flag, f"{spec.id} should expect a flag"


def test_percentile_uses_nearest_rank():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9) == 9
    assert percentile([], 0.9) == 0.0


def test_score_result_flags_a_fabricated_citation():
    result = AgentResult(query="q")
    result.sources_used = [SourceRecord(source_id="s1", url="https://a.com/1", domain="a.com")]
    result.brief.pricing = BriefSection(text="A claim.", citations=["s1", "s404"])
    metrics = score_result(QuerySpec(id="q1", query="q"), result)
    assert metrics.citation_validity == 0.5
    assert metrics.groundedness == 0.0  # the asserted section is not fully grounded


def test_score_result_treats_a_declined_section_as_a_decline():
    result = AgentResult(query="q")
    result.brief.pricing = BriefSection(text="", citations=[], status="insufficient_data")
    metrics = score_result(
        QuerySpec(id="q1", query="q", expect_flag=True, focus_sections=["pricing"]), result
    )
    assert metrics.declined_to_assert
    assert metrics.correct_flag_behaviour


def test_summarise_fails_the_suite_on_an_invalid_citation():
    result = AgentResult(query="q")
    result.sources_used = [SourceRecord(source_id="s1", url="https://a.com/1", domain="a.com")]
    result.brief.pricing = BriefSection(text="A claim.", citations=["s404"])
    metrics = [score_result(QuerySpec(id="q1", query="q"), result)]
    report = summarise("test", metrics, "2026-08-30T00:00:00+00:00")
    assert not report.passed
    assert any("citation validity" in failure for failure in report.failures)


def test_summarise_excludes_sparse_queries_from_the_domain_floor():
    def thin(spec_id: str, expect_flag: bool):
        result = AgentResult(query="q")
        result.sources_used = [
            SourceRecord(source_id="s1", url="https://a.com/1", domain="a.com")
        ]
        return score_result(
            QuerySpec(id=spec_id, query="q", expect_flag=expect_flag), result
        )

    # One answerable query below the floor, one sparse query below it: the sparse one
    # must not drag the rate down, because thin evidence is the correct outcome there.
    only_sparse_is_thin = summarise("t", [thin("q2", True)], "2026-08-30T00:00:00+00:00")
    assert only_sparse_is_thin.domain_floor_rate == 1.0

    answerable_is_thin = summarise("t", [thin("q1", False)], "2026-08-30T00:00:00+00:00")
    assert answerable_is_thin.domain_floor_rate == 0.0


def test_rubric_rows_emit_one_line_per_query_with_a_blank_accuracy_column():
    result = AgentResult(query="q")
    metrics = [score_result(QuerySpec(id="q1", query="q"), result)]
    rows = rubric_rows(metrics).splitlines()
    assert rows[0].endswith("accuracy_0_to_1,reviewer_notes")
    assert rows[1].startswith("q1,") and rows[1].endswith(",")


@pytest.mark.asyncio
async def test_sparse_fixture_query_declines_to_assert():
    settings = Settings(
        search_provider="mock",
        mock_corpus_path=str(CORPUS),
        total_budget_seconds=15.0,
    )
    agent = MarketIntelligenceAgent(
        settings,
        provider=MockSearchProvider(corpus_path=CORPUS),
        llm=LLMClient(settings),
    )
    result = await agent.run("How does Zeitwerk Analytics position against its competitors?")
    assert result.unverified_flags
    assert not result.brief.positioning.is_asserted()


@pytest.mark.asyncio
async def test_conflicting_fixture_query_is_marked_conflicting():
    settings = Settings(
        search_provider="mock",
        mock_corpus_path=str(CORPUS),
        total_budget_seconds=15.0,
    )
    agent = MarketIntelligenceAgent(
        settings,
        provider=MockSearchProvider(corpus_path=CORPUS),
        llm=LLMClient(settings),
    )
    result = await agent.run("What is OpenAI's current annualised revenue?")
    statuses = {name: getattr(result.brief, name).status for name, _ in result.brief.sections()}
    assert "conflicting" in statuses.values()
    assert any("conflicting sources" in flag for flag in result.unverified_flags)
