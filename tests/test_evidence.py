from __future__ import annotations

from datetime import UTC, datetime

from market_intelligence_agent.evidence import EvidenceStore, tokenize
from market_intelligence_agent.models import SourceRecord


def record(source_id: str, domain: str, passage: str, **kwargs) -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        url=f"https://{domain}/{source_id}",
        domain=domain,
        passage=passage,
        retrieved_at=datetime(2026, 8, 30, tzinfo=UTC),
        **kwargs,
    )


def test_tokenize_drops_stopwords_and_short_tokens():
    assert tokenize("The pricing of a company") == {"pricing"}


def test_add_is_idempotent_but_merges_section_hints():
    store = EvidenceStore()
    source = record("s1", "g2.com", "Ramp pricing tiers start at zero.")
    store.add(source, ["pricing"])
    store.add(source, ["strengths"])
    assert len(store) == 1
    assert {r.source_id for r in store.for_section("pricing")} == {"s1"}
    assert {r.source_id for r in store.for_section("strengths")} == {"s1"}


def test_resolve_and_citation_tuples_skip_unknown_ids():
    store = EvidenceStore()
    store.add(record("s1", "g2.com", "Ramp has a free tier."))
    assert [r.source_id for r in store.resolve(["s1", "s99"])] == ["s1"]
    url, passage, retrieved_at = store.citation_tuples(["s1"])[0]
    assert url == "https://g2.com/s1"
    assert passage.startswith("Ramp has")
    assert retrieved_at.startswith("2026-08-30")


def test_enforce_citations_rejects_hallucinated_ids_and_dedupes():
    store = EvidenceStore()
    store.add(record("s1", "g2.com", "real passage"))
    store.add(record("s2", "techcrunch.com", "another real passage"))
    assert store.enforce_citations(["s2", "s99", "s1", "s2"]) == ["s2", "s1"]


def test_enforce_citations_on_all_fake_ids_returns_empty():
    store = EvidenceStore()
    store.add(record("s1", "g2.com", "real passage"))
    assert store.enforce_citations(["s42", "made-up"]) == []


def test_for_section_orders_by_relevance():
    store = EvidenceStore()
    store.add(record("s1", "a.com", "low", relevance=0.2), ["pricing"])
    store.add(record("s2", "b.com", "high", relevance=0.9), ["pricing"])
    assert [r.source_id for r in store.for_section("pricing")] == ["s2", "s1"]


def test_search_ranks_by_lexical_overlap():
    store = EvidenceStore()
    store.add(record("s1", "a.com", "Ramp offers corporate cards and expense management."))
    store.add(record("s2", "b.com", "Unrelated gardening equipment catalogue."))
    hits = store.search("corporate expense management cards")
    assert hits[0].source_id == "s1"


def test_coverage_reports_domains_and_kinds():
    store = EvidenceStore()
    store.add(record("s1", "g2.com", "x", source_kind="review_platform"), ["strengths"])
    store.add(record("s2", "techcrunch.com", "y", source_kind="news"), ["strengths"])
    coverage = store.coverage("strengths")
    assert coverage.domain_count == 2
    assert coverage.kinds == {"review_platform", "news"}
