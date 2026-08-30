from __future__ import annotations

from datetime import timedelta

from market_intelligence_agent.confidence import ConfidenceScorer, FallbackController
from market_intelligence_agent.config import Settings
from market_intelligence_agent.evidence import EvidenceStore
from market_intelligence_agent.models import BriefSection, SourceRecord, utcnow

PASSAGE = "Ramp offers corporate cards with expense management and a free base tier."


def source(source_id: str, domain: str, passage: str = PASSAGE, age_days: int = 30) -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        url=f"https://{domain}/{source_id}",
        domain=domain,
        passage=passage,
        published_at=utcnow() - timedelta(days=age_days),
    )


def test_no_sources_scores_zero_and_flags_missing_evidence(settings: Settings):
    assessment = ConfidenceScorer(settings).score_section("pricing", "Some claim.", [])
    assert assessment.score == 0.0
    assert assessment.missing_evidence


def test_more_distinct_domains_raises_confidence(settings: Settings):
    scorer = ConfidenceScorer(settings)
    single = scorer.score_section("pricing", PASSAGE, [source("s1", "g2.com")])
    many = scorer.score_section(
        "pricing",
        PASSAGE,
        [source("s1", "g2.com"), source("s2", "techcrunch.com"), source("s3", "reuters.com")],
    )
    assert many.score > single.score
    assert many.breakdown.supporting_domains == 3


def test_stale_sources_lower_the_recency_signal(settings: Settings):
    scorer = ConfidenceScorer(settings)
    fresh = scorer.score_section("recent_moves", PASSAGE, [source("s1", "reuters.com", age_days=5)])
    stale = scorer.score_section(
        "recent_moves", PASSAGE, [source("s2", "reuters.com", age_days=2000)]
    )
    assert fresh.breakdown.recency > stale.breakdown.recency
    assert "stale" in " ".join(stale.breakdown.reasons)


def test_hedged_wording_incurs_an_ambiguity_penalty(settings: Settings):
    scorer = ConfidenceScorer(settings)
    plain = scorer.score_section("pricing", PASSAGE, [source("s1", "g2.com")])
    hedged = scorer.score_section(
        "pricing", "Ramp reportedly offers corporate cards and expense management.",
        [source("s1", "g2.com")],
    )
    assert hedged.breakdown.ambiguity_penalty > plain.breakdown.ambiguity_penalty


def test_numeric_disagreement_across_domains_is_flagged(settings: Settings):
    assessment = ConfidenceScorer(settings).score_section(
        "pricing",
        "Pricing starts at a low monthly rate.",
        [
            source("s1", "g2.com", "The plan starts at $12 per user per month."),
            source("s2", "capterra.com", "The plan starts at $30 per user per month."),
        ],
    )
    assert assessment.conflicting


def test_matching_figures_are_not_treated_as_conflict(settings: Settings):
    assessment = ConfidenceScorer(settings).score_section(
        "pricing",
        "Pricing starts at $12 per user.",
        [
            source("s1", "g2.com", "The plan starts at $12 per user per month."),
            source("s2", "capterra.com", "It costs $12 per user per month."),
        ],
    )
    assert not assessment.conflicting


# ------------------------------------------------------------------ fallback control


def weak_assessment(settings: Settings, section: str = "pricing"):
    return ConfidenceScorer(settings).score_section(section, "Claim.", [])


def test_fallback_triggers_on_weak_sections(settings: Settings):
    store = EvidenceStore()
    for i in range(6):
        store.add(source(f"s{i}", f"d{i}.com"))
    decision = FallbackController(settings).decide(
        {"pricing": weak_assessment(settings)}, store, rounds_used=0, seconds_left=40.0
    )
    assert decision.should_retry
    assert decision.weak_sections == ["pricing"]


def test_fallback_is_capped_at_one_round(settings: Settings):
    decision = FallbackController(settings).decide(
        {"pricing": weak_assessment(settings)}, EvidenceStore(), rounds_used=1, seconds_left=40.0
    )
    assert not decision.should_retry
    assert "cap" in decision.reason


def test_fallback_is_skipped_when_the_budget_is_spent(settings: Settings):
    decision = FallbackController(settings).decide(
        {"pricing": weak_assessment(settings)}, EvidenceStore(), rounds_used=0, seconds_left=3.0
    )
    assert not decision.should_retry
    assert "below fallback budget" in decision.reason


def test_fallback_triggers_on_domain_shortfall_alone(settings: Settings):
    store = EvidenceStore()
    store.add(source("s1", "g2.com"))
    strong = ConfidenceScorer(settings).score_section(
        "pricing", PASSAGE, [source("s1", "g2.com"), source("s2", "reuters.com")]
    )
    decision = FallbackController(settings).decide(
        {"pricing": strong}, store, rounds_used=0, seconds_left=40.0
    )
    assert decision.should_retry
    assert any("distinct domains" in gap for gap in decision.gaps)


# ------------------------------------------------------------------ status stamping


def test_uncited_section_becomes_insufficient_data_and_loses_its_text(settings: Settings):
    controller = FallbackController(settings)
    section = BriefSection(text="Unsupported assertion.", citations=[])
    updated, flag = controller.apply("pricing", section, weak_assessment(settings))
    assert updated.status == "insufficient_data"
    assert updated.text == ""
    assert flag and "insufficient data" in flag


def test_conflicting_section_is_flagged_not_asserted(settings: Settings):
    assessment = ConfidenceScorer(settings).score_section(
        "pricing",
        "Pricing is unclear.",
        [
            source("s1", "g2.com", "It starts at $12 per user."),
            source("s2", "capterra.com", "It starts at $30 per user."),
        ],
    )
    section = BriefSection(text="Pricing is unclear.", citations=["s1", "s2"])
    updated, flag = FallbackController(settings).apply("pricing", section, assessment)
    assert updated.status == "conflicting"
    assert not updated.is_asserted()
    assert flag and "conflicting sources" in flag


def test_high_confidence_section_is_asserted_without_a_flag(settings: Settings):
    assessment = ConfidenceScorer(settings).score_section(
        "pricing",
        PASSAGE,
        [
            source("s1", "reuters.com", age_days=10),
            source("s2", "bloomberg.com", age_days=15),
            source("s3", "sec.gov", age_days=20),
        ],
    )
    section = BriefSection(text=PASSAGE, citations=["s1", "s2", "s3"])
    updated, flag = FallbackController(settings).apply("pricing", section, assessment)
    assert updated.status == "grounded"
    assert updated.is_asserted()
    assert flag is None


def test_a_table_of_many_figures_is_not_read_as_disagreement(settings: Settings):
    """A funding leaderboard lists amounts for several different companies. Its spread
    is not evidence that sources disagree about our subject."""
    leaderboard = (
        "1st Smartsheet $121M Madrona. 2nd Notion $418M Temasek. 3rd Anaplan $300M "
        "Premji Invest. 4th Pigment $397M Iconiq. 5th Airtable $271M Thrive."
    )
    assessment = ConfidenceScorer(settings).score_section(
        "market_signals",
        "Funding across the category varies widely.",
        [
            source("s1", "tracxn.com", leaderboard),
            source("s2", "crunchbase.com", "Notion has raised $418M to date."),
        ],
    )
    assert not assessment.conflicting


def test_two_genuine_figures_for_one_subject_still_conflict(settings: Settings):
    """The tabular guard must not swallow a real disagreement."""
    assessment = ConfidenceScorer(settings).score_section(
        "market_signals",
        "Valuation is unclear.",
        [
            source("s1", "reuters.com", "Shein was valued at $45 billion in secondaries."),
            source("s2", "bloomberg.com", "Advisers target a $66 billion valuation."),
        ],
    )
    assert assessment.conflicting
