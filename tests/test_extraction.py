"""Passage cleaning and extractive section quality.

Every case here came from an actual live Tavily run - the raw extractor output was
being quoted verbatim into the brief.
"""

from __future__ import annotations

from market_intelligence_agent.evidence import EvidenceStore
from market_intelligence_agent.executor import clean_passage
from market_intelligence_agent.models import SECTION_NAMES, SourceRecord
from market_intelligence_agent.synthesizer import extractive_brief


def test_clean_strips_title_prefix():
    assert clean_passage("Title: Coda vs Notion") == "Coda vs Notion"


def test_clean_strips_markdown_headings():
    assert clean_passage("## Should You Use Coda? Yes.") == "Should You Use Coda? Yes."


def test_clean_unwraps_markdown_links_and_drops_images():
    assert clean_passage("See [the pricing page](https://x.com/p) now") == (
        "See the pricing page now"
    )
    assert clean_passage("![logo](https://x.com/l.png) Notion") == "Notion"


def test_clean_flattens_table_pipes():
    assert clean_passage("| Rank | Company | --- | 1st | Notion |") == "Rank Company 1st Notion"


def test_clean_drops_elision_markers_without_mangling_them():
    # Collapsing runs of dots turned "[...]" into a stray "[.]" that read as a sentence.
    assert clean_passage("First part [...] second part") == "First part second part"
    assert "[.]" not in clean_passage("a [...] b")


def test_clean_keeps_ordinary_sentence_punctuation():
    assert clean_passage("Notion costs $10/user. Coda bills makers only!") == (
        "Notion costs $10/user. Coda bills makers only!"
    )


def test_clean_removes_replacement_and_zero_width_characters():
    assert clean_passage("May 27, 2026�8 min read�") == "May 27, 2026 8 min read"


def source(source_id: str, domain: str, passage: str, relevance: float = 0.8) -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        url=f"https://{domain}/{source_id}",
        domain=domain,
        passage=passage,
        relevance=relevance,
    )


def build_store() -> EvidenceStore:
    """One sub-question feeding several sections - the shape that caused duplication."""
    store = EvidenceStore()
    every_section = list(SECTION_NAMES)
    store.add(
        source("s1", "a.com", "Notion pricing is $10 per user per month on the Plus plan."),
        every_section,
    )
    store.add(
        source("s2", "b.com", "Users praise Notion as the best and easiest tool to adopt."),
        every_section,
    )
    store.add(
        source("s3", "c.com", "The main complaint is that reporting lacks depth and feels slow."),
        every_section,
    )
    store.add(
        source("s4", "d.com", "Notion raised funding and announced new agents in 2026."),
        every_section,
    )
    store.add(
        source("s5", "e.com", "Notion competes as an alternative in the workspace category."),
        every_section,
    )
    return store


def test_sections_fed_by_one_subquestion_do_not_repeat_each_other():
    brief = extractive_brief("Notion overview", build_store(), per_section=1)
    texts = [getattr(brief, name).text for name in SECTION_NAMES if getattr(brief, name).text]
    assert len(texts) == len(set(texts)), "sections must not be byte-identical"


def test_a_section_with_no_relevant_evidence_declines_rather_than_padding():
    """5 sources cannot honestly fill 7 sections. Sections with nothing relevant to say
    must report insufficient_data instead of recycling another section's passage."""
    brief = extractive_brief("Notion overview", build_store(), per_section=1)
    declined = [n for n in SECTION_NAMES if getattr(brief, n).status == "insufficient_data"]
    assert declined, "expected at least one section to decline on thin evidence"
    for name in declined:
        assert getattr(brief, name).text == ""


def test_cue_ranking_sends_each_passage_to_the_right_section():
    brief = extractive_brief("Notion overview", build_store(), per_section=1)
    assert "s1" in brief.pricing.citations          # the pricing passage
    assert "s2" in brief.strengths.citations        # "praise", "best", "easiest"
    assert "s3" in brief.weaknesses.citations       # "complaint", "lacks", "slow"


def test_section_order_does_not_decide_who_gets_the_best_source():
    """company_overview is scored first; it must not be able to claim the pricing
    passage simply by running before pricing does."""
    brief = extractive_brief("Notion overview", build_store(), per_section=1)
    assert "s1" not in brief.company_overview.citations
    assert "s1" in brief.pricing.citations


def test_extractive_sentences_avoid_boilerplate_leads():
    store = EvidenceStore()
    store.add(
        source(
            "s1",
            "a.com",
            "Published February 19, 2026 in App Alternatives. "
            "Coda bills only doc makers at $36 per maker per month on the Team plan.",
        ),
        ["pricing"],
    )
    brief = extractive_brief("Coda pricing", store, per_section=1)
    assert "$36 per maker" in brief.pricing.text
    assert not brief.pricing.text.startswith("Published February")
