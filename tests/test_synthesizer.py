from __future__ import annotations

import pytest

from market_intelligence_agent.config import Settings
from market_intelligence_agent.evidence import EvidenceStore
from market_intelligence_agent.llm import LLMClient
from market_intelligence_agent.models import SECTION_NAMES, SourceRecord
from market_intelligence_agent.synthesizer import (
    BriefSynthesizer,
    _BriefDraft,
    _SectionDraft,
    extractive_brief,
    render_evidence,
)


def source(source_id: str, domain: str, passage: str) -> SourceRecord:
    return SourceRecord(
        source_id=source_id, url=f"https://{domain}/{source_id}", domain=domain, passage=passage
    )


@pytest.fixture()
def store() -> EvidenceStore:
    store = EvidenceStore()
    store.add(
        source("s1", "ramp.com", "Ramp provides corporate cards and expense management."),
        ["company_overview", "pricing"],
    )
    store.add(
        source("s2", "g2.com", "Reviewers praise the onboarding but complain about reporting."),
        ["strengths", "weaknesses"],
    )
    store.add(
        source("s3", "techcrunch.com", "Ramp announced a new AI agent product in 2026."),
        ["recent_moves"],
    )
    return store


def test_render_evidence_exposes_ids_domains_and_urls(store: EvidenceStore):
    block = render_evidence(store.all())
    assert "[s1]" in block and "domain=ramp.com" in block and "url=https://ramp.com/s1" in block


def test_materialise_strips_hallucinated_citations(store: EvidenceStore):
    draft = _BriefDraft(
        pricing=_SectionDraft(text="Pricing is usage-based.", citations=["s1", "s99"]),
    )
    brief = BriefSynthesizer._materialise(draft, store)
    assert brief.pricing.citations == ["s1"]
    assert brief.pricing.text == "Pricing is usage-based."


def test_materialise_discards_text_with_no_valid_citation(store: EvidenceStore):
    draft = _BriefDraft(
        positioning=_SectionDraft(text="It is the clear category leader.", citations=["s99"]),
    )
    brief = BriefSynthesizer._materialise(draft, store)
    assert brief.positioning.text == ""
    assert brief.positioning.status == "insufficient_data"


def test_materialise_leaves_unset_sections_as_insufficient_data(store: EvidenceStore):
    brief = BriefSynthesizer._materialise(_BriefDraft(), store)
    assert all(getattr(brief, name).status == "insufficient_data" for name in SECTION_NAMES)


def test_extractive_brief_only_cites_stored_ids(store: EvidenceStore):
    brief = extractive_brief("Ramp overview", store)
    for name in SECTION_NAMES:
        section = getattr(brief, name)
        assert all(citation in store for citation in section.citations)


def test_extractive_brief_grounds_the_hinted_sections(store: EvidenceStore):
    brief = extractive_brief("Ramp overview", store)
    assert "s3" in brief.recent_moves.citations
    assert brief.recent_moves.text


@pytest.mark.asyncio
async def test_synthesize_without_evidence_flags_it(settings: Settings):
    synthesizer = BriefSynthesizer(LLMClient(settings), settings)
    brief, conflicts = await synthesizer.synthesize("anything", EvidenceStore())
    assert conflicts == ["no evidence was retrieved for this query"]
    assert brief.pricing.text == ""


@pytest.mark.asyncio
async def test_synthesize_uses_extractive_path_offline(settings: Settings, store: EvidenceStore):
    synthesizer = BriefSynthesizer(LLMClient(settings), settings)
    brief, conflicts = await synthesizer.synthesize("Ramp overview", store)
    assert conflicts == []
    assert brief.company_overview.citations
