"""Brief Synthesizer - stage 5 of the pipeline.

The model only ever sees numbered evidence and is told it may cite nothing else; every
citation it returns is then re-checked against the evidence store, and a section that
ends up with no valid citation loses its text. An extractive synthesiser mirrors the
same contract for offline runs and for the case where the model call fails.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from .config import Settings
from .evidence import EvidenceStore
from .llm import LLMClient, LLMError
from .models import SECTION_NAMES, Brief, BriefSection, SourceRecord

logger = logging.getLogger(__name__)

MAX_EVIDENCE_ITEMS = 24
MAX_PASSAGE_IN_PROMPT = 700

SYNTHESIZER_SYSTEM = """You are the synthesis stage of a market-intelligence research agent.

You will receive a query and a numbered evidence list. Write a competitor brief.

Hard rules:
- Every section's text must be supported by the evidence given. Cite the source ids
  ("s3", "s7") that support it in that section's `citations` list.
- You may ONLY cite ids that appear in the evidence list. Never invent an id.
- If the evidence does not support a section, leave its text empty and its citations
  empty. An empty section is correct; a guessed section is a failure.
- Do not soften a disagreement between sources into a single number. If sources
  conflict, say so in the text and cite both.
- Be specific and compact: 2-4 sentences per section, concrete figures and dates where
  the evidence gives them, no marketing language.
"""


class _SectionDraft(BaseModel):
    text: str = Field(default="", description="2-4 sentences, or empty if unsupported.")
    citations: list[str] = Field(
        default_factory=list, description="Source ids from the evidence list only."
    )


class _BriefDraft(BaseModel):
    """Model-facing schema; confidence and status are set by the scorer, not the model."""

    company_overview: _SectionDraft = Field(default_factory=_SectionDraft)
    positioning: _SectionDraft = Field(default_factory=_SectionDraft)
    pricing: _SectionDraft = Field(default_factory=_SectionDraft)
    recent_moves: _SectionDraft = Field(default_factory=_SectionDraft)
    strengths: _SectionDraft = Field(default_factory=_SectionDraft)
    weaknesses: _SectionDraft = Field(default_factory=_SectionDraft)
    market_signals: _SectionDraft = Field(default_factory=_SectionDraft)
    conflicts_noted: list[str] = Field(
        default_factory=list, description="Disagreements the evidence exposed."
    )


SectionStatus = Literal["grounded", "unverified", "conflicting", "insufficient_data"]


def render_evidence(records: list[SourceRecord]) -> str:
    """Numbered evidence block. Ids here are the only citable vocabulary."""
    lines = []
    for record in records[:MAX_EVIDENCE_ITEMS]:
        published = record.published_at.date().isoformat() if record.published_at else "undated"
        passage = record.passage[:MAX_PASSAGE_IN_PROMPT]
        lines.append(
            f"[{record.source_id}] domain={record.domain} kind={record.source_kind} "
            f"published={published} url={record.url}\n"
            f"    title: {record.title}\n"
            f"    passage: {passage}"
        )
    return "\n\n".join(lines)


class BriefSynthesizer:
    """Turns the evidence store into a structured brief with enforced citations."""

    # Ordered cheapest-last, so degrade_effort() always steps down and never up.
    _EFFORT_LADDER = ("max", "xhigh", "high", "medium", "low")

    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self._llm = llm
        self._settings = settings
        self._effort = settings.synthesizer_effort

    def degrade_effort(self) -> str:
        """Step synthesis effort down one rung. Called when the run is behind schedule."""
        ladder = self._EFFORT_LADDER
        if self._effort in ladder:
            index = ladder.index(self._effort)
            self._effort = ladder[min(index + 1, len(ladder) - 1)]
        else:
            self._effort = ladder[-1]
        return self._effort

    def reset_effort(self) -> None:
        """Restore the configured effort. The agent instance is reused across queries."""
        self._effort = self._settings.synthesizer_effort

    async def synthesize(
        self,
        query: str,
        store: EvidenceStore,
        *,
        timeout: float | None = None,
    ) -> tuple[Brief, list[str]]:
        """Return the brief plus any conflicts the synthesiser explicitly noted."""
        records = sorted(store.all(), key=lambda r: r.relevance, reverse=True)
        if not records:
            return Brief(), ["no evidence was retrieved for this query"]

        if not self._llm.available:
            return extractive_brief(query, store), []

        prompt = (
            f"Query:\n{query}\n\nEvidence:\n{render_evidence(records)}\n\n"
            "Write the brief now. Cite only the ids above."
        )
        try:
            draft = await self._llm.structured(
                _BriefDraft,
                system=SYNTHESIZER_SYSTEM,
                user=prompt,
                effort=self._effort,
                max_tokens=8000,
                timeout=timeout,
            )
        except LLMError as exc:
            logger.warning("synthesis fell back to extractive mode: %s", exc)
            return extractive_brief(query, store), [f"synthesis degraded: {exc}"]

        return self._materialise(draft, store), list(draft.conflicts_noted)

    @staticmethod
    def _materialise(draft: _BriefDraft, store: EvidenceStore) -> Brief:
        """Re-check every citation against the store; unsupported text is discarded."""
        brief = Brief()
        for name in SECTION_NAMES:
            section_draft: _SectionDraft = getattr(draft, name)
            citations = store.enforce_citations(section_draft.citations)
            text = section_draft.text.strip()
            if not citations:
                # Hard constraint from section 9 of the spec: no citation, no claim.
                setattr(brief, name, BriefSection(text="", citations=[],
                                                  status="insufficient_data"))
                continue
            setattr(brief, name, BriefSection(text=text, citations=citations))
        return brief


# --------------------------------------------------------------------- extractive


# Cue words that make a passage a plausible extract for a given section.
_SECTION_CUES: dict[str, tuple[str, ...]] = {
    "company_overview": ("founded", "headquarters", "company", "platform", "provides", "offers"),
    "positioning": ("positions", "competes", "alternative", "category", "versus", "differentiat"),
    "pricing": ("pricing", "price", "per user", "per month", "plan", "tier", "free", "$"),
    "recent_moves": ("launched", "announced", "raised", "acquired", "released", "2025", "2026"),
    "strengths": ("strength", "praise", "easy", "best", "advantage", "pros", "loved"),
    "weaknesses": ("weakness", "complaint", "lacks", "limitation", "cons", "expensive", "slow"),
    "market_signals": ("market", "growth", "hiring", "headcount", "share", "funding", "valuation"),
}


def extractive_brief(query: str, store: EvidenceStore, *, per_section: int = 3) -> Brief:
    """Model-free synthesis: quote the best-matching passages and cite them.

    Deliberately conservative - it summarises nothing it cannot point at, which keeps the
    offline path honest under the same citation rules as the model path.
    """
    brief = Brief()
    for name in SECTION_NAMES:
        cues = " ".join(_SECTION_CUES[name])
        candidates = store.for_section(name, limit=per_section) or store.search(
            f"{query} {cues}", limit=per_section
        )
        if not candidates:
            setattr(brief, name, BriefSection(text="", citations=[], status="insufficient_data"))
            continue
        sentences = []
        for record in candidates:
            head = record.passage.split(". ")[0].strip()
            if head:
                sentences.append(f"{head} [{record.source_id}]")
        setattr(
            brief,
            name,
            BriefSection(
                text=". ".join(sentences)[:900],
                citations=[r.source_id for r in candidates],
            ),
        )
    return brief
