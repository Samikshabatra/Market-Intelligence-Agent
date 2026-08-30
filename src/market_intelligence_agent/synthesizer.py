"""Brief Synthesizer - stage 5 of the pipeline.

The model only ever sees numbered evidence and is told it may cite nothing else; every
citation it returns is then re-checked against the evidence store, and a section that
ends up with no valid citation loses its text. An extractive synthesiser mirrors the
same contract for offline runs and for the case where the model call fails.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel, Field

from .config import Settings
from .evidence import EvidenceStore
from .llm import LLMClient, LLMError
from .models import SECTION_NAMES, Brief, BriefSection, SourceRecord

logger = logging.getLogger(__name__)

MAX_EVIDENCE_ITEMS = 24
# Measured: halving this cut synthesis latency from ~19s to ~11s with no loss of
# filled sections. Every retrieved source stays citable; only the quoted span shrinks.
MAX_PASSAGE_IN_PROMPT = 450

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
                # Seven sections of 2-4 sentences each. A larger ceiling does not
                # improve the brief, it just lengthens generation inside the budget.
                max_tokens=4000,
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


def _mentions_section(record: SourceRecord, cues: tuple[str, ...]) -> bool:
    """Whether the passage says anything about this section's subject at all."""
    haystack = f"{record.title} {record.passage}".lower()
    return any(cue in haystack for cue in cues)


def _cue_score(record: SourceRecord, cues: tuple[str, ...]) -> float:
    """How strongly a passage reads like evidence for this particular section."""
    haystack = f"{record.title} {record.passage}".lower()
    hits = sum(1 for cue in cues if cue in haystack)
    return hits / len(cues) + 0.2 * record.relevance


def _best_sentence(passage: str, cues: tuple[str, ...]) -> str:
    """Pick the sentence that actually carries the section's subject.

    Taking the first sentence blindly is what produced boilerplate leads ("Title: Coda
    vs", "Should You Use..."); scoring by cue hits picks the sentence with the substance.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage) if len(s.strip()) > 40]
    if not sentences:
        return passage.strip()
    scored = [
        (sum(1 for cue in cues if cue in s.lower()), -index, s)
        for index, s in enumerate(sentences)
    ]
    return max(scored)[2]


MAX_COUNTERPARTS = 2


def _contradicting(
    chosen: list[SourceRecord],
    pool: list[SourceRecord],
) -> list[SourceRecord]:
    """Sources from other domains quoting a figure of a kind the section already quotes.

    A disagreement only exists where two figures sit side by side. Section assignment is
    otherwise exclusive, so without this a contradicting source is filed elsewhere and
    the confidence scorer never sees the two together.
    """
    from .confidence import ConfidenceScorer

    picked = {r.source_id for r in chosen}
    families = {
        family
        for record in chosen
        for family, values in ConfidenceScorer._extract_figures(record.passage).items()
        if values
    }
    if not families:
        return []

    domains = {r.domain for r in chosen}
    counterparts = []
    for record in pool:
        if record.source_id in picked or record.domain in domains:
            continue
        figures = ConfidenceScorer._extract_figures(record.passage)
        if any(figures[family] for family in families):
            counterparts.append(record)
            if len(counterparts) >= MAX_COUNTERPARTS:
                break
    return counterparts


def extractive_brief(query: str, store: EvidenceStore, *, per_section: int = 3) -> Brief:
    """Model-free synthesis: quote the best-matching passages and cite them.

    Deliberately conservative - it summarises nothing it cannot point at, which keeps the
    offline path honest under the same citation rules as the model path.

    The planner maps one sub-question to several sections, so the naive version handed
    every one of those sections the same candidates and produced byte-identical text for
    (say) strengths and weaknesses. Candidates are therefore re-ranked per section by
    cue overlap, and a passage already used elsewhere is only reused when a section has
    nothing else to stand on.
    """
    brief = Brief()

    # Which sources are even in play for each section.
    pools: dict[str, list[SourceRecord]] = {}
    for name in SECTION_NAMES:
        cues = _SECTION_CUES[name]
        pools[name] = store.for_section(name, limit=per_section * 3) or store.search(
            f"{query} {' '.join(cues)}", limit=per_section * 3
        )

    # Assign each source to the section it fits best, before any section picks. Letting
    # sections choose in declaration order made the first one claim the strongest passage
    # whatever it was about - company_overview would take the pricing passage, and
    # pricing then quoted a leftover.
    best_fit: dict[str, list[SourceRecord]] = {name: [] for name in SECTION_NAMES}
    for record in {r.source_id: r for pool in pools.values() for r in pool}.values():
        eligible = [name for name in SECTION_NAMES if record in pools[name]]
        if not eligible:
            continue
        winner = max(eligible, key=lambda name: _cue_score(record, _SECTION_CUES[name]))
        best_fit[winner].append(record)

    for name in SECTION_NAMES:
        cues = _SECTION_CUES[name]
        if not pools[name]:
            setattr(brief, name, BriefSection(text="", citations=[], status="insufficient_data"))
            continue

        preferred = sorted(best_fit[name], key=lambda r: _cue_score(r, cues), reverse=True)
        candidates = preferred[:per_section]
        if len(candidates) < per_section:
            # Top up from the section's own pool, but only with passages that actually
            # mention something this section is about. Recycling an unrelated passage
            # would pad the brief with a correctly-cited claim that is nonetheless not
            # evidence for this section - the padding failure the confidence stage
            # exists to prevent.
            chosen = {r.source_id for r in candidates}
            fallback = sorted(pools[name], key=lambda r: _cue_score(r, cues), reverse=True)
            candidates += [
                r
                for r in fallback
                if r.source_id not in chosen and _mentions_section(r, cues)
            ][: per_section - len(candidates)]

        # Best-fit assignment is exclusive, which quietly hid disagreements: two sources
        # quoting different market-share figures would land in different sections, and a
        # section holding one figure has nothing to disagree with. Pull the counterpart
        # back in, so a contradiction is visible where the claim is made.
        candidates += _contradicting(candidates, pools[name])

        if not candidates:
            setattr(brief, name, BriefSection(text="", citations=[], status="insufficient_data"))
            continue

        sentences = []
        for record in candidates:
            sentence = _best_sentence(record.passage, cues)
            if sentence:
                sentences.append(f"{sentence.rstrip('.')} [{record.source_id}]")
        setattr(
            brief,
            name,
            BriefSection(
                text=". ".join(sentences)[:900],
                citations=[r.source_id for r in candidates],
            ),
        )
    return brief
