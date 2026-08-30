"""Confidence Scorer & Fallback Controller - stage 4 of the pipeline.

Scoring is rule-based on purpose: the spec asks for heuristics first (source agreement,
recency, authority, extraction ambiguity) so the score is explainable and auditable
against the eval set before any learned scorer is layered on.

The controller turns low scores into one bounded extra search round, then into explicit
"unverified" / "conflicting" flags rather than an asserted claim.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .config import Settings
from .evidence import EvidenceStore, tokenize
from .models import BriefSection, ConfidenceBreakdown, SourceRecord, utcnow

# Weights sum to 1.0; the ambiguity penalty is subtracted afterwards.
W_CORROBORATION = 0.45
W_AUTHORITY = 0.30
W_RECENCY = 0.25

# Phrases that signal the passage itself is hedged or speculative, which should not be
# laundered into a confident claim.
_HEDGE_PATTERNS = re.compile(
    r"\b(reportedly|rumou?red|allegedly|may|might|could|appears to|is said to|"
    r"unconfirmed|we believe|estimated|approximately|roughly)\b",
    re.IGNORECASE,
)

# Numeric disagreement across sources is the most common conflict in market research:
# money for pricing and valuation queries, percentages for market share, and plain large
# magnitudes for headcount. Each family is compared only against its own kind, so "$45
# billion" is never compared with "22,000 employees".
_MONEY = re.compile(r"\$\s?([0-9][0-9,]*(?:\.[0-9]+)?)\s*(billion|million|bn|m|k)?", re.IGNORECASE)
_PERCENT = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s?(?:%|per ?cent)", re.IGNORECASE)
_MAGNITUDE = re.compile(
    r"\b([0-9][0-9,]{2,}(?:\.[0-9]+)?)\s*(?:\+)?\s*"
    r"(?:employees|staff|people|customers|users|headcount)",
    re.IGNORECASE,
)
_SCALE = {"k": 1e3, "m": 1e6, "bn": 1e9, "million": 1e6, "billion": 1e9}

# A figure has to differ by more than this to count as a real disagreement rather than
# rounding or a reporting-window difference.
CONFLICT_TOLERANCE = 0.25


@dataclass(slots=True)
class SectionAssessment:
    """Scoring outcome for one section, including why it scored the way it did."""

    section: str
    breakdown: ConfidenceBreakdown
    conflicting: bool = False
    missing_evidence: bool = False

    @property
    def score(self) -> float:
        return self.breakdown.score


@dataclass(slots=True)
class FallbackDecision:
    """Whether another round is worth running, and what it should chase."""

    should_retry: bool
    gaps: list[str] = field(default_factory=list)
    weak_sections: list[str] = field(default_factory=list)
    reason: str = ""


class ConfidenceScorer:
    """Scores a section from the sources cited for it."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def score_section(
        self,
        section: str,
        text: str,
        sources: list[SourceRecord],
    ) -> SectionAssessment:
        if not sources:
            return SectionAssessment(
                section=section,
                breakdown=ConfidenceBreakdown(
                    score=0.0, reasons=["no grounding sources for this section"]
                ),
                missing_evidence=True,
            )

        domains = {s.domain for s in sources}
        reasons: list[str] = []

        corroboration = self._corroboration(len(domains))
        if len(domains) == 1:
            reasons.append(f"single-domain support ({next(iter(domains))})")

        authority = sum(self._settings.authority_for(s.domain) for s in sources) / len(sources)
        if authority < 0.55:
            reasons.append("supporting domains are low-authority")

        recency = self._recency(sources)
        if recency < 0.4:
            reasons.append("supporting sources are stale")

        conflicting = self._has_numeric_conflict(sources)
        if conflicting:
            reasons.append("sources disagree on a numeric value")

        ambiguity = self._ambiguity(text, sources)
        if ambiguity > 0.1:
            reasons.append("hedged or weakly-supported wording")

        score = (
            W_CORROBORATION * corroboration
            + W_AUTHORITY * authority
            + W_RECENCY * recency
            - ambiguity
            - (0.15 if conflicting else 0.0)
        )
        score = max(0.0, min(1.0, score))

        return SectionAssessment(
            section=section,
            breakdown=ConfidenceBreakdown(
                corroboration=round(corroboration, 3),
                recency=round(recency, 3),
                authority=round(authority, 3),
                ambiguity_penalty=round(ambiguity, 3),
                score=round(score, 3),
                supporting_domains=len(domains),
                reasons=reasons,
            ),
            conflicting=conflicting,
        )

    # ------------------------------------------------------------------ signals

    @staticmethod
    def _corroboration(domain_count: int) -> float:
        """Saturating curve: the second independent domain matters most, the fifth least."""
        if domain_count <= 0:
            return 0.0
        return min(1.0, math.log1p(domain_count) / math.log(5))

    def _recency(self, sources: list[SourceRecord]) -> float:
        """Exponential decay on published date; undated sources score neutral, not zero."""
        now = utcnow()
        half_life = self._settings.recency_half_life_days
        scores = []
        for source in sources:
            if source.published_at is None:
                scores.append(0.6)
                continue
            published = source.published_at
            if published.tzinfo is None:
                published = published.replace(tzinfo=now.tzinfo)
            age_days = max(0.0, (now - published).total_seconds() / 86400)
            scores.append(0.5 ** (age_days / half_life))
        return sum(scores) / len(scores)

    @staticmethod
    def _ambiguity(text: str, sources: list[SourceRecord]) -> float:
        """Penalty for hedged wording, and for text that its own sources barely support."""
        penalty = 0.0
        if _HEDGE_PATTERNS.search(text):
            penalty += 0.08
        claim_tokens = tokenize(text)
        if claim_tokens:
            support = tokenize(" ".join(s.passage for s in sources))
            overlap = len(claim_tokens & support) / len(claim_tokens)
            if overlap < 0.35:
                penalty += 0.12
        return penalty

    @staticmethod
    def _extract_figures(passage: str) -> dict[str, set[float]]:
        """Figures in the passage, bucketed by family so unlike quantities never clash."""
        figures: dict[str, set[float]] = {"money": set(), "percent": set(), "magnitude": set()}
        for amount, unit in _MONEY.findall(passage):
            value = float(amount.replace(",", "")) * _SCALE.get((unit or "").lower(), 1.0)
            figures["money"].add(value)
        for amount in _PERCENT.findall(passage):
            figures["percent"].add(float(amount))
        for amount in _MAGNITUDE.findall(passage):
            figures["magnitude"].add(float(amount.replace(",", "")))
        return figures

    @classmethod
    def _has_numeric_conflict(cls, sources: list[SourceRecord]) -> bool:
        """True when two or more domains quote materially different figures of one kind."""
        by_family: dict[str, dict[str, set[float]]] = {}
        for source in sources:
            for family, values in cls._extract_figures(source.passage).items():
                if values:
                    by_family.setdefault(family, {}).setdefault(source.domain, set()).update(values)

        for by_domain in by_family.values():
            if len(by_domain) < 2:
                continue
            all_values = sorted({v for values in by_domain.values() for v in values})
            low, high = all_values[0], all_values[-1]
            if low > 0 and (high - low) / low > CONFLICT_TOLERANCE:
                return True
        return False


class FallbackController:
    """Decides on the extra round, then applies the final status to each section."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def decide(
        self,
        assessments: dict[str, SectionAssessment],
        store: EvidenceStore,
        *,
        rounds_used: int,
        seconds_left: float,
    ) -> FallbackDecision:
        """Trigger a fallback round only when it is both useful and affordable."""
        if not self._settings.fallback_enabled:
            return FallbackDecision(False, reason="fallback disabled")
        if rounds_used >= self._settings.max_fallback_rounds:
            return FallbackDecision(False, reason="fallback round cap reached")
        if seconds_left < self._settings.fallback_budget_seconds:
            return FallbackDecision(
                False, reason=f"only {seconds_left:.1f}s left, below fallback budget"
            )

        weak = [
            name
            for name, assessment in assessments.items()
            if assessment.score < self._settings.confidence_threshold
        ]
        domain_shortfall = len(store.distinct_domains()) < self._settings.min_distinct_domains

        if not weak and not domain_shortfall:
            return FallbackDecision(False, reason="all sections cleared the threshold")

        gaps: list[str] = []
        if domain_shortfall:
            gaps.append(
                f"only {len(store.distinct_domains())} distinct domains, "
                f"need {self._settings.min_distinct_domains}"
            )
        for name in weak:
            assessment = assessments[name]
            detail = "; ".join(assessment.breakdown.reasons) or "confidence below threshold"
            gaps.append(f"{name}: {detail}")

        return FallbackDecision(
            should_retry=True,
            gaps=gaps,
            weak_sections=weak,
            reason=f"{len(weak)} weak section(s), domain shortfall={domain_shortfall}",
        )

    def apply(
        self,
        section_name: str,
        section: BriefSection,
        assessment: SectionAssessment,
    ) -> tuple[BriefSection, str | None]:
        """Stamp the section with its final confidence and status.

        Returns the updated section plus an `unverified_flags` entry when the claim could
        not be asserted, so the caller can surface it in the output rather than drop it.
        """
        section.confidence = assessment.score
        section.breakdown = assessment.breakdown

        if not section.citations or assessment.missing_evidence:
            section.status = "insufficient_data"
            if not section.text.strip():
                return section, None
            flag = f"{section_name}: insufficient data - no source supports this section."
            section.text = ""
            return section, flag

        if assessment.conflicting:
            section.status = "conflicting"
            return section, (
                f"{section_name}: conflicting sources - "
                "figures disagree across domains, treat as unresolved."
            )

        if assessment.score < self._settings.confidence_threshold:
            section.status = "unverified"
            detail = "; ".join(assessment.breakdown.reasons) or "low corroboration"
            return section, (
                f"{section_name}: unverified (confidence {assessment.score:.2f}) - {detail}."
            )

        section.status = "grounded"
        return section, None
