"""Pydantic models for the agent's plan, evidence and output brief.

`AgentResult.to_spec_dict()` emits exactly the JSON schema in section 5 of the spec;
the richer internal fields (per-claim scoring, stage timings) live alongside it so the
evaluation harness can inspect them without changing the public contract.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

SectionName = Literal[
    "company_overview",
    "positioning",
    "pricing",
    "recent_moves",
    "strengths",
    "weaknesses",
    "market_signals",
]

SECTION_NAMES: tuple[SectionName, ...] = (
    "company_overview",
    "positioning",
    "pricing",
    "recent_moves",
    "strengths",
    "weaknesses",
    "market_signals",
)

SourceKind = Literal[
    "company_site",
    "news",
    "review_platform",
    "funding_database",
    "social_professional",
    "industry_report",
    "community",
    "other",
]


def utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------- planning


class SubQuestion(BaseModel):
    """One step of the agent's self-authored search strategy."""

    id: str = Field(description="Stable short identifier, e.g. 'q1'.")
    question: str = Field(description="Standalone question this step must answer.")
    search_query: str = Field(description="Literal query string to send to the search API.")
    source_kind: SourceKind = Field(
        default="other",
        description="Kind of source most likely to answer this sub-question.",
    )
    target_sections: list[SectionName] = Field(
        default_factory=list,
        description="Brief sections this sub-question feeds.",
    )
    rationale: str = Field(default="", description="Why this step is worth spending budget on.")


class SearchPlan(BaseModel):
    """The planner's output: an ordered, adaptive set of sub-questions."""

    query: str
    subject: str = Field(default="", description="Primary company or market the query is about.")
    complexity: Literal["simple", "moderate", "complex"] = "moderate"
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    round_index: int = Field(default=0, description="0 for the initial plan, 1+ for fallbacks.")

    @field_validator("sub_questions")
    @classmethod
    def _non_empty(cls, value: list[SubQuestion]) -> list[SubQuestion]:
        if not value:
            raise ValueError("A search plan needs at least one sub-question.")
        return value

    def trace(self) -> list[str]:
        return [sq.question for sq in self.sub_questions]


# --------------------------------------------------------------------------- evidence


class SourceRecord(BaseModel):
    """A retrieved source plus the passage that makes it usable as evidence."""

    source_id: str
    url: str
    domain: str
    title: str = ""
    passage: str = ""
    retrieved_at: datetime = Field(default_factory=utcnow)
    published_at: datetime | None = None
    source_kind: SourceKind = "other"
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    sub_question_id: str = ""
    round_index: int = 0

    def citation_tuple(self) -> tuple[str, str, str]:
        """The (source_url, passage, retrieved_at) grounding tuple from the spec."""
        return (self.url, self.passage, self.retrieved_at.isoformat())


# --------------------------------------------------------------------------- output


class ConfidenceBreakdown(BaseModel):
    """Per-signal contributions, kept so low scores can be explained and audited."""

    corroboration: float = 0.0
    recency: float = 0.0
    authority: float = 0.0
    ambiguity_penalty: float = 0.0
    score: float = 0.0
    supporting_domains: int = 0
    reasons: list[str] = Field(default_factory=list)


class BriefSection(BaseModel):
    """One section of the brief. Text is only asserted when at least one citation backs it."""

    text: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    citations: list[str] = Field(default_factory=list)
    status: Literal["grounded", "unverified", "conflicting", "insufficient_data"] = "grounded"
    breakdown: ConfidenceBreakdown | None = None

    def is_asserted(self) -> bool:
        return self.status == "grounded" and bool(self.text.strip())


class Brief(BaseModel):
    company_overview: BriefSection = Field(default_factory=BriefSection)
    positioning: BriefSection = Field(default_factory=BriefSection)
    pricing: BriefSection = Field(default_factory=BriefSection)
    recent_moves: BriefSection = Field(default_factory=BriefSection)
    strengths: BriefSection = Field(default_factory=BriefSection)
    weaknesses: BriefSection = Field(default_factory=BriefSection)
    market_signals: BriefSection = Field(default_factory=BriefSection)

    def sections(self) -> list[tuple[str, BriefSection]]:
        return [(name, getattr(self, name)) for name in SECTION_NAMES]


class StageTimings(BaseModel):
    """Wall-clock milliseconds per pipeline stage, for the section 6 budget check."""

    planning_ms: float = 0.0
    search_ms: float = 0.0
    grounding_ms: float = 0.0
    fallback_ms: float = 0.0
    synthesis_ms: float = 0.0

    def total_ms(self) -> float:
        return (
            self.planning_ms
            + self.search_ms
            + self.grounding_ms
            + self.fallback_ms
            + self.synthesis_ms
        )


class AgentResult(BaseModel):
    """Everything one run produced - the brief, its evidence and its execution trace."""

    query: str
    generated_at: datetime = Field(default_factory=utcnow)
    latency_ms: float = 0.0
    sources_used: list[SourceRecord] = Field(default_factory=list)
    brief: Brief = Field(default_factory=Brief)
    unverified_flags: list[str] = Field(default_factory=list)
    search_plan_trace: list[str] = Field(default_factory=list)
    fallback_rounds: int = 0
    timings: StageTimings = Field(default_factory=StageTimings)
    budget_exceeded: bool = False

    def distinct_domains(self) -> set[str]:
        return {s.domain for s in self.sources_used}

    def meets_source_floor(self, minimum: int) -> bool:
        return len(self.distinct_domains()) >= minimum

    def to_spec_dict(self) -> dict:
        """Serialise to the exact output shape defined in section 5 of the spec."""
        return {
            "query": self.query,
            "generated_at": self.generated_at.isoformat(),
            "latency_ms": round(self.latency_ms, 1),
            "sources_used": [
                {
                    "domain": s.domain,
                    "url": s.url,
                    "retrieved_at": s.retrieved_at.isoformat(),
                }
                for s in self.sources_used
            ],
            "brief": {
                name: {
                    "text": section.text,
                    "confidence": round(section.confidence, 3),
                    "citations": section.citations,
                }
                for name, section in self.brief.sections()
            },
            "unverified_flags": self.unverified_flags,
            "search_plan_trace": self.search_plan_trace,
        }
