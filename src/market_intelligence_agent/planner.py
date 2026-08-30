"""Query Planner - stage 1 of the pipeline.

The planner decides *what* to search for and in what order. Plan length adapts to query
complexity rather than filling a fixed template, and `replan()` lets the fallback
controller insert targeted sub-questions once the first round exposes gaps.

A heuristic planner mirrors the same interface so the pipeline runs offline (tests,
`--offline`, and eval runs that must not spend API budget).
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel, Field

from .config import Settings
from .llm import LLMClient, LLMError
from .models import SearchPlan, SubQuestion

logger = logging.getLogger(__name__)

PLANNER_SYSTEM = """You are the planning stage of a market-intelligence research agent.

Given one market or competitor question, design the search strategy an expert analyst
would run. You decide the steps - do not fill a fixed template.

Rules:
- Produce between {min_q} and {max_q} sub-questions. Simple lookups deserve fewer;
  multi-company comparisons deserve more.
- Each sub-question must be independently answerable from public web sources.
- Together the sub-questions must reach at least 5 DISTINCT kinds of source
  (company site, news, review platform, funding database, professional/hiring signal,
  industry report), because the brief needs corroboration across domains.
- `search_query` is the literal string sent to a web search API: keywords, company
  names, years - not a sentence. Include the company name in every query.
- `target_sections` maps each step to the brief sections it feeds.
- Do not plan steps that require paid databases, logins, or private data.
"""

REPLAN_SYSTEM = """You are the re-planning stage of a market-intelligence research agent.

The first search round left specific gaps. Produce ONLY new sub-questions that attack
those gaps from a different angle than the queries already tried: different phrasing,
different source kind, or a different time window.

Rules:
- Produce between 2 and 4 sub-questions. This is the last round; budget is tight.
- Never repeat a search query that was already run.
- Prefer source kinds that are missing from the evidence gathered so far.
"""


class _PlanDraft(BaseModel):
    """Model-facing schema. Kept separate from SearchPlan so the model never sets
    bookkeeping fields like `round_index`."""

    subject: str = Field(description="Primary company or market the query is about.")
    complexity: Literal["simple", "moderate", "complex"]
    sub_questions: list[SubQuestion]


class Planner:
    """LLM planner with a heuristic fallback when the model is unavailable."""

    def __init__(self, llm: LLMClient, settings: Settings) -> None:
        self._llm = llm
        self._settings = settings

    async def plan(self, query: str, *, timeout: float | None = None) -> SearchPlan:
        system = PLANNER_SYSTEM.format(
            min_q=self._settings.min_sub_questions,
            max_q=self._settings.max_sub_questions,
        )
        if not self._llm.available:
            return heuristic_plan(query, self._settings)
        try:
            draft = await self._llm.structured(
                _PlanDraft,
                system=system,
                user=f"Market intelligence query:\n{query}",
                effort=self._settings.planner_effort,
                max_tokens=4000,
                timeout=timeout,
            )
        except LLMError as exc:
            logger.warning("planner fell back to heuristics: %s", exc)
            return heuristic_plan(query, self._settings)

        return self._finalise(query, draft, round_index=0)

    async def replan(
        self,
        query: str,
        *,
        previous: SearchPlan,
        gaps: list[str],
        covered_kinds: set[str],
        timeout: float | None = None,
    ) -> SearchPlan:
        """Produce a bounded extra round aimed at the gaps the first round left."""
        tried = "\n".join(f"- {sq.search_query}" for sq in previous.sub_questions)
        missing = sorted(
            {"company_site", "news", "review_platform", "funding_database",
             "social_professional", "industry_report"} - covered_kinds
        )
        prompt = (
            f"Original query:\n{query}\n\n"
            f"Subject: {previous.subject or 'unknown'}\n\n"
            f"Queries already run:\n{tried or '- none'}\n\n"
            f"Gaps to close:\n" + "\n".join(f"- {g}" for g in gaps or ["insufficient corroboration"])
            + f"\n\nSource kinds still missing: {', '.join(missing) or 'none'}"
        )
        if not self._llm.available:
            return heuristic_replan(query, previous, missing, self._settings)
        try:
            draft = await self._llm.structured(
                _PlanDraft,
                system=REPLAN_SYSTEM,
                user=prompt,
                effort="low",
                max_tokens=2000,
                timeout=timeout,
            )
        except LLMError as exc:
            logger.warning("replanner fell back to heuristics: %s", exc)
            return heuristic_replan(query, previous, missing, self._settings)

        plan = self._finalise(query, draft, round_index=previous.round_index + 1)
        plan.subject = plan.subject or previous.subject
        return self._drop_repeats(plan, previous)

    # ------------------------------------------------------------------ internals

    def _finalise(self, query: str, draft: _PlanDraft, *, round_index: int) -> SearchPlan:
        """Clamp plan size and re-issue stable ids so downstream joins are reliable."""
        limit = self._settings.max_sub_questions
        questions = draft.sub_questions[:limit]
        for index, sub_question in enumerate(questions, start=1):
            sub_question.id = f"r{round_index}q{index}"
        if not questions:
            return heuristic_plan(query, self._settings)
        return SearchPlan(
            query=query,
            subject=draft.subject,
            complexity=draft.complexity,
            sub_questions=questions,
            round_index=round_index,
        )

    @staticmethod
    def _drop_repeats(plan: SearchPlan, previous: SearchPlan) -> SearchPlan:
        seen = {sq.search_query.strip().lower() for sq in previous.sub_questions}
        fresh = [sq for sq in plan.sub_questions if sq.search_query.strip().lower() not in seen]
        if fresh:
            plan.sub_questions = fresh
        return plan


# --------------------------------------------------------------------- heuristics

_QUERY_NOISE = re.compile(
    r"\b(how|does|do|is|are|the|a|an|of|in|for|to|what|with|compare|compared|vs|versus|"
    r"against|their|its|and|or|our|us|we)\b",
    re.IGNORECASE,
)

# One template per source kind, so a heuristic plan still clears the 5-domain floor.
_HEURISTIC_TEMPLATES: tuple[tuple[str, str, str, list[str]], ...] = (
    ("company_site", "{subject} official product and pricing page",
     "What does the company itself claim about its product and pricing?",
     ["company_overview", "pricing", "positioning"]),
    ("news", "{subject} news funding launch 2026",
     "What has recently been reported about the company?",
     ["recent_moves", "market_signals"]),
    ("review_platform", "{subject} G2 Capterra reviews pros cons",
     "What do verified customers say are its strengths and weaknesses?",
     ["strengths", "weaknesses"]),
    ("funding_database", "{subject} Crunchbase funding investors valuation",
     "What is the company's funding and investor profile?",
     ["company_overview", "market_signals"]),
    ("social_professional", "{subject} LinkedIn headcount hiring roles",
     "What do hiring and headcount signals say about its direction?",
     ["market_signals", "recent_moves"]),
    ("industry_report", "{subject} market share analyst report category",
     "How do analysts position it within its category?",
     ["positioning", "market_signals"]),
)


def extract_subject(query: str) -> str:
    """Cheap subject guess: the capitalised proper nouns, else the leading keywords."""
    proper = re.findall(r"\b[A-Z][A-Za-z0-9&.\-]{2,}\b", query)
    ignored = {"How", "What", "Why", "When", "Compare", "Does", "Who", "Which"}
    proper = [p for p in proper if p not in ignored]
    if proper:
        return " ".join(proper[:2])
    # No proper nouns (e.g. "revenue of a typical bootstrapped devtools company"): keep
    # the longest content words, which carry more search signal than the leading ones,
    # but return them in their original order so the query still reads naturally.
    cleaned = _QUERY_NOISE.sub(" ", query)
    words = [w.strip("?.,") for w in cleaned.split() if len(w) > 3]
    if not words:
        return query[:40]
    distinctive = sorted(words, key=len, reverse=True)[:2]
    return " ".join(w for w in words if w in distinctive)


def heuristic_plan(query: str, settings: Settings) -> SearchPlan:
    """Deterministic plan used offline or when the model call fails."""
    subject = extract_subject(query)
    complexity = "complex" if len(query.split()) > 18 else "moderate"
    questions = [
        SubQuestion(
            id=f"r0q{index}",
            question=question,
            search_query=template.format(subject=subject),
            source_kind=kind,  # type: ignore[arg-type]
            target_sections=sections,  # type: ignore[arg-type]
            rationale="Heuristic plan: guarantees coverage of this source kind.",
        )
        for index, (kind, template, question, sections) in enumerate(
            _HEURISTIC_TEMPLATES[: settings.max_sub_questions], start=1
        )
    ]
    return SearchPlan(
        query=query,
        subject=subject,
        complexity=complexity,
        sub_questions=questions,
        round_index=0,
    )


def heuristic_replan(
    query: str,
    previous: SearchPlan,
    missing_kinds: list[str],
    settings: Settings,
) -> SearchPlan:
    """Offline fallback round: re-target the source kinds that are still missing."""
    subject = previous.subject or extract_subject(query)
    by_kind = {kind: (template, question, sections)
               for kind, template, question, sections in _HEURISTIC_TEMPLATES}
    targets = missing_kinds or ["news", "review_platform"]
    questions: list[SubQuestion] = []
    for index, kind in enumerate(targets[:3], start=1):
        template, question, sections = by_kind.get(
            kind, ("{subject} independent analysis", f"Independent evidence about {subject}.",
                   ["positioning"])
        )
        questions.append(
            SubQuestion(
                id=f"r{previous.round_index + 1}q{index}",
                question=question,
                search_query=f"{template.format(subject=subject)} review",
                source_kind=kind,  # type: ignore[arg-type]
                target_sections=sections,  # type: ignore[arg-type]
                rationale=f"Fallback round: {kind} evidence was missing.",
            )
        )
    return SearchPlan(
        query=query,
        subject=subject,
        complexity=previous.complexity,
        sub_questions=questions,
        round_index=previous.round_index + 1,
    )
