"""Orchestration: plan -> search -> ground -> score -> (fallback) -> synthesise.

The whole run is governed by one monotonic deadline. Each stage gets the smaller of its
own budget and the time actually left, so a slow search cannot starve synthesis - the
brief is always produced, even if it is produced from thinner evidence and says so.
"""

from __future__ import annotations

import asyncio
import logging
import time

from .confidence import ConfidenceScorer, FallbackController, SectionAssessment
from .config import Settings
from .evidence import EvidenceStore
from .executor import ExecutionReport, SearchExecutor
from .llm import LLMClient
from .models import SECTION_NAMES, AgentResult, SearchPlan, StageTimings
from .planner import Planner
from .search import SearchProvider, build_provider
from .synthesizer import BriefSynthesizer

logger = logging.getLogger(__name__)


class MarketIntelligenceAgent:
    """One instance can serve many queries; per-run state is local to `run()`."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        provider: SearchProvider | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self._provider = provider or build_provider(self.settings)
        self._owns_provider = provider is None
        self._llm = llm or LLMClient(self.settings)
        self._planner = Planner(self._llm, self.settings)
        self._scorer = ConfidenceScorer(self.settings)
        self._controller = FallbackController(self.settings)
        self._synthesizer = BriefSynthesizer(self._llm, self.settings)

    async def run(self, query: str) -> AgentResult:
        started = time.perf_counter()
        deadline = time.monotonic() + self.settings.total_budget_seconds
        timings = StageTimings()
        store = EvidenceStore()
        executor = SearchExecutor(self._provider, self.settings)
        result = AgentResult(query=query)

        # --- 1. plan ------------------------------------------------------
        stage = time.perf_counter()
        plan = await self._planner.plan(
            query, timeout=self._slice(deadline, self.settings.planning_budget_seconds)
        )
        timings.planning_ms = (time.perf_counter() - stage) * 1000
        trace = list(plan.trace())

        # --- 2. search + 3. ground ---------------------------------------
        stage = time.perf_counter()
        report = await executor.run(plan, deadline=deadline, round_index=0)
        timings.search_ms = (time.perf_counter() - stage) * 1000
        self._ingest(store, plan, report)

        # --- 4. score -----------------------------------------------------
        stage = time.perf_counter()
        assessments = self._score_all(store)
        timings.grounding_ms = (time.perf_counter() - stage) * 1000

        # --- 4b. bounded fallback round -----------------------------------
        decision = self._controller.decide(
            assessments,
            store,
            rounds_used=0,
            seconds_left=self._seconds_left(deadline),
        )
        if decision.should_retry:
            stage = time.perf_counter()
            try:
                fallback_plan = await self._planner.replan(
                    query,
                    previous=plan,
                    gaps=decision.gaps,
                    covered_kinds=set(store.distinct_kinds()),
                    timeout=self._slice(deadline, self.settings.planning_budget_seconds),
                )
                fallback_report = await executor.run(
                    fallback_plan, deadline=deadline, round_index=1
                )
                self._ingest(store, fallback_plan, fallback_report)
                trace.extend(fallback_plan.trace())
                result.fallback_rounds = 1
                assessments = self._score_all(store)
            except asyncio.CancelledError:  # pragma: no cover - cooperative shutdown
                raise
            except Exception as exc:
                logger.warning("fallback round failed: %s", exc)
                result.unverified_flags.append(f"fallback round failed: {exc}")
            timings.fallback_ms = (time.perf_counter() - stage) * 1000
        else:
            logger.debug("no fallback round: %s", decision.reason)

        # --- 5. synthesise ------------------------------------------------
        stage = time.perf_counter()
        brief, conflicts = await self._synthesizer.synthesize(
            query, store, timeout=self._slice(deadline, self.settings.synthesis_budget_seconds)
        )
        timings.synthesis_ms = (time.perf_counter() - stage) * 1000

        # --- 6. final scoring, status stamping and flags -------------------
        flags: list[str] = list(result.unverified_flags)
        for name in SECTION_NAMES:
            section = getattr(brief, name)
            sources = store.resolve(section.citations)
            assessment = self._scorer.score_section(name, section.text, sources)
            section, flag = self._controller.apply(name, section, assessment)
            setattr(brief, name, section)
            if flag:
                flags.append(flag)

        for conflict in conflicts:
            flags.append(f"synthesiser noted a conflict: {conflict}")

        domains = store.distinct_domains()
        if len(domains) < self.settings.min_distinct_domains:
            flags.append(
                f"source diversity below target: {len(domains)} distinct domains, "
                f"need {self.settings.min_distinct_domains}."
            )
        for error in report.errors:
            flags.append(f"search issue: {error}")

        result.brief = brief
        result.sources_used = sorted(store.all(), key=lambda r: r.source_id)
        result.unverified_flags = flags
        result.search_plan_trace = trace
        result.timings = timings
        result.latency_ms = (time.perf_counter() - started) * 1000
        result.budget_exceeded = result.latency_ms > self.settings.total_budget_seconds * 1000
        return result

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _seconds_left(deadline: float) -> float:
        return max(0.0, deadline - time.monotonic())

    def _slice(self, deadline: float, stage_budget: float) -> float:
        """Never let a stage wait longer than the run has left."""
        return max(1.0, min(stage_budget, self._seconds_left(deadline)))

    @staticmethod
    def _ingest(store: EvidenceStore, plan: SearchPlan, report: ExecutionReport) -> None:
        """Move retrieved sources into the evidence store, carrying section hints."""
        hints = {sq.id: list(sq.target_sections) for sq in plan.sub_questions}
        store.add_all(report.records, hints)

    def _score_all(self, store: EvidenceStore) -> dict[str, SectionAssessment]:
        """Pre-synthesis scoring over the evidence aimed at each section.

        Runs before the brief exists so the fallback decision is based on evidence
        coverage rather than on wording the model has not written yet.
        """
        assessments: dict[str, SectionAssessment] = {}
        for name in SECTION_NAMES:
            sources = store.for_section(name, limit=8)
            probe = " ".join(s.passage[:200] for s in sources)
            assessments[name] = self._scorer.score_section(name, probe, sources)
        return assessments

    async def aclose(self) -> None:
        if self._owns_provider:
            await self._provider.aclose()
        await self._llm.aclose()

    async def __aenter__(self) -> "MarketIntelligenceAgent":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
