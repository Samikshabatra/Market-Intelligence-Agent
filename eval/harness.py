"""Evaluation harness (spec section 7).

Runs the 25-query set, logs the full evidence trace and brief per query, and computes
the metrics the spec names: groundedness, source diversity, latency, and fallback
behaviour. Accuracy is human-reviewed, so the harness emits a rubric sheet rather than
guessing a score.

The ablation mode runs each query twice - fallback on and fallback off - which is the
headline result the spec asks for: how much reliability the confidence-based fallback
actually buys.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import yaml

from market_intelligence_agent.agent import MarketIntelligenceAgent
from market_intelligence_agent.config import Settings
from market_intelligence_agent.models import SECTION_NAMES, AgentResult
from market_intelligence_agent.render import render_markdown, render_summary

# Pass thresholds for the suite as a whole (spec section 3).
TARGET_GROUNDEDNESS = 1.0          # every asserted section must carry a valid citation
TARGET_DOMAIN_FLOOR_RATE = 0.8     # >=80% of *answerable* queries reach the 5-domain floor
TARGET_P90_LATENCY_S = 60.0
TARGET_FLAG_RECALL = 0.8           # >=80% of expect_flag queries must decline to assert

# Per-stage budgets from section 6 of the spec, checked at p90.
STAGE_BUDGETS_S = {
    "planning": 5.0,
    "search": 25.0,
    "grounding": 10.0,
    "fallback": 15.0,
    "synthesis": 5.0,
}


@dataclass
class QuerySpec:
    id: str
    query: str
    category: str = "comparison"
    expect_flag: bool = False
    focus_sections: list[str] = field(default_factory=list)


@dataclass
class QueryMetrics:
    """Per-query metrics. `accuracy` stays None until a human fills in the rubric."""

    id: str
    category: str
    query: str
    latency_s: float
    distinct_domains: int
    total_sources: int
    sections_asserted: int
    groundedness: float
    citation_validity: float
    fallback_rounds: int
    flags: int
    declined_to_assert: bool
    expect_flag: bool
    correct_flag_behaviour: bool
    budget_exceeded: bool
    planning_s: float = 0.0
    search_s: float = 0.0
    grounding_s: float = 0.0
    fallback_s: float = 0.0
    synthesis_s: float = 0.0
    accuracy: float | None = None


@dataclass
class SuiteReport:
    started_at: str
    variant: str
    queries: int
    mean_latency_s: float
    p90_latency_s: float
    mean_distinct_domains: float
    domain_floor_rate: float
    mean_fallback_rounds: float
    groundedness: float
    citation_validity: float
    flag_recall: float
    flag_precision: float
    mean_sections_asserted: float
    stage_p90_s: dict[str, float]
    passed: bool
    failures: list[str] = field(default_factory=list)
    per_query: list[QueryMetrics] = field(default_factory=list)


def load_query_set(path: Path) -> list[QuerySpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [QuerySpec(**item) for item in raw]


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; stable for the small n the eval set has."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered)) - 1))
    return ordered[index]


def score_result(spec: QuerySpec, result: AgentResult) -> QueryMetrics:
    """Turn one run into metrics. Groundedness and citation validity are computed
    against the evidence actually stored, so a fabricated id would show up here."""
    valid_ids = {s.source_id for s in result.sources_used}

    asserted = 0
    grounded = 0
    cited_total = 0
    cited_valid = 0
    for name in SECTION_NAMES:
        section = getattr(result.brief, name)
        cited_total += len(section.citations)
        cited_valid += sum(1 for c in section.citations if c in valid_ids)
        if section.is_asserted():
            asserted += 1
            if section.citations and all(c in valid_ids for c in section.citations):
                grounded += 1

    declined = any(
        getattr(result.brief, name).status in {"unverified", "conflicting", "insufficient_data"}
        for name in (spec.focus_sections or list(SECTION_NAMES))
    )

    return QueryMetrics(
        id=spec.id,
        category=spec.category,
        query=spec.query,
        latency_s=result.latency_ms / 1000,
        distinct_domains=len(result.distinct_domains()),
        total_sources=len(result.sources_used),
        sections_asserted=asserted,
        groundedness=(grounded / asserted) if asserted else 1.0,
        citation_validity=(cited_valid / cited_total) if cited_total else 1.0,
        fallback_rounds=result.fallback_rounds,
        flags=len(result.unverified_flags),
        declined_to_assert=declined,
        expect_flag=spec.expect_flag,
        correct_flag_behaviour=(declined == spec.expect_flag) if spec.expect_flag else not declined,
        budget_exceeded=result.budget_exceeded,
        planning_s=result.timings.planning_ms / 1000,
        search_s=result.timings.search_ms / 1000,
        grounding_s=result.timings.grounding_ms / 1000,
        fallback_s=result.timings.fallback_ms / 1000,
        synthesis_s=result.timings.synthesis_ms / 1000,
    )


def summarise(variant: str, metrics: list[QueryMetrics], started_at: str) -> SuiteReport:
    latencies = [m.latency_s for m in metrics]

    # The 5-distinct-domain floor is only a fair target for queries the public web can
    # actually answer. Sparse and conflicting queries are in the set precisely because
    # the evidence is thin, so scoring them against the floor would punish the agent for
    # behaving correctly. They are still scored on flag recall below.
    answerable = [m for m in metrics if not m.expect_flag]
    floor_hits = sum(1 for m in answerable if m.distinct_domains >= 5)

    expected_flag = [m for m in metrics if m.expect_flag]
    flagged_correctly = sum(1 for m in expected_flag if m.declined_to_assert)
    unexpected_flags = [m for m in metrics if not m.expect_flag and m.declined_to_assert]

    flag_recall = flagged_correctly / len(expected_flag) if expected_flag else 1.0
    # Precision: of everything the agent declined to assert, how much should have been
    # declined. A low value means the agent is timid, not reliable.
    declined_total = flagged_correctly + len(unexpected_flags)
    flag_precision = flagged_correctly / declined_total if declined_total else 1.0

    report = SuiteReport(
        started_at=started_at,
        variant=variant,
        queries=len(metrics),
        mean_latency_s=round(statistics.fmean(latencies), 2) if latencies else 0.0,
        p90_latency_s=round(percentile(latencies, 0.9), 2),
        mean_distinct_domains=round(
            statistics.fmean([m.distinct_domains for m in metrics]), 2
        ) if metrics else 0.0,
        domain_floor_rate=round(floor_hits / len(answerable), 3) if answerable else 1.0,
        mean_fallback_rounds=round(
            statistics.fmean([m.fallback_rounds for m in metrics]), 2
        ) if metrics else 0.0,
        groundedness=round(statistics.fmean([m.groundedness for m in metrics]), 3)
        if metrics else 0.0,
        citation_validity=round(statistics.fmean([m.citation_validity for m in metrics]), 3)
        if metrics else 0.0,
        flag_recall=round(flag_recall, 3),
        flag_precision=round(flag_precision, 3),
        mean_sections_asserted=round(
            statistics.fmean([m.sections_asserted for m in metrics]), 2
        ) if metrics else 0.0,
        stage_p90_s={
            stage: round(percentile([getattr(m, f"{stage}_s") for m in metrics], 0.9), 2)
            for stage in ("planning", "search", "grounding", "fallback", "synthesis")
        },
        passed=True,
        per_query=metrics,
    )

    if report.citation_validity < TARGET_GROUNDEDNESS:
        report.failures.append(
            f"citation validity {report.citation_validity} < {TARGET_GROUNDEDNESS}"
        )
    if report.groundedness < TARGET_GROUNDEDNESS:
        report.failures.append(f"groundedness {report.groundedness} < {TARGET_GROUNDEDNESS}")
    if report.domain_floor_rate < TARGET_DOMAIN_FLOOR_RATE:
        report.failures.append(
            f"domain floor rate {report.domain_floor_rate} < {TARGET_DOMAIN_FLOOR_RATE}"
        )
    if report.p90_latency_s > TARGET_P90_LATENCY_S:
        report.failures.append(f"p90 latency {report.p90_latency_s}s > {TARGET_P90_LATENCY_S}s")
    if report.flag_recall < TARGET_FLAG_RECALL:
        report.failures.append(f"flag recall {report.flag_recall} < {TARGET_FLAG_RECALL}")
    for stage, budget in STAGE_BUDGETS_S.items():
        observed = report.stage_p90_s.get(stage, 0.0)
        if observed > budget:
            report.failures.append(f"{stage} p90 {observed}s > {budget}s budget")
    report.passed = not report.failures
    return report


def rubric_rows(metrics: list[QueryMetrics]) -> str:
    """CSV skeleton for the human accuracy review (spec section 7.3)."""
    header = "id,category,query,sections_asserted,groundedness,accuracy_0_to_1,reviewer_notes"
    rows = [
        f'{m.id},{m.category},"{m.query}",{m.sections_asserted},{m.groundedness:.2f},,'
        for m in metrics
    ]
    return "\n".join([header, *rows]) + "\n"


async def run_variant(
    specs: list[QuerySpec],
    settings: Settings,
    out_dir: Path,
    variant: str,
) -> SuiteReport:
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    variant_dir = out_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[QueryMetrics] = []
    async with MarketIntelligenceAgent(settings) as agent:
        for spec in specs:
            result = await agent.run(spec.query)
            metrics.append(score_result(spec, result))
            (variant_dir / f"{spec.id}.json").write_text(
                json.dumps(result.to_spec_dict(), indent=2), encoding="utf-8"
            )
            (variant_dir / f"{spec.id}.md").write_text(render_markdown(result), encoding="utf-8")
            print(f"  {spec.id} {render_summary(result)}")

    report = summarise(variant, metrics, started_at)
    (variant_dir / "report.json").write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )
    (variant_dir / "rubric.csv").write_text(rubric_rows(metrics), encoding="utf-8")
    return report


def render_comparison(full: SuiteReport, baseline: SuiteReport) -> str:
    """The headline ablation table: what the confidence-based fallback bought."""

    def delta(a: float, b: float) -> str:
        diff = a - b
        return f"{diff:+.3f}"

    lines = [
        "# Ablation: confidence-based fallback on vs. off",
        "",
        f"Queries: {full.queries}   Run: {full.started_at}",
        "",
        "| Metric | With fallback | Without fallback | Delta |",
        "|---|---|---|---|",
        f"| Groundedness | {full.groundedness} | {baseline.groundedness} | "
        f"{delta(full.groundedness, baseline.groundedness)} |",
        f"| Citation validity | {full.citation_validity} | {baseline.citation_validity} | "
        f"{delta(full.citation_validity, baseline.citation_validity)} |",
        f"| Mean distinct domains | {full.mean_distinct_domains} | "
        f"{baseline.mean_distinct_domains} | "
        f"{delta(full.mean_distinct_domains, baseline.mean_distinct_domains)} |",
        f"| Domain floor rate (answerable queries) | {full.domain_floor_rate} | "
        f"{baseline.domain_floor_rate} | "
        f"{delta(full.domain_floor_rate, baseline.domain_floor_rate)} |",
        f"| Mean fallback rounds | {full.mean_fallback_rounds} | "
        f"{baseline.mean_fallback_rounds} | "
        f"{delta(full.mean_fallback_rounds, baseline.mean_fallback_rounds)} |",
        f"| Flag recall (declined when it should) | {full.flag_recall} | "
        f"{baseline.flag_recall} | {delta(full.flag_recall, baseline.flag_recall)} |",
        f"| Flag precision | {full.flag_precision} | {baseline.flag_precision} | "
        f"{delta(full.flag_precision, baseline.flag_precision)} |",
        f"| Mean sections asserted | {full.mean_sections_asserted} | "
        f"{baseline.mean_sections_asserted} | "
        f"{delta(full.mean_sections_asserted, baseline.mean_sections_asserted)} |",
        f"| Mean latency (s) | {full.mean_latency_s} | {baseline.mean_latency_s} | "
        f"{delta(full.mean_latency_s, baseline.mean_latency_s)} |",
        f"| p90 latency (s) | {full.p90_latency_s} | {baseline.p90_latency_s} | "
        f"{delta(full.p90_latency_s, baseline.p90_latency_s)} |",
        "",
        "Reliability uplift is the flag-recall and groundedness delta at an acceptable",
        "latency cost. Accuracy is scored separately by a human against `rubric.csv`.",
        "",
        "Note: against the offline fixture corpus the deltas are expected to be near zero.",
        "The corpus is static, so a second search round retrieves the same documents and",
        "the fallback has nothing new to find. Run the ablation against a live provider",
        "for a meaningful uplift number; offline it verifies wiring, not reliability.",
        "",
    ]
    return "\n".join(lines)


async def run_evaluation(
    *,
    query_set: Path,
    out_dir: Path,
    settings: Settings,
    ablation: bool = False,
    limit: int | None = None,
) -> SuiteReport:
    specs = load_query_set(query_set)
    if limit:
        specs = specs[:limit]

    started = time.perf_counter()
    print(f"Running {len(specs)} queries (provider={settings.search_provider}) ...")
    full = await run_variant(specs, settings, out_dir, "with_fallback")

    if ablation:
        baseline_settings = replace(settings, fallback_enabled=False)
        print("Running ablation baseline (fallback disabled) ...")
        baseline = await run_variant(specs, baseline_settings, out_dir, "no_fallback")
        comparison = render_comparison(full, baseline)
        (out_dir / "ablation.md").write_text(comparison, encoding="utf-8")
        print("\n" + comparison)

    print(
        f"\n{full.variant}: groundedness={full.groundedness} "
        f"citation_validity={full.citation_validity} "
        f"p90_latency={full.p90_latency_s}s "
        f"domains(mean)={full.mean_distinct_domains} "
        f"flag_recall={full.flag_recall}"
    )
    print("\nStage p90 vs. section 6 budget:")
    for stage, budget in STAGE_BUDGETS_S.items():
        observed = full.stage_p90_s.get(stage, 0.0)
        mark = "ok " if observed <= budget else "OVER"
        print(f"  {mark} {stage:<10} {observed:6.2f}s / {budget:>5.1f}s")
    print(
        f"\nSuite {'PASSED' if full.passed else 'FAILED'} "
        f"in {time.perf_counter() - started:.1f}s"
    )
    for failure in full.failures:
        print(f"  - {failure}")
    print(f"Artifacts in {out_dir.resolve()}")
    return full


def main() -> int:  # pragma: no cover - convenience entry point
    import argparse

    parser = argparse.ArgumentParser(description="Run the Market Intelligence Agent eval set.")
    parser.add_argument("--set", dest="query_set", default="eval/queries.yaml")
    parser.add_argument("--out", dest="out_dir", default="eval/runs")
    parser.add_argument("--provider", choices=("tavily", "mock"), default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    settings = Settings.from_env(search_provider=args.provider)
    if args.offline:
        settings.anthropic_api_key = None
        settings.search_provider = args.provider or "mock"

    report = asyncio.run(
        run_evaluation(
            query_set=Path(args.query_set),
            out_dir=Path(args.out_dir),
            settings=settings,
            ablation=args.ablation,
            limit=args.limit,
        )
    )
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
