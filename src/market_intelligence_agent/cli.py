"""Command line entry point: `mia run "<query>"` and `mia eval`."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .agent import MarketIntelligenceAgent
from .config import Settings
from .render import render_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mia",
        description="Autonomous market intelligence agent: query in, grounded brief out.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log pipeline decisions")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Research one query and print a competitor brief.")
    run.add_argument("query", help="Natural-language market or competitor question.")
    run.add_argument(
        "--provider",
        choices=("tavily", "mock"),
        help="Search backend. 'mock' runs fully offline.",
    )
    run.add_argument("--offline", action="store_true", help="Skip all model calls.")
    run.add_argument("--json", dest="json_path", help="Also write the spec JSON to this path.")
    run.add_argument("--budget", type=float, help="Total latency budget in seconds.")
    run.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable the fallback round (the ablation baseline).",
    )

    evaluate = sub.add_parser("eval", help="Run the evaluation set and report metrics.")
    evaluate.add_argument("--set", dest="query_set", default="eval/queries.yaml")
    evaluate.add_argument("--out", dest="out_dir", default="eval/runs")
    evaluate.add_argument("--provider", choices=("tavily", "mock"), default=None)
    evaluate.add_argument("--offline", action="store_true", help="Skip all model calls.")
    evaluate.add_argument(
        "--ablation",
        action="store_true",
        help="Run each query twice - with and without the fallback round - and compare.",
    )
    evaluate.add_argument("--limit", type=int, default=None, help="Only run the first N queries.")
    return parser


def _settings_for(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env(
        search_provider=getattr(args, "provider", None),
        total_budget_seconds=getattr(args, "budget", None),
    )
    if getattr(args, "no_fallback", False):
        settings.fallback_enabled = False
    if getattr(args, "offline", False):
        settings.anthropic_api_key = None
        settings.search_provider = getattr(args, "provider", None) or "mock"
    return settings


async def _run(args: argparse.Namespace) -> int:
    settings = _settings_for(args)
    async with MarketIntelligenceAgent(settings) as agent:
        result = await agent.run(args.query)

    print(render_markdown(result))

    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_spec_dict(), indent=2), encoding="utf-8")
        print(f"\nJSON written to {path}", file=sys.stderr)

    # Non-zero exit when the run missed a headline target, so CI can gate on it.
    if result.budget_exceeded or not result.meets_source_floor(settings.min_distinct_domains):
        return 1
    return 0


async def _eval(args: argparse.Namespace) -> int:
    from eval.harness import run_evaluation  # imported lazily; harness is dev-only

    settings = _settings_for(args)
    report = await run_evaluation(
        query_set=Path(args.query_set),
        out_dir=Path(args.out_dir),
        settings=settings,
        ablation=args.ablation,
        limit=args.limit,
    )
    return 0 if report.passed else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    handler = {"run": _run, "eval": _eval}[args.command]
    return asyncio.run(handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
