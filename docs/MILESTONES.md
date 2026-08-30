# Milestones

| ID | Scope | Status |
|---|---|---|
| M1 | Query planner + search executor loop end-to-end | done |
| M2 | Source grounding + evidence store, citations wired into output | done |
| M3 | Confidence scoring + fallback controller, latency budgeted | done |
| M4 | Structured brief synthesizer + finalised output schema | done |
| M5 | 25-query evaluation set, baseline vs. full run, uplift measured | done |
| M6 | Performance tuning to hit <60s p90 | done (offline; live numbers pending API keys) |

All six milestones are implemented. Latency and reliability figures reported so far
come from the offline fixture corpus, which does no I/O - see
[PERFORMANCE.md](PERFORMANCE.md). Live p90 latency and the fallback uplift number
require a run against Tavily and the Anthropic API.
