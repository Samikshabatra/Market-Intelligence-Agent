# Performance: hitting the 60-second target

The spec allocates a per-stage budget (section 6). This document records how the
implementation spends it, what was done to fit, and how to measure it.

## Budget

| Stage | Budget | How it is enforced |
|---|---|---|
| Query planning | 5s | `Planner.plan()` is given a timeout sliced from the run deadline; on timeout or model failure the heuristic plan is used instead of failing. |
| Search + fetch | 25s | Every sub-question runs concurrently under a semaphore; the round as a whole is wrapped in `asyncio.wait_for`, and unfinished tasks are cancelled rather than awaited. |
| Grounding + scoring | 10s | Pure Python over in-memory passages - no network, no embedding model. |
| Fallback round | 15s, max 1 | Skipped entirely if less than the fallback budget remains. |
| Synthesis | 5s | Given a sliced timeout; effort degrades if the run is already behind. |
| **Total** | **60s** | One monotonic deadline set at `run()`; every stage takes `min(stage_budget, time_left)`. |

## What was done to fit

**Search runs in parallel, not in sequence.** Sub-questions are independent, so they
are issued concurrently. Sequential execution would cost roughly `n x per_request_timeout`
and blow the 25s search budget at n=6.

**Planning time is overlapped with retrieval.** Planning is a model call that retrieves
nothing, so up to 5s of the budget would otherwise be dead air. A two-step seed plan -
the raw query plus a broad overview query - is dispatched before the planner is awaited.
Anything the real plan also asks for is deduplicated for free by the executor, so the
overlap costs nothing and typically returns the run's first 8-10 sources.

**One search call covers search and fetch.** Tavily returns extracted page content, so
there is no second round-trip per URL. A snippet-only provider would need a fetch-and-
extract step per result, which is where a naive implementation loses its budget.

**Connections are pooled.** The Tavily provider holds one `httpx.AsyncClient` for the
process, so a run pays the TLS handshake once rather than per sub-question.

**Fallback is bounded twice.** At most one extra round, and only when the remaining
wall-clock time still covers the fallback budget. A late fallback is worse than no
fallback: it converts a slightly weak brief into a missed deadline.

**Synthesis degrades instead of overrunning.** If the earlier stages have eaten the
budget, synthesis steps its effort down one rung. A terser brief inside 60s beats a
better one that misses the target. The degradation is reset per query so it never leaks
into the next run.

**Thin and duplicate results are dropped before they cost anything.** Denylisting,
URL canonicalisation and passage fingerprinting happen in the executor, so low-value
sources never reach the scorer or the synthesis prompt - which also keeps the prompt
small, and the prompt is what synthesis latency scales with.

## Measuring it

The harness records per-stage timings for every query and reports the p90 against each
stage budget:

```bash
mia eval --ablation                                       # live providers
MIA_MOCK_CORPUS=eval/fixtures/offline_corpus.json \
  mia eval --offline --ablation                           # offline, no API spend
```

Output ends with:

```
Stage p90 vs. section 6 budget:
  ok  planning     0.00s /   5.0s
  ok  search       0.00s /  25.0s
  ...
```

A stage over its budget is a suite failure, not a warning, so a latency regression fails
the run the same way a groundedness regression does.

Offline numbers are near zero by construction - the mock provider does no I/O. They
verify that the budget plumbing works, not that the system is fast. **Latency claims
must come from a live run against Tavily and the Anthropic API.**

## Known headroom

- The two model calls (plan, synthesise) are sequential by necessity: synthesis needs the
  evidence that planning leads to. The seed search already hides most of the planning
  cost, but the synthesis call remains the single largest fixed cost in a live run.
- `results_per_subquestion` and `max_sub_questions` trade source diversity against
  search time directly; both are settings, so they can be tuned per deployment against
  the eval set rather than guessed.
- The evidence store's lexical search is O(sources x tokens) per section. That is
  negligible at 24 sources; it would need an index long before it needed a GPU.
