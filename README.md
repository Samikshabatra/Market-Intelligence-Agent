# Market Intelligence Agent

An autonomous AI research agent that plans its own multi-step search strategy, pulls from
5+ web sources per query, and returns a **structured, source-grounded competitor brief** in
under 60 seconds.

Full requirements live in [`market-intelligence-agent-spec.md`](market-intelligence-agent-spec.md).

## Pipeline

```
User query
   -> Query Planner        decomposes into adaptive sub-questions
   -> Search Executor      parallel web search + fetch, dedupe, denylist
   -> Evidence Store       every claim bound to (source_url, passage, retrieved_at)
   -> Confidence Scorer    corroboration / recency / authority / ambiguity
   -> Fallback Controller  one bounded extra round, then explicit "unverified" flags
   -> Brief Synthesizer    structured JSON brief + Markdown render
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
cp .env.example .env            # then fill in the two API keys
```

## Run

```bash
mia run "How does Notion position against Coda in the mid-market?"
mia run "Figma pricing changes 2026" --json out/figma.json
```

### Running without an Anthropic key

Both API keys give you the full pipeline. With **only** a Tavily key, the agent still
runs end to end - it degrades rather than failing:

```bash
mia run "How does Notion position against Coda?" --no-llm    # real search, no model
mia run "..." --offline                                      # mock search, no model
```

| | With both keys | Tavily only (`--no-llm`) |
|---|---|---|
| Planning | Claude decides the sub-questions, adapting depth to the query | Fixed template, one sub-question per source kind |
| Search | Live Tavily | Live Tavily |
| Grounding, dedupe, confidence, fallback | identical | identical |
| Brief | Claude writes each section from the evidence | Best-matching passages quoted verbatim, cited |

The citation rules are the same in both modes: a section with no valid citation is
never asserted. The model-free brief is honest but blunt - it quotes rather than
synthesises, and cannot reconcile two sources into one sentence. Every such run carries
an explicit flag saying so, so an extractive brief is never mistaken for a synthesised
one.

## Web interface

```bash
pip install -e ".[web]"
mia-web                       # http://127.0.0.1:8000
```

Ask a question, watch the pipeline run stage by stage over a live event stream, and read
the brief with its evidence. Four views on a result: **Executive summary** (the seven
sections with confidence and status), **Evidence** (per-claim confidence breakdown and
the passages behind it), **Sources** (every retrieved page), **Search plan** (the steps
the agent chose and where the time went).

The progress screen is not a spinner - each stage reports when the agent actually enters
and leaves it, with the counts it produced. Runs are written to `runs/` and listed under
Research history.

## Evaluate

```bash
mia eval --ablation                          # live: needs both API keys
MIA_MOCK_CORPUS=eval/fixtures/offline_corpus.json mia eval --offline --ablation
```

The 25-query set spans direct comparisons, pricing lookups, recency-sensitive news,
deliberately sparse queries and queries whose public sources are known to disagree.
The last two groups are marked `expect_flag: true`: the correct behaviour there is to
decline to assert, and the harness scores that as **flag recall**.

Metrics per run: groundedness, citation validity, source diversity, latency (mean and
p90), flag recall/precision and mean fallback rounds. Accuracy is human-reviewed - the
harness writes a `rubric.csv` skeleton rather than guessing a score. `--ablation` runs
the whole set twice, fallback on and off, and writes `ablation.md`.

Offline runs use a fixture corpus so sparse and conflicting behaviour can be exercised
without network access. Because that corpus is static, the ablation deltas are near
zero offline - it verifies wiring, not reliability. Run it against Tavily for a real
uplift number.

## Layout

| Path | Purpose |
|---|---|
| `src/market_intelligence_agent/planner.py` | adaptive query planning + re-planning |
| `src/market_intelligence_agent/executor.py` | parallel search execution, dedupe, source hygiene |
| `src/market_intelligence_agent/evidence.py` | grounding store, citation resolution |
| `src/market_intelligence_agent/confidence.py` | heuristic scoring + fallback control |
| `src/market_intelligence_agent/synthesizer.py` | structured brief synthesis with citation enforcement |
| `src/market_intelligence_agent/agent.py` | orchestration under the 60s budget |
| `eval/` | 25-query evaluation set and metrics harness |

## Status

All six milestones from the spec are implemented; 68 tests pass.
Milestone detail is in [`docs/MILESTONES.md`](docs/MILESTONES.md) and the latency
work in [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

Reported numbers so far come from the offline fixture corpus, which performs no I/O:
groundedness 1.0, citation validity 1.0, domain floor 1.0 on answerable queries,
flag recall 0.8. Those verify behaviour, not speed or real-world accuracy - live
latency and the fallback uplift need a run against Tavily and the Anthropic API.

## License

MIT
