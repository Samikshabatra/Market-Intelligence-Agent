# Market Intelligence Agent — Project Spec

## 1. Overview

**Project name:** Market Intelligence Agent

**One-line summary:** An autonomous AI research agent that plans its own multi-step search strategy, pulls from 5+ web sources per query, and outputs a structured competitor brief in under 60 seconds.

**Problem statement:** Manual competitor research is slow and inconsistent — analysts spend hours cross-referencing company sites, news, funding databases, and review platforms to answer a single question like "How does Competitor X position against us in the mid-market segment?" This project automates that workflow with an agent that plans, searches, grounds, and synthesizes autonomously.

**Primary goal:** Given a natural-language market/competitor query, the agent should independently decide what to search for, gather sufficient corroborating evidence, and return a reliable, structured brief — without a human specifying the search steps.

---

## 2. Goals & Non-Goals

### Goals
- Autonomous multi-step query planning (the agent decides *what* to search and *in what order*, not just executes a fixed pipeline).
- Aggregate data from **5+ distinct web sources** per query.
- Produce a **structured competitor brief** (not raw text dump) in **under 60 seconds** end-to-end.
- Ground every factual claim in a retrieved source (source grounding).
- Detect low-confidence answers and trigger **fallback strategies** (broaden search, try alternate sources, flag uncertainty) rather than hallucinate.
- Validate reliability against a **25-query evaluation set** with defined pass/fail criteria.

### Non-Goals (v1)
- Not a general-purpose web research agent — scoped to market/competitor intelligence queries.
- No persistent multi-turn negotiation with the user mid-research (single-shot query → brief).
- No paid/proprietary data source integrations in v1 (e.g., Crunchbase API, PitchBook) — public web sources only.
- No automated report scheduling/monitoring (e.g., "alert me if competitor changes pricing") — out of scope for v1.

---

## 3. Success Metrics

| Metric | Target |
|---|---|
| Sources aggregated per query | ≥ 5 distinct domains |
| End-to-end latency | < 60 seconds (p90) |
| Answer reliability on 25-query eval set | Defined accuracy/groundedness threshold (see §8) |
| Claims with valid source citation | 100% of factual claims traceable to a retrieved source |
| Fallback trigger precision | Low-confidence flag correlates with actual errors (validated via eval set) |

---

## 4. System Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  1. Query Planner    │  → decomposes query into sub-questions,
│  (Planning Agent)    │     drafts a multi-step search strategy
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  2. Search Executor  │  → runs web searches per sub-question,
│  (Tool-use loop)     │     fetches pages, dedupes sources
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  3. Source Grounding │  → maps each extracted claim to a
│  & Evidence Store    │     specific source + passage
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  4. Confidence       │  → scores answer/claim confidence;
│  Scorer & Fallback   │     triggers additional search rounds
│  Controller          │     if confidence < threshold
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  5. Brief Synthesizer│  → compiles structured competitor
│                       │     brief from grounded evidence
└─────────┬────────────┘
          ▼
   Structured Output (JSON + rendered brief)
```

### 4.1 Query Planner
- Input: raw user query (e.g., "Compare Competitor X's pricing and positioning vs us in enterprise SaaS").
- Output: an ordered list of sub-questions / search steps (e.g., pricing page lookup, recent funding news, review-site sentiment, product changelog, headcount/hiring signals).
- Should adapt plan length/depth to query complexity — not a fixed template.
- Re-planning: if early search rounds surface unexpected findings, the planner may insert new sub-questions.

### 4.2 Search Executor
- Executes web search + page fetch tools in a loop.
- Must hit **≥ 5 distinct domains** per query (e.g., company site, news outlet, review platform like G2/Capterra, funding database, social/LinkedIn signal, industry report).
- Deduplicates redundant sources; discards low-quality/unreliable domains (denylist).
- Tracks per-source metadata: URL, retrieval timestamp, extracted passage.

### 4.3 Source Grounding & Evidence Store
- Every factual claim in the final brief must be linked to a specific `(source_url, passage, retrieved_at)` tuple.
- Evidence store is queryable — supports the confidence scorer and enables citation rendering.
- No claim is allowed into the synthesizer without at least one grounding tuple.

### 4.4 Confidence Scorer & Fallback Controller
- Scores each synthesized claim/section (e.g., 0–1 confidence) based on: source agreement/corroboration, source recency, source authority, extraction ambiguity.
- Fallback strategies when confidence < threshold:
  1. Expand search (new query variants / additional domains).
  2. Try an alternate source type (e.g., if company site is stale, check recent news).
  3. If still low confidence after N fallback rounds → explicitly flag the claim as "unverified" or "conflicting sources" in the output rather than asserting it.
- Fallback rounds are bounded to respect the 60-second budget (see §6).

### 4.5 Brief Synthesizer
- Compiles evidence into the structured output format (§5).
- Enforces citation-per-claim.
- Surfaces confidence flags and any "unverified" sections clearly.

---

## 5. Output Format — Structured Competitor Brief

Proposed JSON schema (renderable to a human-readable brief):

```json
{
  "query": "string",
  "generated_at": "ISO-8601 timestamp",
  "latency_ms": "number",
  "sources_used": [
    {"domain": "string", "url": "string", "retrieved_at": "ISO-8601"}
  ],
  "brief": {
    "company_overview": { "text": "string", "confidence": 0.0, "citations": ["source_id"] },
    "positioning": { "text": "string", "confidence": 0.0, "citations": ["source_id"] },
    "pricing": { "text": "string", "confidence": 0.0, "citations": ["source_id"] },
    "recent_moves": { "text": "string", "confidence": 0.0, "citations": ["source_id"] },
    "strengths": { "text": "string", "confidence": 0.0, "citations": ["source_id"] },
    "weaknesses": { "text": "string", "confidence": 0.0, "citations": ["source_id"] },
    "market_signals": { "text": "string", "confidence": 0.0, "citations": ["source_id"] }
  },
  "unverified_flags": ["string describing any low-confidence/conflicting claims"],
  "search_plan_trace": ["sub-question 1", "sub-question 2", "..."]
}
```

Brief sections can be adjusted per query type (e.g., a "hiring trends" query emphasizes different fields), but the schema should stay consistent enough to render predictably.

---

## 6. Performance Budget (60-second target)

| Stage | Target time |
|---|---|
| Query planning | ≤ 5s |
| Search + fetch (initial round, parallelized across sources) | ≤ 25s |
| Grounding + confidence scoring | ≤ 10s |
| Fallback round (if triggered, capped at 1 round) | ≤ 15s |
| Synthesis + formatting | ≤ 5s |
| **Total budget** | **≤ 60s (p90)** |

Design implication: search steps must run in parallel where possible, and fallback logic must be capped (e.g., max 1 extra round) to stay within budget.

---

## 7. Evaluation Plan

### 7.1 Evaluation Set
- 25 representative queries spanning:
  - Direct competitor comparisons
  - Pricing/positioning lookups
  - Recent news / funding / product launch queries
  - Ambiguous or sparse-data queries (to test fallback behavior)
  - Queries with likely conflicting sources (to test confidence flagging)

### 7.2 Metrics per query
- **Groundedness**: % of claims with valid, verifiable citations.
- **Accuracy**: human-reviewed correctness of factual claims.
- **Source diversity**: number of distinct domains used.
- **Latency**: end-to-end time.
- **Fallback behavior**: did the agent correctly flag low-confidence claims rather than assert them?

### 7.3 Method
- Run all 25 queries through the agent; log full evidence trace + brief output.
- Human review (rubric-based) scores accuracy and groundedness per section.
- Compare "with confidence-based fallback" vs. "without" (ablation) to quantify reliability uplift — this is the key result to report.

---

## 8. Tech Stack (proposed)

| Component | Choice |
|---|---|
| Agent orchestration | LLM with tool-use / function-calling loop (planning + search + synthesis) |
| Web search | Search API (e.g., web search tool) + page fetch/extraction |
| Evidence storage | Lightweight structured store (in-memory or JSON per session; DB if persistence needed) |
| Confidence scoring | Rule-based heuristics (source count, agreement, recency) to start; can layer a learned scorer later |
| Output rendering | JSON schema → Markdown/HTML brief renderer |
| Evaluation harness | Script to batch-run the 25-query set, log traces, compute metrics |

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Search latency blows the 60s budget | Parallelize source fetches; cap fallback rounds; timeout per source fetch |
| Low-quality/unreliable sources pollute the brief | Domain allowlist/denylist; require corroboration for high-impact claims |
| Hallucinated claims without grounding | Hard constraint: synthesizer rejects any claim lacking a citation tuple |
| Sparse data for niche competitors | Explicit "insufficient data" fallback path rather than forced answer |
| Conflicting information across sources | Confidence scorer flags disagreement explicitly rather than silently picking one |

---

## 10. Milestones

1. **M1** — Query planner + search executor loop working end-to-end (no grounding/confidence yet).
2. **M2** — Source grounding + evidence store; citations wired into output.
3. **M3** — Confidence scoring + fallback controller implemented and latency-budgeted.
4. **M4** — Structured brief synthesizer + output schema finalized.
5. **M5** — 25-query evaluation set built; baseline run (no fallback) vs. full system run; reliability uplift measured and documented.
6. **M6** — Performance tuning to hit <60s p90 latency target.

---

## 11. Open Questions

- What counts as a "source" for the 5+ requirement — must they be distinct domains, or distinct pages acceptable?
- Should the confidence threshold for fallback be fixed or query-adaptive?
- Do we need rate-limit/cost controls on search calls given the tight latency budget?
- What's the human-review rubric for "accuracy" in the eval set — binary correct/incorrect, or graded?
