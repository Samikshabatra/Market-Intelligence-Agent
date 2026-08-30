# Decisions on the spec's open questions

Section 11 of the spec left four questions open. This records how each was resolved in
v1 and what would change the answer.

## 1. What counts as a "source" for the 5+ requirement?

**Distinct registrable domains, not distinct pages.**

Five pages from one vendor's own site are one perspective, not five. The corroboration
signal in the confidence scorer counts domains for the same reason: the second
*independent* domain is worth far more than the fifth page of the first one.

Implementation: `SearchExecutor` canonicalises URLs (stripping tracking parameters) and
fingerprints passages, so a syndicated copy of one article across two domains collapses
to one source rather than counting twice. `AgentResult.distinct_domains()` is what the
5-source floor is measured against.

Consequence worth knowing: a query about a company with little third-party coverage will
legitimately fail the floor. That is reported as a flag, not hidden.

## 2. Should the confidence threshold be fixed or query-adaptive?

**Fixed in v1 (`0.55`, configurable), with the adaptivity moved elsewhere.**

An adaptive threshold is a second thing to tune with no data to tune it on yet, and it
makes the eval set harder to interpret: if both the score and the bar move per query,
a regression cannot be attributed. A fixed bar with an explainable score is easier to
validate first.

The behaviour the adaptive threshold was meant to provide is instead handled by making
the *score* query-sensitive: recency decay matters more for a recent-news query because
its sources are dated, and the ambiguity penalty fires on hedged sources regardless of
topic. The scorer also emits a `ConfidenceBreakdown` with reasons, so a wrong threshold
is diagnosable from the eval artifacts rather than invisible.

Revisit once the human-reviewed accuracy column in `rubric.csv` is populated across a
few runs: if a single bar is systematically wrong for one query category, make it
per-category before making it per-query.

## 3. Do we need rate-limit / cost controls on search calls?

**Yes, and they are the same controls that protect the latency budget.**

- `max_parallel_searches` (default 8) caps concurrency, which bounds burst rate.
- `results_per_subquestion` and `max_sub_questions` bound calls per run.
- `max_sources` caps what reaches the synthesis prompt, which is the real cost driver -
  model input tokens, not search calls.
- The fallback round is capped at one, so a hard query cannot fan out indefinitely.
- `per_request_timeout_seconds` stops a slow provider from holding a connection.

Worst case per query is `(max_sub_questions + 2 seed + fallback) x results_per_subquestion`
search results. With the defaults that is bounded and predictable, which is what matters
for both a rate limit and a bill.

Not implemented in v1: a persistent cross-run cache. Two runs of the same query pay
twice. Worth adding if the eval loop is run often, but it would make the offline eval
results depend on run order, so it needs to be cache-bypassed under test.

## 4. What is the human-review rubric for accuracy?

**Graded per section, not binary per query.**

A brief is seven sections; scoring the whole run correct/incorrect throws away the
information about *which* section failed, which is exactly what is needed to improve the
system. `rubric.csv` is emitted per run with one row per query and a blank
`accuracy_0_to_1` column, alongside the machine-computed groundedness so the reviewer can
see whether a wrong claim was also badly grounded.

Suggested grading, to keep reviewers consistent:

| Score | Meaning |
|---|---|
| 1.0 | Every asserted claim is correct and the citations support them. |
| 0.5 | Asserted claims are broadly correct but at least one is imprecise, stale, or supported only indirectly by its citation. |
| 0.0 | At least one asserted claim is wrong, or a citation does not support the claim it is attached to. |

A section the agent declined to assert is **not** scored as an accuracy failure - that is
what flag recall measures separately. Penalising a correct decline in the accuracy column
would push the system towards confident guessing, which is the failure mode the whole
confidence stage exists to prevent.
