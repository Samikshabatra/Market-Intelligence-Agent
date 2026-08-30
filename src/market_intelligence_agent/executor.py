"""Search Executor - stage 2 of the pipeline.

Runs every sub-question concurrently, enforces the per-request timeout and the overall
search budget, drops denylisted and duplicate sources, and turns surviving hits into
`SourceRecord`s ready for the evidence store.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field

from .config import Settings
from .models import SearchPlan, SourceKind, SourceRecord, SubQuestion, utcnow
from .search import SearchError, SearchProvider, SearchResult, canonical_url, domain_of

logger = logging.getLogger(__name__)

MAX_PASSAGE_CHARS = 1200
MIN_PASSAGE_CHARS = 80

# Domain patterns that identify what kind of source a hit is. Order matters: the first
# match wins, so specific platforms are listed before the generic fallbacks.
_KIND_PATTERNS: tuple[tuple[SourceKind, tuple[str, ...]], ...] = (
    ("funding_database", ("crunchbase.com", "pitchbook.com", "tracxn.com", "sec.gov")),
    ("review_platform", ("g2.com", "capterra.com", "trustradius.com", "getapp.com")),
    ("industry_report", ("gartner.com", "forrester.com", "idc.com", "cbinsights.com")),
    ("social_professional", ("linkedin.com", "glassdoor.com", "indeed.com")),
    ("community", ("news.ycombinator.com", "reddit.com", "stackoverflow.com")),
    (
        "news",
        (
            "techcrunch.com", "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
            "theinformation.com", "businessinsider.com", "venturebeat.com",
            "cnbc.com", "forbes.com", "axios.com",
        ),
    ),
)


def classify_source(domain: str, subject: str = "") -> SourceKind:
    """Best-effort source-kind label, used for diversity checks and fallback routing."""
    domain = domain.lower().removeprefix("www.")
    for kind, patterns in _KIND_PATTERNS:
        if any(domain == p or domain.endswith("." + p) for p in patterns):
            return kind
    # A comparison query names several companies, so test each subject token on its own -
    # "Ramp Brex" must still recognise ramp.com as a company site.
    flattened = domain.replace(".", "").replace("-", "")
    for token in re.findall(r"[a-z0-9]{3,}", subject.lower()):
        if token in flattened:
            return "company_site"
    return "other"


def _passage_from(result: SearchResult) -> str:
    """Trim extracted content to a citable passage without cutting mid-sentence."""
    text = " ".join((result.content or result.title or "").split())
    if len(text) <= MAX_PASSAGE_CHARS:
        return text
    clipped = text[:MAX_PASSAGE_CHARS]
    boundary = max(clipped.rfind(". "), clipped.rfind("! "), clipped.rfind("? "))
    return clipped[: boundary + 1] if boundary > MIN_PASSAGE_CHARS else clipped.rstrip() + "..."


def _content_fingerprint(text: str) -> str:
    """Hash of the normalised passage head, so syndicated copies collapse into one."""
    normalised = re.sub(r"[^a-z0-9 ]", "", text.lower())
    return hashlib.sha1(" ".join(normalised.split())[:400].encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ExecutionReport:
    """What one search round retrieved, and what it discarded on the way."""

    records: list[SourceRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    dropped_denylist: int = 0
    dropped_duplicate: int = 0
    dropped_thin: int = 0
    queries_run: int = 0
    elapsed_ms: float = 0.0

    def distinct_domains(self) -> set[str]:
        return {r.domain for r in self.records}

    def distinct_kinds(self) -> set[SourceKind]:
        return {r.source_kind for r in self.records}


class SearchExecutor:
    """Stateful across rounds: dedupe memory persists so fallbacks add new sources only."""

    def __init__(self, provider: SearchProvider, settings: Settings) -> None:
        self._provider = provider
        self._settings = settings
        self._seen_urls: set[str] = set()
        self._seen_fingerprints: set[str] = set()
        self._counter = 0

    async def run(
        self,
        plan: SearchPlan,
        *,
        deadline: float | None = None,
        round_index: int = 0,
    ) -> ExecutionReport:
        """Execute every sub-question in the plan concurrently and merge the results."""
        started = time.perf_counter()
        report = ExecutionReport()
        semaphore = asyncio.Semaphore(self._settings.max_parallel_searches)

        async def one(sub_question: SubQuestion) -> tuple[SubQuestion, list[SearchResult]]:
            async with semaphore:
                try:
                    hits = await self._provider.search(
                        sub_question.search_query,
                        max_results=self._settings.results_per_subquestion,
                    )
                    return sub_question, hits
                except SearchError as exc:
                    report.errors.append(str(exc))
                    return sub_question, []
                except Exception as exc:  # a bad provider must not kill the run
                    logger.warning("search failed for %s: %s", sub_question.id, exc)
                    report.errors.append(f"{sub_question.id}: {exc}")
                    return sub_question, []

        tasks = [asyncio.create_task(one(sq)) for sq in plan.sub_questions]
        report.queries_run = len(tasks)
        timeout = self._remaining(deadline)
        try:
            gathered = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
        except asyncio.TimeoutError:
            report.errors.append(f"search round {round_index} hit its {timeout:.1f}s budget")
            gathered = [t.result() for t in tasks if t.done() and not t.cancelled()]
            for task in tasks:
                if not task.done():
                    task.cancel()

        for sub_question, hits in gathered:
            for hit in hits:
                if len(report.records) >= self._settings.max_sources:
                    break
                record = self._admit(hit, sub_question, plan.subject, round_index, report)
                if record is not None:
                    report.records.append(record)

        report.records.sort(key=lambda r: r.relevance, reverse=True)
        report.elapsed_ms = (time.perf_counter() - started) * 1000
        return report

    # ------------------------------------------------------------------ internals

    def _remaining(self, deadline: float | None) -> float:
        """Seconds left for this round, bounded by the configured search budget."""
        budget = self._settings.search_budget_seconds
        if deadline is None:
            return budget
        return max(0.5, min(budget, deadline - time.monotonic()))

    def _admit(
        self,
        hit: SearchResult,
        sub_question: SubQuestion,
        subject: str,
        round_index: int,
        report: ExecutionReport,
    ) -> SourceRecord | None:
        """Apply hygiene rules; return a SourceRecord or None with the reason counted."""
        url = canonical_url(hit.url)
        domain = domain_of(url)
        if not domain or self._settings.is_denied(domain):
            report.dropped_denylist += 1
            return None
        if url in self._seen_urls:
            report.dropped_duplicate += 1
            return None

        passage = _passage_from(hit)
        if len(passage) < MIN_PASSAGE_CHARS:
            report.dropped_thin += 1
            return None

        fingerprint = _content_fingerprint(passage)
        if fingerprint in self._seen_fingerprints:
            report.dropped_duplicate += 1
            return None

        self._seen_urls.add(url)
        self._seen_fingerprints.add(fingerprint)
        self._counter += 1

        return SourceRecord(
            source_id=f"s{self._counter}",
            url=url,
            domain=domain,
            title=hit.title,
            passage=passage,
            retrieved_at=utcnow(),
            published_at=hit.published_at,
            source_kind=classify_source(domain, subject),
            relevance=max(0.0, min(1.0, hit.score)),
            sub_question_id=sub_question.id,
            round_index=round_index,
        )
