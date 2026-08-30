"""Deterministic offline search backend.

Used by the test-suite and by `--provider mock` so the whole pipeline (planning,
grounding, confidence, fallback, synthesis) can be exercised without network calls or
API spend. Results come from a JSON corpus when one matches, and from a deterministic
synthetic generator otherwise, so a given query always yields the same sources.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .base import SearchProvider, SearchResult

# Domain pool spans the six source kinds the spec asks for, so a synthetic run still
# clears the "5+ distinct domains" floor the way a real run would.
_SYNTHETIC_DOMAINS: tuple[tuple[str, str, int], ...] = (
    ("{slug}.com", "Official site", 30),
    ("techcrunch.com", "TechCrunch", 90),
    ("g2.com", "G2 reviews", 200),
    ("crunchbase.com", "Crunchbase profile", 150),
    ("linkedin.com", "LinkedIn company page", 45),
    ("reuters.com", "Reuters", 120),
    ("gartner.com", "Gartner note", 400),
    ("news.ycombinator.com", "Hacker News discussion", 60),
)

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "vs", "versus", "how", "does", "do", "is",
    "in", "for", "to", "what", "with", "compare", "against", "their", "its",
}


def _slug(query: str) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in _STOPWORDS]
    return (words[0] if words else "example")[:24]


def _stable_float(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


class MockSearchProvider(SearchProvider):
    """Fixture-first, synthetic-fallback provider with a recorded call log."""

    name = "mock"

    def __init__(
        self,
        fixtures: dict[str, list[dict]] | None = None,
        *,
        corpus_path: str | Path | None = None,
        results_per_query: int = 5,
        now: datetime | None = None,
    ) -> None:
        self._fixtures: dict[str, list[dict]] = dict(fixtures or {})
        if corpus_path is not None:
            self._fixtures.update(json.loads(Path(corpus_path).read_text(encoding="utf-8")))
        self._results_per_query = results_per_query
        self._now = now or datetime.now(UTC)
        self.calls: list[str] = []

    async def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        self.calls.append(query)
        fixture = self._match_fixture(query)
        if fixture is not None:
            return [self._from_fixture(item) for item in fixture[:max_results]]
        return self._synthesise(query, max_results)

    # ------------------------------------------------------------------ internals

    def _match_fixture(self, query: str) -> list[dict] | None:
        """Exact key, then substring, then token overlap.

        The token pass matters because the planner rewrites a user question into its own
        search strings; a fixture keyed on the topic still has to match those rewrites.
        """
        if query in self._fixtures:
            return self._fixtures[query]
        needle = query.lower().strip()
        for key, value in self._fixtures.items():
            if key.lower().strip() in needle or needle in key.lower().strip():
                return value
        query_tokens = set(re.findall(r"[a-z0-9-]{3,}", needle))
        for key, value in self._fixtures.items():
            key_tokens = set(re.findall(r"[a-z0-9-]{3,}", key.lower()))
            if key_tokens and len(key_tokens & query_tokens) / len(key_tokens) >= 0.6:
                return value
        return None

    def _from_fixture(self, item: dict) -> SearchResult:
        published = item.get("published_at")
        parsed = None
        if isinstance(published, str) and published:
            parsed = datetime.fromisoformat(published.replace("Z", "+00:00"))
        return SearchResult(
            url=item["url"],
            title=item.get("title", ""),
            content=item.get("content", ""),
            score=float(item.get("score", 0.8)),
            published_at=parsed,
            provider=self.name,
            raw=item,
        )

    def _synthesise(self, query: str, max_results: int) -> list[SearchResult]:
        slug = _slug(query)
        # Rotate the domain window per query so sibling sub-questions surface different
        # domains, the way real searches do - otherwise cross-query dedupe would collapse
        # a whole plan down to one sub-question's worth of sources.
        offset = int(_stable_float(query) * len(_SYNTHETIC_DOMAINS))
        window = [
            _SYNTHETIC_DOMAINS[(offset + i) % len(_SYNTHETIC_DOMAINS)]
            for i in range(min(max_results, len(_SYNTHETIC_DOMAINS)))
        ]
        fingerprint = f"{int(_stable_float(query) * 1_000_000):06d}"
        results: list[SearchResult] = []
        for index, (template, label, age_days) in enumerate(window):
            domain = template.format(slug=slug)
            jitter = _stable_float(query, domain)
            results.append(
                SearchResult(
                    url=f"https://{domain}/{slug}/{fingerprint}-{index}",
                    title=f"{label}: {query}",
                    content=(
                        f"{label} coverage relevant to '{query}'. "
                        f"Synthetic passage {fingerprint}-{index} generated offline for "
                        f"{domain}; it carries no real-world facts and exists only to "
                        "exercise the pipeline end to end."
                    ),
                    score=round(0.9 - 0.07 * index + 0.05 * jitter, 3),
                    published_at=self._now - timedelta(days=age_days + int(30 * jitter)),
                    provider=self.name,
                    raw={"synthetic": True, "domain": domain},
                )
            )
        return results
