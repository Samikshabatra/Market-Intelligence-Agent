"""Source Grounding & Evidence Store - stage 3 of the pipeline.

Every factual claim in the brief must resolve to a (source_url, passage, retrieved_at)
tuple. This store is the only place those tuples live: the synthesiser can cite nothing
it did not receive from here, and `enforce_citations()` strips any claim that slips
through without grounding.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from .models import SourceKind, SourceRecord

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "for", "on", "with", "by",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that",
    "these", "those", "as", "at", "from", "about", "into", "over", "after", "before",
    "than", "then", "so", "such", "not", "no", "company",
})


def tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2}


@dataclass(slots=True)
class Coverage:
    """How well the evidence set covers one brief section."""

    section: str
    source_ids: list[str]
    domains: set[str]
    kinds: set[SourceKind]

    @property
    def domain_count(self) -> int:
        return len(self.domains)


class EvidenceStore:
    """Queryable store of grounded sources, indexed by id, domain and target section."""

    def __init__(self) -> None:
        self._by_id: dict[str, SourceRecord] = {}
        self._by_domain: dict[str, list[str]] = defaultdict(list)
        self._section_hints: dict[str, set[str]] = defaultdict(set)

    # ------------------------------------------------------------------ ingestion

    def add(self, record: SourceRecord, target_sections: list[str] | None = None) -> None:
        """Register a source. Re-adding the same id merges its section hints."""
        if record.source_id not in self._by_id:
            self._by_id[record.source_id] = record
            self._by_domain[record.domain].append(record.source_id)
        for section in target_sections or []:
            self._section_hints[section].add(record.source_id)

    def add_all(
        self,
        records: list[SourceRecord],
        hints: dict[str, list[str]] | None = None,
    ) -> None:
        """Bulk-add, mapping each record to the sections its sub-question was aimed at."""
        hints = hints or {}
        for record in records:
            self.add(record, hints.get(record.sub_question_id))

    # ------------------------------------------------------------------ queries

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, source_id: object) -> bool:
        return source_id in self._by_id

    def get(self, source_id: str) -> SourceRecord | None:
        return self._by_id.get(source_id)

    def all(self) -> list[SourceRecord]:
        return list(self._by_id.values())

    def ids(self) -> set[str]:
        return set(self._by_id)

    def distinct_domains(self) -> set[str]:
        return set(self._by_domain)

    def distinct_kinds(self) -> set[SourceKind]:
        return {r.source_kind for r in self._by_id.values()}

    def domains_for(self, source_ids: list[str]) -> set[str]:
        return {self._by_id[s].domain for s in source_ids if s in self._by_id}

    def resolve(self, source_ids: list[str]) -> list[SourceRecord]:
        """Records for the given ids, silently skipping ids that were never stored."""
        return [self._by_id[s] for s in source_ids if s in self._by_id]

    def citation_tuples(self, source_ids: list[str]) -> list[tuple[str, str, str]]:
        return [record.citation_tuple() for record in self.resolve(source_ids)]

    # ------------------------------------------------------------------ retrieval

    def for_section(self, section: str, *, limit: int = 8) -> list[SourceRecord]:
        """Sources the planner aimed at this section, best-relevance first."""
        hinted = self.resolve(sorted(self._section_hints.get(section, set())))
        hinted.sort(key=lambda r: r.relevance, reverse=True)
        return hinted[:limit]

    def search(self, text: str, *, limit: int = 8) -> list[SourceRecord]:
        """Lexical overlap search across stored passages - enough to rank grounding
        candidates without pulling in an embedding dependency inside the 60s budget."""
        needle = tokenize(text)
        if not needle:
            return []
        scored: list[tuple[float, SourceRecord]] = []
        for record in self._by_id.values():
            haystack = tokenize(f"{record.title} {record.passage}")
            if not haystack:
                continue
            overlap = len(needle & haystack) / len(needle)
            if overlap > 0:
                scored.append((overlap + 0.15 * record.relevance, record))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [record for _, record in scored[:limit]]

    def coverage(self, section: str) -> Coverage:
        records = self.for_section(section, limit=100)
        return Coverage(
            section=section,
            source_ids=[r.source_id for r in records],
            domains={r.domain for r in records},
            kinds={r.source_kind for r in records},
        )

    # ------------------------------------------------------------------ enforcement

    def enforce_citations(self, citations: list[str]) -> list[str]:
        """Drop ids the store never issued, de-duplicated and order-preserving.

        This is the hard constraint from section 9 of the spec: a hallucinated citation
        cannot survive into the brief, and a claim left with no citation cannot be
        asserted at all.
        """
        seen: set[str] = set()
        kept: list[str] = []
        for source_id in citations:
            if source_id in self._by_id and source_id not in seen:
                seen.add(source_id)
                kept.append(source_id)
        return kept
