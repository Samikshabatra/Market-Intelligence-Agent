"""Search provider interface shared by every backend."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse

_TRACKING_PARAMS = re.compile(r"[?&](utm_[^=]+|ref|ref_src|fbclid|gclid|mc_cid|mc_eid)=[^&]*")


def domain_of(url: str) -> str:
    """Registrable-ish host for a URL, lowercased and without a `www.` prefix."""
    host = urlparse(url).netloc.lower()
    if "@" in host:
        host = host.rsplit("@", 1)[1]
    host = host.split(":", 1)[0]
    return host.removeprefix("www.")


def canonical_url(url: str) -> str:
    """Strip tracking params and trailing slashes so near-identical URLs dedupe."""
    cleaned = _TRACKING_PARAMS.sub("", url)
    cleaned = cleaned.replace("?&", "?").rstrip("?&")
    if cleaned.endswith("/") and len(urlparse(cleaned).path) > 1:
        cleaned = cleaned[:-1]
    return cleaned


@dataclass(slots=True)
class SearchResult:
    """One raw hit from a provider, before it becomes grounded evidence."""

    url: str
    title: str = ""
    content: str = ""
    score: float = 0.0
    published_at: datetime | None = None
    provider: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def domain(self) -> str:
        return domain_of(self.url)


class SearchError(RuntimeError):
    """Raised when a provider fails in a way the executor should record but survive."""


class SearchProvider(ABC):
    """Backends implement one coroutine; the executor handles parallelism and budget."""

    name: str = "base"

    @abstractmethod
    async def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        """Return ranked results for one query. Must not raise on empty results."""

    async def aclose(self) -> None:
        """Release any held connections. Safe to call more than once."""
        return
