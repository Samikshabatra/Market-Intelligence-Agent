"""Tavily search backend.

Tavily returns extracted page content alongside the URL, so one call covers both the
"web search" and "page fetch" steps of the executor and keeps us inside the 25s search
budget in section 6 of the spec.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from .base import SearchError, SearchProvider, SearchResult

TAVILY_ENDPOINT = "https://api.tavily.com/search"


def _parse_date(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().replace("Z", "+00:00")
    for parse in (datetime.fromisoformat,):
        try:
            parsed = parse(text)
        except ValueError:
            continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%d", "%d %b %Y"):
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


class TavilySearchProvider(SearchProvider):
    """Async Tavily client with a per-request timeout and a shared connection pool."""

    name = "tavily"

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 8.0,
        search_depth: str = "advanced",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise SearchError("TAVILY_API_KEY is not set; use --provider mock for offline runs.")
        self._api_key = api_key
        self._timeout = timeout
        self._search_depth = search_depth
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        payload = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": self._search_depth,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
        try:
            response = await self._client.post(
                TAVILY_ENDPOINT, json=payload, timeout=self._timeout
            )
            response.raise_for_status()
            body = response.json()
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise SearchError(f"Tavily timed out after {self._timeout}s for {query!r}") from exc
        except httpx.HTTPStatusError as exc:
            raise SearchError(
                f"Tavily returned {exc.response.status_code} for {query!r}"
            ) from exc
        except httpx.HTTPError as exc:
            raise SearchError(f"Tavily request failed for {query!r}: {exc}") from exc
        except ValueError as exc:
            raise SearchError(f"Tavily returned malformed JSON for {query!r}") from exc

        results: list[SearchResult] = []
        for item in body.get("results", []) or []:
            url = item.get("url")
            if not url:
                continue
            results.append(
                SearchResult(
                    url=url,
                    title=item.get("title") or "",
                    content=item.get("content") or "",
                    score=float(item.get("score") or 0.0),
                    published_at=_parse_date(item.get("published_date")),
                    provider=self.name,
                    raw=item,
                )
            )
        return results

    async def aclose(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()
