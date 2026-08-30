"""Run store: in-flight state plus a durable record of finished research.

Runs live in memory while they execute (so the progress stream can read them) and are
written to one JSON file each when they finish, which is what the history view reads.
No database: a run is a self-contained document and the volume is a handful per user.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ..agent import STAGES
from ..models import AgentResult

RunStatus = Literal["queued", "running", "done", "failed"]

MAX_HISTORY = 200


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Run:
    """One research run, from submission to stored brief."""

    id: str
    query: str
    depth: str
    status: RunStatus = "queued"
    created_at: str = field(default_factory=_now)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    result: dict | None = None
    error: str | None = None

    # Consumers attached to this run's live event stream.
    _subscribers: list[asyncio.Queue] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self.stages:
            self.stages = {
                key: {"key": key, "label": label, "status": "pending", "detail": {}}
                for key, label in STAGES
            }

    # ---------------------------------------------------------------- events

    def snapshot(self) -> dict:
        """Everything a client needs to render this run at its current moment."""
        return {
            "id": self.id,
            "query": self.query,
            "depth": self.depth,
            "status": self.status,
            "created_at": self.created_at,
            "stages": list(self.stages.values()),
            "result": self.result,
            "error": self.error,
        }

    def summary(self) -> dict:
        """Compact row for the history list."""
        result = self.result or {}
        brief = result.get("brief", {})
        confidences = [
            section.get("confidence", 0.0)
            for section in brief.values()
            if section.get("text")
        ]
        return {
            "id": self.id,
            "query": self.query,
            "status": self.status,
            "created_at": self.created_at,
            "latency_ms": result.get("latency_ms"),
            "sources": len(result.get("sources_used", [])),
            "domains": len({s["domain"] for s in result.get("sources_used", [])}),
            "confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
            "flags": len(result.get("unverified_flags", [])),
        }

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        # Hand the newcomer the current state first, so a client that connects late
        # still renders completed stages instead of an empty rail.
        queue.put_nowait(self.snapshot())
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def publish(self) -> None:
        snapshot = self.snapshot()
        for queue in list(self._subscribers):
            queue.put_nowait(snapshot)

    def mark_stage(self, stage: str, status: str, detail: dict) -> None:
        entry = self.stages.get(stage)
        if entry is None:
            return
        entry["status"] = status
        if detail:
            entry["detail"] = {**entry["detail"], **detail}
        self.publish()

    def finish(self, result: AgentResult) -> None:
        self.result = result.to_spec_dict()
        # The spec dict is deliberately minimal; the UI also shows per-section status,
        # stage timings and the source metadata the brief cites.
        self.result["detail"] = {
            "sections": {
                name: {
                    "status": section.status,
                    "breakdown": section.breakdown.model_dump() if section.breakdown else None,
                }
                for name, section in result.brief.sections()
            },
            "sources": [
                {
                    "source_id": s.source_id,
                    "domain": s.domain,
                    "url": s.url,
                    "title": s.title,
                    "passage": s.passage,
                    "kind": s.source_kind,
                    "relevance": s.relevance,
                    "published_at": s.published_at.isoformat() if s.published_at else None,
                    "retrieved_at": s.retrieved_at.isoformat(),
                }
                for s in result.sources_used
            ],
            "timings": result.timings.model_dump(),
            "fallback_rounds": result.fallback_rounds,
            "budget_exceeded": result.budget_exceeded,
        }
        self.status = "done"
        for entry in self.stages.values():
            if entry["status"] != "done":
                entry["status"] = "done"
        self.publish()

    def fail(self, message: str) -> None:
        self.status = "failed"
        self.error = message
        for entry in self.stages.values():
            if entry["status"] == "active":
                entry["status"] = "failed"
        self.publish()


class RunStore:
    """Keeps live runs addressable and finished runs on disk."""

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._runs: dict[str, Run] = {}
        self._order: deque[str] = deque(maxlen=MAX_HISTORY)

    def create(self, query: str, depth: str) -> Run:
        run = Run(id=uuid.uuid4().hex[:12], query=query, depth=depth)
        self._runs[run.id] = run
        self._order.appendleft(run.id)
        return run

    def get(self, run_id: str) -> Run | None:
        run = self._runs.get(run_id)
        if run is not None:
            return run
        return self._load(run_id)

    def save(self, run: Run) -> None:
        path = self._dir / f"{run.id}.json"
        path.write_text(json.dumps(run.snapshot(), indent=2), encoding="utf-8")

    def history(self, limit: int = 50) -> list[dict]:
        """Live runs first, then stored ones, newest first, de-duplicated by id."""
        seen: set[str] = set()
        rows: list[dict] = []
        for run_id in self._order:
            run = self._runs.get(run_id)
            if run is not None and run.id not in seen:
                seen.add(run.id)
                rows.append(run.summary())

        stored = sorted(self._dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in stored:
            if path.stem in seen or len(rows) >= limit:
                continue
            run = self._load(path.stem)
            if run is not None:
                seen.add(run.id)
                rows.append(run.summary())
        return rows[:limit]

    # ---------------------------------------------------------------- internals

    def _load(self, run_id: str) -> Run | None:
        path = self._dir / f"{run_id}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        run = Run(
            id=payload["id"],
            query=payload["query"],
            depth=payload.get("depth", "standard"),
            status=payload.get("status", "done"),
            created_at=payload.get("created_at", _now()),
            result=payload.get("result"),
            error=payload.get("error"),
        )
        for entry in payload.get("stages", []):
            run.stages[entry["key"]] = entry
        return run
