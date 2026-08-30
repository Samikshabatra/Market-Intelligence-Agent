"""HTTP layer: submit a query, watch the pipeline run, read the brief.

The agent is async already, so a run is just a background task; progress reaches the
browser over Server-Sent Events. SSE rather than WebSockets because the traffic is
one-directional and a reconnect should replay state, which SSE gives for free.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..agent import MarketIntelligenceAgent
from ..config import Settings
from ..llm import resolve_provider
from .store import Run, RunStore

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_RUN_DIR = Path("runs")

# Depth presets. These trade source breadth against wall-clock time; the names are what
# the user picks in the interface, so they are phrased as outcomes, not parameters.
DEPTH_PRESETS: dict[str, dict[str, float | int]] = {
    "quick": {
        "max_sub_questions": 3,
        "results_per_subquestion": 4,
        "total_budget_seconds": 30.0,
        "search_budget_seconds": 12.0,
    },
    "standard": {
        "max_sub_questions": 6,
        "results_per_subquestion": 5,
        "total_budget_seconds": 60.0,
        "search_budget_seconds": 25.0,
    },
    "deep": {
        "max_sub_questions": 8,
        "results_per_subquestion": 8,
        "total_budget_seconds": 180.0,
        "search_budget_seconds": 90.0,
    },
}


class ResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    depth: str = Field(default="standard")


def settings_for_depth(base: Settings, depth: str) -> Settings:
    preset = DEPTH_PRESETS.get(depth, DEPTH_PRESETS["standard"])
    return replace(base, **preset)


def create_app(settings: Settings | None = None, run_dir: Path | None = None) -> FastAPI:
    base_settings = settings or Settings.from_env()
    store = RunStore(run_dir or DEFAULT_RUN_DIR)

    @asynccontextmanager
    async def lifespan(instance: FastAPI) -> AsyncIterator[None]:
        yield
        # Cancel in-flight research so shutdown does not wait on a slow provider.
        for task in list(instance.state.tasks):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="Market Intelligence Agent", docs_url="/api/docs", lifespan=lifespan)
    app.state.settings = base_settings
    app.state.store = store
    app.state.tasks: set[asyncio.Task] = set()

    async def execute(run: Run) -> None:
        """Run the agent, forwarding stage transitions to any attached stream."""
        loop = asyncio.get_running_loop()
        run.status = "running"
        run.publish()

        def on_progress(stage: str, status: str, detail: dict) -> None:
            # Called from inside the agent's own coroutine, so it is already on the loop.
            run.mark_stage(stage, status, detail)

        try:
            run_settings = settings_for_depth(app.state.settings, run.depth)
            async with MarketIntelligenceAgent(run_settings) as agent:
                result = await asyncio.wait_for(
                    agent.run(run.query, on_progress=on_progress),
                    timeout=run_settings.total_budget_seconds * 2,
                )
            run.finish(result)
            await loop.run_in_executor(None, store.save, run)
        except TimeoutError:
            run.fail("The run exceeded twice its latency budget and was stopped.")
        except Exception as exc:  # surfaced to the user rather than swallowed
            logger.exception("run %s failed", run.id)
            run.fail(str(exc))

    @app.get("/api/config")
    async def config() -> dict:
        """What the interface needs to know about this deployment before a run starts."""
        current: Settings = app.state.settings
        return {
            "search_provider": current.search_provider,
            "has_search_key": bool(current.tavily_api_key) or current.search_provider == "mock",
            "model_provider": resolve_provider(current),
            "has_model": resolve_provider(current) is not None,
            "min_distinct_domains": current.min_distinct_domains,
            "confidence_threshold": current.confidence_threshold,
            "depths": {
                name: {"seconds": preset["total_budget_seconds"],
                       "steps": preset["max_sub_questions"]}
                for name, preset in DEPTH_PRESETS.items()
            },
        }

    @app.post("/api/research")
    async def start_research(request: ResearchRequest) -> dict:
        run = store.create(request.query.strip(), request.depth)
        task = asyncio.create_task(execute(run))
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)
        return {"id": run.id}

    @app.get("/api/runs")
    async def list_runs(limit: int = 50) -> dict:
        return {"runs": store.history(limit=limit)}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict:
        run = store.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="No run with that id.")
        return run.snapshot()

    @app.get("/api/runs/{run_id}/stream")
    async def stream_run(run_id: str) -> StreamingResponse:
        run = store.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="No run with that id.")

        async def events():
            queue = run.subscribe()
            try:
                while True:
                    try:
                        snapshot = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except TimeoutError:
                        # Comment frame: keeps proxies from closing an idle connection.
                        yield ": keep-alive\n\n"
                        continue
                    yield f"data: {json.dumps(snapshot)}\n\n"
                    if snapshot["status"] in {"done", "failed"}:
                        break
            finally:
                run.unsubscribe(queue)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


def main() -> int:  # pragma: no cover - process entry point
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000, log_level="info")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
