from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from market_intelligence_agent.config import Settings
from market_intelligence_agent.web.app import create_app, settings_for_depth
from market_intelligence_agent.web.store import Run, RunStore


@pytest.fixture()
def client(tmp_path):
    settings = Settings(
        search_provider="mock",
        anthropic_api_key=None,
        total_budget_seconds=15.0,
    )
    with TestClient(create_app(settings, run_dir=tmp_path)) as test_client:
        yield test_client


def wait_for_completion(client: TestClient, run_id: str) -> dict:
    """Drive the SSE stream to completion and return the final snapshot."""
    with client.stream("GET", f"/api/runs/{run_id}/stream") as response:
        assert response.status_code == 200
        last = None
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            last = json.loads(line[6:])
            if last["status"] in {"done", "failed"}:
                break
    assert last is not None
    return last


def test_config_reports_what_the_interface_needs(client: TestClient):
    payload = client.get("/api/config").json()
    assert payload["search_provider"] == "mock"
    assert payload["has_model"] is False
    assert set(payload["depths"]) == {"quick", "standard", "deep"}


def test_depth_presets_change_breadth_and_budget():
    base = Settings()
    quick = settings_for_depth(base, "quick")
    deep = settings_for_depth(base, "deep")
    assert quick.total_budget_seconds < deep.total_budget_seconds
    assert quick.max_sub_questions < deep.max_sub_questions


def test_unknown_depth_falls_back_to_standard():
    resolved = settings_for_depth(Settings(), "turbo")
    assert resolved.total_budget_seconds == 60.0


def test_research_run_completes_and_returns_a_grounded_brief(client: TestClient):
    run_id = client.post("/api/research", json={"query": "Ramp versus Brex"}).json()["id"]
    final = wait_for_completion(client, run_id)

    assert final["status"] == "done"
    assert all(stage["status"] == "done" for stage in final["stages"])

    result = final["result"]
    valid_ids = {s["source_id"] for s in result["detail"]["sources"]}
    for section in result["brief"].values():
        assert set(section["citations"]) <= valid_ids


def test_stream_replays_state_to_a_late_subscriber(client: TestClient):
    run_id = client.post("/api/research", json={"query": "Ramp versus Brex"}).json()["id"]
    wait_for_completion(client, run_id)
    # Subscribing after the run finished must still yield a terminal snapshot,
    # otherwise a refreshed browser tab would hang on an empty progress rail.
    final = wait_for_completion(client, run_id)
    assert final["status"] == "done"


def test_finished_run_is_readable_and_listed(client: TestClient):
    run_id = client.post("/api/research", json={"query": "Ramp versus Brex"}).json()["id"]
    wait_for_completion(client, run_id)

    fetched = client.get(f"/api/runs/{run_id}").json()
    assert fetched["result"]["query"] == "Ramp versus Brex"

    rows = client.get("/api/runs").json()["runs"]
    assert any(row["id"] == run_id for row in rows)
    row = next(row for row in rows if row["id"] == run_id)
    assert row["sources"] > 0 and row["domains"] > 0


def test_missing_run_returns_404(client: TestClient):
    assert client.get("/api/runs/deadbeef").status_code == 404
    assert client.get("/api/runs/deadbeef/stream").status_code == 404


def test_query_must_be_substantial(client: TestClient):
    assert client.post("/api/research", json={"query": "hi"}).status_code == 422


def test_index_is_served(client: TestClient):
    page = client.get("/")
    assert page.status_code == 200
    assert "Market Intelligence Agent" in page.text


# ------------------------------------------------------------------ store


def test_store_persists_and_reloads_a_run(tmp_path):
    store = RunStore(tmp_path)
    run = store.create("Notion versus Coda", "standard")
    run.status = "done"
    run.result = {"query": "Notion versus Coda", "sources_used": [], "brief": {}}
    store.save(run)

    reloaded = RunStore(tmp_path).get(run.id)
    assert reloaded is not None
    assert reloaded.query == "Notion versus Coda"


def test_store_ignores_a_corrupt_record(tmp_path):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert RunStore(tmp_path).get("broken") is None
    assert RunStore(tmp_path).history() == []


def test_new_run_starts_with_every_stage_pending():
    run = Run(id="x", query="q", depth="standard")
    assert [stage["status"] for stage in run.stages.values()] == ["pending"] * 5


def test_summary_averages_confidence_over_asserted_sections_only():
    run = Run(id="x", query="q", depth="standard")
    run.result = {
        "sources_used": [{"domain": "a.com"}],
        "brief": {
            "pricing": {"text": "asserted", "confidence": 0.8, "citations": ["s1"]},
            "positioning": {"text": "", "confidence": 0.0, "citations": []},
        },
        "unverified_flags": ["one"],
    }
    summary = run.summary()
    assert summary["confidence"] == 0.8  # the empty section must not drag the mean down
    assert summary["flags"] == 1
