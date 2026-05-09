"""End-to-end test against the real docker-compose stack.

This is the only test in the suite that exercises Postgres + Redis +
Kafka + ChromaDB + FastAPI + Celery together. If it passes, the
integrated system actually works. If it doesn't, no amount of unit
tests covers the bug.

Marker `e2e` keeps it opt-in — `pytest backend/tests/unit` skips it.

The LLM is the FakeLLMClient (selected via AGENTPROBE_LLM_PROVIDER=fake)
so the test is deterministic and runs without a real model API.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.e2e


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
POLL_TIMEOUT_SECONDS = 90.0
POLL_INTERVAL_SECONDS = 2.0


def _post(http: httpx.Client, path: str, body: dict[str, Any]) -> dict[str, Any]:
    r = http.post(path, json=body)
    r.raise_for_status()
    return r.json()


def _get(http: httpx.Client, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    r = http.get(path, params=params)
    r.raise_for_status()
    return r.json()


def _poll_until_terminal(http: httpx.Client, run_id: str) -> dict[str, Any]:
    deadline = time.time() + POLL_TIMEOUT_SECONDS
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = _get(http, f"/api/v1/eval-runs/{run_id}")
        if last.get("status") in TERMINAL_STATUSES:
            return last
        time.sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError(
        f"eval-run {run_id} did not reach a terminal status within "
        f"{POLL_TIMEOUT_SECONDS}s; last status was {last.get('status')!r}"
    )


def test_full_evaluation_flow(http: httpx.Client) -> None:
    """End-to-end: create config + scenario + rubric, run an eval, assert outputs."""

    agent = _post(
        http,
        "/api/v1/agent-configs",
        {
            "name": "e2e-agent",
            "system_prompt": "You are a test assistant.",
            "model": "fake",
            "temperature": 0.0,
            "max_tokens": 512,
            "tools": [
                {
                    "name": "lookup_order",
                    "description": "Look up an order by ID",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                    },
                }
            ],
        },
    )

    scenario = _post(
        http,
        "/api/v1/scenarios",
        {
            "name": "e2e-scenario",
            "category": "support",
            "difficulty": "easy",
            "tags": ["e2e"],
            "turns_template": [
                {"role": "user", "content": "I need to look up order #42"},
            ],
            "user_persona": {"goal": "find order"},
            "constraints": {
                "expected_tools": ["lookup_order"],
            },
        },
    )

    rubric = _post(
        http,
        "/api/v1/rubrics",
        {
            "name": "e2e-rubric",
            "description": "minimal rubric for e2e",
            "dimensions": [
                {"name": "helpfulness", "description": "Did it help?", "weight": 1.0},
            ],
        },
    )

    run = _post(
        http,
        "/api/v1/eval-runs",
        {
            "name": "e2e-run",
            "agent_config_id": agent["id"],
            "scenario_id": scenario["id"],
            "rubric_id": rubric["id"],
            "num_conversations": 2,
        },
    )

    final = _poll_until_terminal(http, run["id"])
    assert final["status"] == "completed", (
        f"expected completed, got {final['status']}: {final.get('error_message')!r}"
    )

    # Conversations were persisted.
    convs_resp = _get(
        http,
        "/api/v1/conversations",
        params={"eval_run_id": run["id"], "limit": 100},
    )
    assert convs_resp["total"] == 2
    conv_ids = [c["id"] for c in convs_resp["items"]]
    assert all(c["status"] == "completed" for c in convs_resp["items"])

    # Each conversation has at least one evaluation per evaluator type that
    # we expect to run automatically: model_judge, rubric_grader, trajectory.
    expected_evaluators = {"model_judge", "rubric_grader", "trajectory"}
    for cid in conv_ids:
        evals_resp = _get(http, f"/api/v1/conversations/{cid}/evaluations")
        types = {e["evaluator_type"] for e in evals_resp["items"]}
        missing = expected_evaluators - types
        assert not missing, f"conversation {cid} missing evaluators: {missing}"

    # Metrics rows exist (token totals, etc.).
    for cid in conv_ids:
        metrics_resp = _get(http, f"/api/v1/conversations/{cid}/metrics")
        assert metrics_resp["total"] > 0, f"no metrics for conversation {cid}"

    # Similarity search has both records indexed in Chroma — querying one
    # should at least come back without erroring, and ideally find the other.
    similar = _get(
        http,
        f"/api/v1/conversations/{conv_ids[0]}/similar",
        params={"limit": 5, "same_scenario": "false"},
    )
    assert similar["source_conversation_id"] == conv_ids[0]
    # Allow zero results (Chroma might still be indexing) — but the
    # endpoint must respond with a valid shape, not 500.
    assert isinstance(similar["items"], list)
