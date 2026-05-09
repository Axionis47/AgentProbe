"""Test fixtures for the Streamlit smoke tier.

These tests use streamlit.testing.v1.AppTest, which runs each page in
the same Python process as the test. The pages instantiate
AgentProbeClient at module import time, so we patch the class with a
fake before AppTest.run() — no real HTTP calls happen.

Each test starts from the same fake state: a couple of agent configs,
scenarios, rubrics, runs, and conversations. Enough to drive every
page past its `if not items: st.stop()` guard.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Make the streamlit_app package importable from this test's perspective so
# pages' `from lib.api_client import ...` resolves the same way it does when
# the user runs `streamlit run app.py` from the streamlit_app/ directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _agent_dict(idx: int = 1) -> dict[str, Any]:
    return {
        "id": f"agent-{idx}",
        "name": f"Agent {idx}",
        "description": None,
        "system_prompt": "You are helpful.",
        "model": "vertex_ai/gemini-2.0-flash",
        "temperature": 0.7,
        "max_tokens": 4096,
        "tools": [],
        "metadata": {},
        "agent_type": "builtin",
        "endpoint_url": None,
        "is_active": True,
        "created_at": _now(),
        "updated_at": _now(),
    }


def _scenario_dict(idx: int = 1) -> dict[str, Any]:
    return {
        "id": f"scen-{idx}",
        "name": f"Scenario {idx}",
        "category": "support",
        "difficulty": "easy",
        "tags": ["smoke"],
        "turns_template": [{"role": "user", "content": "hello"}],
        "user_persona": {},
        "constraints": {},
        "is_active": True,
        "created_at": _now(),
        "updated_at": _now(),
    }


def _rubric_dict(idx: int = 1) -> dict[str, Any]:
    return {
        "id": f"rub-{idx}",
        "name": f"Rubric {idx}",
        "description": None,
        "dimensions": [{"name": "helpfulness", "weight": 1.0}],
        "version": 1,
        "parent_id": None,
        "is_active": True,
        "created_at": _now(),
    }


def _run_dict(idx: int = 1, status: str = "completed") -> dict[str, Any]:
    return {
        "id": f"run-{idx}",
        "name": f"run-{idx}",
        "agent_config_id": "agent-1",
        "scenario_id": "scen-1",
        "rubric_id": "rub-1",
        "status": status,
        "num_conversations": 2,
        "config": {},
        "error_message": None,
        "started_at": _now(),
        "completed_at": _now(),
        "created_at": _now(),
        "updated_at": _now(),
    }


def _conv_dict(idx: int = 1) -> dict[str, Any]:
    return {
        "id": f"conv-{idx}",
        "eval_run_id": "run-1",
        "sequence_num": idx,
        "turns": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        "turn_count": 2,
        "total_tokens": 50,
        "total_input_tokens": 30,
        "total_output_tokens": 20,
        "total_latency_ms": 200,
        "status": "completed",
        "error_message": None,
        "metadata": {},
        "started_at": _now(),
        "completed_at": _now(),
        "created_at": _now(),
    }


def _eval_dict(conv_id: str, evaluator_type: str, score: float = 8.0) -> dict[str, Any]:
    return {
        "id": f"ev-{conv_id}-{evaluator_type}",
        "conversation_id": conv_id,
        "evaluator_type": evaluator_type,
        "evaluator_id": None,
        "rubric_id": "rub-1",
        "scores": {"helpfulness": score},
        "overall_score": score,
        "reasoning": "looks good",
        "per_turn_scores": None,
        "metadata": {},
        "created_at": _now(),
    }


def _make_fake_client() -> MagicMock:
    """Return an AgentProbeClient stand-in with predictable responses."""
    fake = MagicMock()
    fake.health.return_value = {"status": "ready", "checks": {"postgres": "ok", "redis": "ok"}}
    fake.list_agent_configs.return_value = {
        "total": 2, "offset": 0, "limit": 100,
        "items": [_agent_dict(1), _agent_dict(2)],
    }
    fake.list_scenarios.return_value = {
        "total": 1, "offset": 0, "limit": 100,
        "items": [_scenario_dict(1)],
    }
    fake.list_rubrics.return_value = {
        "total": 1, "offset": 0, "limit": 100,
        "items": [_rubric_dict(1)],
    }
    fake.list_eval_runs.return_value = {
        "total": 2, "offset": 0, "limit": 100,
        "items": [_run_dict(1, "completed"), _run_dict(2, "completed")],
    }
    fake.list_conversations.return_value = {
        "total": 2, "offset": 0, "limit": 100,
        "items": [_conv_dict(1), _conv_dict(2)],
    }
    fake.get_conversation.side_effect = lambda cid: _conv_dict(int(cid.rsplit("-", 1)[1]) if "-" in cid else 1)
    fake.get_conversation_evaluations.return_value = {
        "total": 3,
        "items": [
            _eval_dict("conv-1", "model_judge", 8.5),
            _eval_dict("conv-1", "rubric_grader", 7.8),
            _eval_dict("conv-1", "trajectory", 9.0),
        ],
    }
    fake.get_conversation_metrics.return_value = {
        "total": 1,
        "items": [
            {
                "id": "m1",
                "conversation_id": "conv-1",
                "metric_name": "tokens",
                "value": 50.0,
                "unit": "tokens",
                "metadata": {},
                "created_at": _now(),
            }
        ],
    }
    fake.get_similar_conversations.return_value = {
        "source_conversation_id": "conv-1",
        "items": [],
    }
    fake.get_rankings.return_value = {
        "scenario_id": None,
        "rankings": [],
        "total_matches": 0,
    }
    fake.get_reliability.return_value = {
        "alpha": 0.8,
        "num_items": 2,
        "num_raters": 2,
        "per_dimension_alpha": {"helpfulness": 0.8},
    }
    fake.get_calibration.return_value = {
        "pearson_r": 0.7,
        "spearman_rho": 0.7,
        "mae": 0.5,
        "rmse": 0.7,
        "bias": 0.0,
        "n": 5,
        "calibration_curve": [],
        "per_dimension": {},
    }
    fake.list_rubric_versions.return_value = []
    fake.get_eval_run.return_value = _run_dict(1, "completed")
    fake.cancel_eval_run.return_value = _run_dict(1, "cancelled")
    fake.create_agent_config.return_value = _agent_dict(99)
    fake.create_scenario.return_value = _scenario_dict(99)
    fake.create_rubric.return_value = _rubric_dict(99)
    fake.create_eval_run.return_value = _run_dict(99, "pending")
    fake.create_human_evaluation.return_value = _eval_dict("conv-1", "human")
    fake.create_pairwise_evaluation.return_value = {
        "match_id": "m1",
        "winner": "a",
        "conversation_id_a": "conv-1",
        "conversation_id_b": "conv-2",
        "reasoning": "a was better",
        "dimension_preferences": {"helpfulness": "a"},
        "confidence": 0.8,
        "evaluations": [],
    }
    fake.update_agent_config.return_value = _agent_dict(1)
    fake.delete_agent_config.return_value = None
    fake.update_scenario.return_value = _scenario_dict(1)
    fake.delete_scenario.return_value = None
    fake.update_rubric.return_value = _rubric_dict(1)
    fake.delete_rubric.return_value = None
    return fake


@pytest.fixture
def patched_client(monkeypatch):
    """Replace AgentProbeClient at module level so pages get the fake instance."""
    fake_instance = _make_fake_client()

    def _factory(*args, **kwargs):
        return fake_instance

    # Patch in the module pages import from
    monkeypatch.setattr("lib.api_client.AgentProbeClient", _factory)
    return fake_instance
