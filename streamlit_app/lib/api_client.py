import os
from typing import Any

import httpx

API_URL = os.getenv("AGENTPROBE_API_URL", "http://localhost:8080")


class AgentProbeClient:
    """HTTP client for the AgentProbe FastAPI backend."""

    def __init__(self, base_url: str = API_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=30.0)

    def _url(self, path: str) -> str:
        return f"/api/v1{path}"

    # Agent Configs
    def list_agent_configs(self, **params: Any) -> dict[str, Any]:
        r = self.client.get(self._url("/agent-configs"), params=params)
        r.raise_for_status()
        return r.json()

    def create_agent_config(self, data: dict[str, Any]) -> dict[str, Any]:
        r = self.client.post(self._url("/agent-configs"), json=data)
        r.raise_for_status()
        return r.json()

    # Scenarios
    def list_scenarios(self, **params: Any) -> dict[str, Any]:
        r = self.client.get(self._url("/scenarios"), params=params)
        r.raise_for_status()
        return r.json()

    def create_scenario(self, data: dict[str, Any]) -> dict[str, Any]:
        r = self.client.post(self._url("/scenarios"), json=data)
        r.raise_for_status()
        return r.json()

    # Rubrics
    def list_rubrics(self, **params: Any) -> dict[str, Any]:
        r = self.client.get(self._url("/rubrics"), params=params)
        r.raise_for_status()
        return r.json()

    def create_rubric(self, data: dict[str, Any]) -> dict[str, Any]:
        r = self.client.post(self._url("/rubrics"), json=data)
        r.raise_for_status()
        return r.json()

    # Eval Runs
    def list_eval_runs(self, **params: Any) -> dict[str, Any]:
        r = self.client.get(self._url("/eval-runs"), params=params)
        r.raise_for_status()
        return r.json()

    def create_eval_run(self, data: dict[str, Any]) -> dict[str, Any]:
        r = self.client.post(self._url("/eval-runs"), json=data)
        r.raise_for_status()
        return r.json()

    # Conversations
    def list_conversations(self, **params: Any) -> dict[str, Any]:
        r = self.client.get(self._url("/conversations"), params=params)
        r.raise_for_status()
        return r.json()

    def get_conversation(self, conv_id: str) -> dict[str, Any]:
        r = self.client.get(self._url(f"/conversations/{conv_id}"))
        r.raise_for_status()
        return r.json()

    def get_conversation_evaluations(self, conv_id: str) -> dict[str, Any]:
        r = self.client.get(self._url(f"/conversations/{conv_id}/evaluations"))
        r.raise_for_status()
        return r.json()

    def get_conversation_metrics(self, conv_id: str) -> dict[str, Any]:
        r = self.client.get(self._url(f"/conversations/{conv_id}/metrics"))
        r.raise_for_status()
        return r.json()

    def get_similar_conversations(
        self,
        conv_id: str,
        limit: int = 5,
        same_scenario: bool = False,
        min_score: float | None = None,
        max_score: float | None = None,
        score_evaluator: str = "model_judge",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "limit": limit,
            "same_scenario": str(same_scenario).lower(),
            "score_evaluator": score_evaluator,
        }
        if min_score is not None:
            params["min_score"] = min_score
        if max_score is not None:
            params["max_score"] = max_score
        r = self.client.get(self._url(f"/conversations/{conv_id}/similar"), params=params)
        r.raise_for_status()
        return r.json()

    # Evaluations
    def create_human_evaluation(self, data: dict[str, Any]) -> dict[str, Any]:
        r = self.client.post(self._url("/evaluations/human"), json=data)
        r.raise_for_status()
        return r.json()

    # Rankings
    def get_rankings(self, scenario_id: str | None = None) -> dict[str, Any]:
        params = {}
        if scenario_id:
            params["scenario_id"] = scenario_id
        r = self.client.get(self._url("/evaluations/rankings"), params=params)
        r.raise_for_status()
        return r.json()

    # Reliability
    def get_reliability(self, eval_run_id: str) -> dict[str, Any]:
        r = self.client.get(self._url("/evaluations/reliability"), params={"eval_run_id": eval_run_id})
        r.raise_for_status()
        return r.json()

    # Calibration
    def get_calibration(self, eval_run_id: str) -> dict[str, Any]:
        r = self.client.get(self._url("/evaluations/calibration"), params={"eval_run_id": eval_run_id})
        r.raise_for_status()
        return r.json()

    # Health
    def health(self) -> dict[str, Any]:
        r = self.client.get(self._url("/health/ready"))
        r.raise_for_status()
        return r.json()
