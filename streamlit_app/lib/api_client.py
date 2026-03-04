import os
from typing import Any

import httpx

API_URL = os.getenv("AGENTPROBE_API_URL", "http://localhost:8080")


class AgentProbeClient:
    """Typed HTTP client for the AgentProbe FastAPI backend."""

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

    def get_agent_config(self, config_id: str) -> dict[str, Any]:
        r = self.client.get(self._url(f"/agent-configs/{config_id}"))
        r.raise_for_status()
        return r.json()

    def update_agent_config(self, config_id: str, data: dict[str, Any]) -> dict[str, Any]:
        r = self.client.put(self._url(f"/agent-configs/{config_id}"), json=data)
        r.raise_for_status()
        return r.json()

    def delete_agent_config(self, config_id: str) -> None:
        r = self.client.delete(self._url(f"/agent-configs/{config_id}"))
        r.raise_for_status()

    # Scenarios
    def list_scenarios(self, **params: Any) -> dict[str, Any]:
        r = self.client.get(self._url("/scenarios"), params=params)
        r.raise_for_status()
        return r.json()

    def create_scenario(self, data: dict[str, Any]) -> dict[str, Any]:
        r = self.client.post(self._url("/scenarios"), json=data)
        r.raise_for_status()
        return r.json()

    def get_scenario(self, scenario_id: str) -> dict[str, Any]:
        r = self.client.get(self._url(f"/scenarios/{scenario_id}"))
        r.raise_for_status()
        return r.json()

    def update_scenario(self, scenario_id: str, data: dict[str, Any]) -> dict[str, Any]:
        r = self.client.put(self._url(f"/scenarios/{scenario_id}"), json=data)
        r.raise_for_status()
        return r.json()

    def delete_scenario(self, scenario_id: str) -> None:
        r = self.client.delete(self._url(f"/scenarios/{scenario_id}"))
        r.raise_for_status()

    # Rubrics
    def list_rubrics(self, **params: Any) -> dict[str, Any]:
        r = self.client.get(self._url("/rubrics"), params=params)
        r.raise_for_status()
        return r.json()

    def create_rubric(self, data: dict[str, Any]) -> dict[str, Any]:
        r = self.client.post(self._url("/rubrics"), json=data)
        r.raise_for_status()
        return r.json()

    def get_rubric(self, rubric_id: str) -> dict[str, Any]:
        r = self.client.get(self._url(f"/rubrics/{rubric_id}"))
        r.raise_for_status()
        return r.json()

    def update_rubric(self, rubric_id: str, data: dict[str, Any]) -> dict[str, Any]:
        r = self.client.put(self._url(f"/rubrics/{rubric_id}"), json=data)
        r.raise_for_status()
        return r.json()

    def get_rubric_versions(self, rubric_id: str) -> list[dict[str, Any]]:
        r = self.client.get(self._url(f"/rubrics/{rubric_id}/versions"))
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

    def get_eval_run(self, run_id: str) -> dict[str, Any]:
        r = self.client.get(self._url(f"/eval-runs/{run_id}"))
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

    # Evaluations
    def create_human_evaluation(self, data: dict[str, Any]) -> dict[str, Any]:
        r = self.client.post(self._url("/evaluations/human"), json=data)
        r.raise_for_status()
        return r.json()

    # Pairwise Comparison
    def create_pairwise_comparison(self, data: dict[str, Any]) -> dict[str, Any]:
        r = self.client.post(self._url("/evaluations/pairwise"), json=data)
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

    # Search
    def semantic_search(self, query: str, **filters: Any) -> dict[str, Any]:
        r = self.client.post(self._url("/search/semantic"), json={"query": query, **filters})
        r.raise_for_status()
        return r.json()

    # Health
    def health(self) -> dict[str, Any]:
        r = self.client.get(self._url("/health/ready"))
        r.raise_for_status()
        return r.json()
