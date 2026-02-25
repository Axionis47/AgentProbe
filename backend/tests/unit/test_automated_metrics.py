"""Unit tests for automated metrics calculator."""

from app.evaluation.automated_metrics import AutomatedMetricsCalculator


def _make_turns(
    assistant_count: int = 3,
    latency: int = 100,
    with_tools: bool = False,
) -> list[dict]:
    turns = []
    for i in range(assistant_count):
        turns.append({"role": "user", "content": f"User message {i}"})

        assistant_turn: dict = {
            "role": "assistant",
            "content": f"Assistant response {i}",
            "latency_ms": latency,
            "input_tokens": 10,
            "output_tokens": 20,
        }

        if with_tools and i == 0:
            assistant_turn["tool_calls"] = [
                {"id": "call_1", "name": "search", "arguments": {"q": "test"}},
            ]
            assistant_turn["tool_results"] = [
                {"tool_call_id": "call_1", "content": "result", "is_error": False},
            ]

        turns.append(assistant_turn)
    return turns


class TestAutomatedMetrics:

    def test_compute_all_returns_expected_metrics(self) -> None:
        calc = AutomatedMetricsCalculator()
        turns = _make_turns()
        metrics = calc.compute_all(
            turns=turns, turn_count=3, total_tokens=90,
            total_input_tokens=30, total_output_tokens=60,
            total_latency_ms=300, status="completed",
        )
        names = {m.name for m in metrics}
        assert names == {
            "tokens_per_turn", "output_input_ratio", "avg_latency_ms",
            "p95_latency_ms", "turns_to_resolution", "conversation_completed",
            "tool_call_count", "tool_success_rate",
        }

    def test_token_metrics(self) -> None:
        calc = AutomatedMetricsCalculator()
        metrics = calc.compute_all(
            turns=_make_turns(), turn_count=5, total_tokens=100,
            total_input_tokens=40, total_output_tokens=60,
            total_latency_ms=500, status="completed",
        )
        by_name = {m.name: m for m in metrics}
        assert by_name["tokens_per_turn"].value == 20.0  # 100 / 5
        assert by_name["output_input_ratio"].value == 1.5  # 60 / 40

    def test_tool_metrics_with_tools(self) -> None:
        calc = AutomatedMetricsCalculator()
        turns = _make_turns(with_tools=True)
        metrics = calc.compute_all(
            turns=turns, turn_count=3, total_tokens=90,
            total_input_tokens=30, total_output_tokens=60,
            total_latency_ms=300, status="completed",
        )
        by_name = {m.name: m for m in metrics}
        assert by_name["tool_call_count"].value == 1.0
        assert by_name["tool_success_rate"].value == 1.0

    def test_resolution_metrics(self) -> None:
        calc = AutomatedMetricsCalculator()
        metrics_completed = calc.compute_all(
            turns=[], turn_count=3, total_tokens=0,
            total_input_tokens=0, total_output_tokens=0,
            total_latency_ms=0, status="goal_achieved",
        )
        metrics_failed = calc.compute_all(
            turns=[], turn_count=3, total_tokens=0,
            total_input_tokens=0, total_output_tokens=0,
            total_latency_ms=0, status="failed",
        )
        completed_map = {m.name: m for m in metrics_completed}
        failed_map = {m.name: m for m in metrics_failed}

        assert completed_map["conversation_completed"].value == 1.0
        assert failed_map["conversation_completed"].value == 0.0
