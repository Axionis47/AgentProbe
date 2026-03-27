"""Print a terminal report of all evaluation results.

Usage:
    docker compose exec api python -m scripts.report
"""

from __future__ import annotations

import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.models.agent_config import AgentConfig
from app.models.conversation import Conversation
from app.models.eval_run import EvalRun
from app.models.evaluation import Evaluation
from app.models.scenario import Scenario


async def main() -> None:
    engine = create_async_engine(settings.database_url)
    sf = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with sf() as session:
        runs = (await session.execute(
            select(EvalRun).where(EvalRun.status == "completed").order_by(EvalRun.name)
        )).scalars().all()

        if not runs:
            print("No completed runs.")
            return

        agents = {}
        scenarios = {}
        for r in runs:
            if r.agent_config_id not in agents:
                a = (await session.execute(select(AgentConfig).where(AgentConfig.id == r.agent_config_id))).scalars().first()
                agents[r.agent_config_id] = a.name if a else "Unknown"
            if r.scenario_id not in scenarios:
                s = (await session.execute(select(Scenario).where(Scenario.id == r.scenario_id))).scalars().first()
                scenarios[r.scenario_id] = s.name if s else "Unknown"

        # Collect trajectory results
        traj_data = []
        for r in runs:
            agent_name = agents[r.agent_config_id]
            scenario_name = scenarios[r.scenario_id]
            convs = (await session.execute(
                select(Conversation).where(Conversation.eval_run_id == r.id, Conversation.status == "completed")
            )).scalars().all()

            for conv in convs:
                traj = (await session.execute(
                    select(Evaluation).where(
                        Evaluation.conversation_id == conv.id,
                        Evaluation.evaluator_type == "trajectory",
                    )
                )).scalars().first()

                if traj:
                    scores = traj.scores or {}
                    recall = scores.get("recall", 0)
                    precision = scores.get("precision", 0)
                    order = scores.get("order_score", 0)
                    passed = recall == 1.0 and precision >= 0.5 and order == 1.0

                    actual = ""
                    expected = ""
                    reasoning = traj.reasoning or ""
                    if "Actual tools:" in reasoning:
                        actual = reasoning.split("Actual tools:")[1].split(".")[0].strip()
                    if "Expected:" in reasoning:
                        expected = reasoning.split("Expected:")[1].split(".")[0].strip()

                    traj_data.append({
                        "agent": agent_name,
                        "scenario": scenario_name,
                        "expected": expected,
                        "actual": actual,
                        "passed": passed,
                        "recall": recall,
                        "precision": precision,
                        "order": order,
                    })

        # Print report
        print()
        print("=" * 70)
        print("  AgentProbe Report — Tool Calling Analysis")
        print("=" * 70)

        for agent_name in sorted(set(d["agent"] for d in traj_data)):
            agent_rows = [d for d in traj_data if d["agent"] == agent_name]
            total = len(agent_rows)
            passed = sum(1 for d in agent_rows if d["passed"])
            rate = (passed / total * 100) if total > 0 else 0

            print(f"\n  {agent_name}")
            print(f"  Pass rate: {passed}/{total} ({rate:.0f}%)")
            print(f"  {'─' * 66}")

            by_scenario = {}
            for d in agent_rows:
                by_scenario.setdefault(d["scenario"], []).append(d)

            for scenario, rows in sorted(by_scenario.items()):
                expected = rows[0]["expected"]
                s_passed = sum(1 for r in rows if r["passed"])
                s_total = len(rows)
                status = "PASS" if s_passed == s_total else "FAIL"
                print(f"  {scenario}")
                print(f"    Expected: {expected}")
                for i, r in enumerate(rows):
                    mark = "✓" if r["passed"] else "✗"
                    print(f"    Run {i+1}: {r['actual']:50s} {mark}")
                print(f"    Result: {s_passed}/{s_total} {status}")
                if s_passed < s_total:
                    failures = [r for r in rows if not r["passed"]]
                    for f in failures:
                        if f["recall"] < 1.0:
                            print(f"    → Missing tools (recall={f['recall']:.0%})")
                        if f["precision"] < 0.5:
                            print(f"    → Unnecessary tools (precision={f['precision']:.0%})")
                        if f["order"] < 1.0 and f["recall"] > 0:
                            print(f"    → Wrong order (order={f['order']:.0%})")
                print()

        # Overall model judge scores
        print("─" * 70)
        print("  Overall Quality Scores (AI Judge)")
        print("─" * 70)

        for r in runs:
            agent_name = agents[r.agent_config_id]
            scenario_name = scenarios[r.scenario_id]
            convs = (await session.execute(
                select(Conversation).where(Conversation.eval_run_id == r.id)
            )).scalars().all()

            judge_scores = []
            for conv in convs:
                judge = (await session.execute(
                    select(Evaluation).where(
                        Evaluation.conversation_id == conv.id,
                        Evaluation.evaluator_type == "model_judge",
                    )
                )).scalars().first()
                if judge and judge.overall_score:
                    judge_scores.append(judge.overall_score)

            if judge_scores:
                avg = sum(judge_scores) / len(judge_scores)
                mn = min(judge_scores)
                mx = max(judge_scores)
                print(f"  {agent_name:30s} / {scenario_name:40s}  {avg:.1f} (min={mn:.1f} max={mx:.1f})")

        print()
        print("=" * 70)
        print(f"  Dashboard: http://localhost:8501")
        print("=" * 70)
        print()


if __name__ == "__main__":
    asyncio.run(main())
