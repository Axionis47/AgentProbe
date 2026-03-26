"""Run pairwise comparisons between agents on matching scenarios.

For each scenario, compares conversations from Agent A vs Agent B.
Stores pairwise evaluations and computes ELO rankings.

Usage:
    docker compose exec api python -m scripts.run_pairwise
"""

from __future__ import annotations

import asyncio
import json

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.engine.llm_client import LLMClient
from app.evaluation.elo import compute_rankings
from app.evaluation.pairwise_judge import PairwiseJudgeEvaluator
from app.evaluation.types import DEFAULT_DIMENSIONS, RubricDimension
from app.models.agent_config import AgentConfig
from app.models.conversation import Conversation
from app.models.eval_run import EvalRun
from app.models.evaluation import Evaluation
from app.models.rubric import Rubric
from app.models.scenario import Scenario

logger = structlog.get_logger()


async def main() -> None:
    engine = create_async_engine(settings.database_url)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    print("=" * 60)
    print("AgentProbe — Pairwise Comparisons")
    print("=" * 60)

    async with session_factory() as session:
        # Load agents and scenarios
        agents = (await session.execute(
            select(AgentConfig).where(AgentConfig.is_active == True).order_by(AgentConfig.name)
        )).scalars().all()
        scenarios = (await session.execute(
            select(Scenario).where(Scenario.is_active == True)
        )).scalars().all()

        if len(agents) < 2:
            print("Need at least 2 agents for pairwise comparison.")
            return

        agent_a, agent_b = agents[0], agents[1]
        print(f"Agent A: {agent_a.name}")
        print(f"Agent B: {agent_b.name}")
        print(f"Scenarios: {len(scenarios)}")

        # Load rubric
        rubric = (await session.execute(
            select(Rubric).where(Rubric.is_active == True)
        )).scalars().first()

        if rubric and rubric.dimensions:
            dimensions = [
                RubricDimension(
                    name=d["name"], description=d.get("description", ""),
                    weight=d.get("weight", 1.0), criteria=d.get("criteria", []),
                )
                for d in rubric.dimensions
            ]
        else:
            dimensions = DEFAULT_DIMENSIONS

        llm_client = LLMClient()
        judge = PairwiseJudgeEvaluator(llm_client=llm_client)

        match_results = []
        comparison_count = 0

        for scenario in scenarios:
            print(f"\n--- {scenario.name} ---")

            # Get runs for each agent on this scenario
            run_a = (await session.execute(
                select(EvalRun).where(
                    EvalRun.agent_config_id == agent_a.id,
                    EvalRun.scenario_id == scenario.id,
                    EvalRun.status == "completed",
                )
            )).scalars().first()

            run_b = (await session.execute(
                select(EvalRun).where(
                    EvalRun.agent_config_id == agent_b.id,
                    EvalRun.scenario_id == scenario.id,
                    EvalRun.status == "completed",
                )
            )).scalars().first()

            if not run_a or not run_b:
                print("  Skipping — missing completed run for one agent")
                continue

            # Get first conversation from each
            conv_a = (await session.execute(
                select(Conversation).where(
                    Conversation.eval_run_id == run_a.id,
                    Conversation.status == "completed",
                ).order_by(Conversation.sequence_num).limit(1)
            )).scalars().first()

            conv_b = (await session.execute(
                select(Conversation).where(
                    Conversation.eval_run_id == run_b.id,
                    Conversation.status == "completed",
                ).order_by(Conversation.sequence_num).limit(1)
            )).scalars().first()

            if not conv_a or not conv_b:
                print("  Skipping — no completed conversations")
                continue

            # Run pairwise comparison
            try:
                result = await judge.compare(
                    turns_a=conv_a.turns or [],
                    turns_b=conv_b.turns or [],
                    rubric_dimensions=dimensions,
                )

                winner_label = {
                    "a": agent_a.name,
                    "b": agent_b.name,
                    "draw": "Draw",
                }
                print(f"  Winner: {winner_label.get(result.winner, result.winner)} (confidence: {result.confidence:.0%})")
                print(f"  Reasoning: {result.reasoning[:120]}...")

                # Store as evaluation on both conversations
                for conv, label in [(conv_a, "a"), (conv_b, "b")]:
                    eval_record = Evaluation(
                        conversation_id=conv.id,
                        evaluator_type="pairwise_judge",
                        scores={
                            "winner": result.winner,
                            "confidence": result.confidence,
                            "this_agent": label,
                            **{f"{k}_pref": v for k, v in result.dimension_preferences.items()},
                        },
                        overall_score=10.0 if result.winner == label else (5.0 if result.winner == "draw" else 2.0),
                        reasoning=result.reasoning,
                        metadata_={
                            "match_id": result.match_id,
                            "opponent_conversation_id": conv_b.id if label == "a" else conv_a.id,
                            "swapped": result.metadata.get("swapped", False),
                        },
                    )
                    session.add(eval_record)

                # Record match for ELO
                elo_result = "draw"
                if result.winner == "a":
                    elo_result = "a_wins"
                elif result.winner == "b":
                    elo_result = "b_wins"

                match_results.append({
                    "agent_config_id_a": agent_a.id,
                    "agent_config_id_b": agent_b.id,
                    "result": elo_result,
                    "scenario": scenario.name,
                })
                comparison_count += 1

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        await session.commit()

        # Compute ELO
        if match_results:
            print("\n" + "=" * 60)
            print("ELO Rankings")
            print("=" * 60)
            ratings = compute_rankings(match_results)
            agent_names = {agent_a.id: agent_a.name, agent_b.id: agent_b.name}
            for agent_id, rating in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
                name = agent_names.get(agent_id, agent_id)
                wins = sum(1 for m in match_results if
                           (m["agent_config_id_a"] == agent_id and m["result"] == "a_wins") or
                           (m["agent_config_id_b"] == agent_id and m["result"] == "b_wins"))
                losses = sum(1 for m in match_results if
                             (m["agent_config_id_a"] == agent_id and m["result"] == "b_wins") or
                             (m["agent_config_id_b"] == agent_id and m["result"] == "a_wins"))
                draws = sum(1 for m in match_results if m["result"] == "draw")
                print(f"  {name:30s}  ELO: {rating:.0f}  W:{wins} L:{losses} D:{draws}")

        print(f"\nDone. {comparison_count} pairwise comparisons stored.")


if __name__ == "__main__":
    asyncio.run(main())
