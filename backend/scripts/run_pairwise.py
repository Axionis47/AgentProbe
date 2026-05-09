"""Run pairwise comparisons between agents on matching scenarios.

For each scenario, compares the first completed conversation from Agent A
with the first completed conversation from Agent B by POSTing to
/api/v1/evaluations/pairwise — the same endpoint the UI uses. The endpoint
runs the PairwiseJudgeEvaluator, persists Evaluation rows on both
conversations, and the rankings endpoint reads ELO from those.

Single source of truth for pairwise logic: this script and the UI both
call the API. Previously this script reimplemented the loop and could
drift from the API's behaviour.

Usage:
    docker compose exec api python -m scripts.run_pairwise
"""

from __future__ import annotations

import asyncio
import os

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.models.agent_config import AgentConfig
from app.models.conversation import Conversation
from app.models.eval_run import EvalRun
from app.models.scenario import Scenario


API_URL = os.getenv("AGENTPROBE_INTERNAL_API_URL", "http://api:8080")


async def _first_completed_conversation(
    session: AsyncSession, agent_id: str, scenario_id: str
) -> Conversation | None:
    run = (
        await session.execute(
            select(EvalRun).where(
                EvalRun.agent_config_id == agent_id,
                EvalRun.scenario_id == scenario_id,
                EvalRun.status == "completed",
            )
        )
    ).scalars().first()
    if run is None:
        return None
    return (
        await session.execute(
            select(Conversation)
            .where(
                Conversation.eval_run_id == run.id,
                Conversation.status == "completed",
            )
            .order_by(Conversation.sequence_num)
            .limit(1)
        )
    ).scalars().first()


async def main() -> None:
    engine = create_async_engine(settings.database_url)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    print("=" * 60)
    print("AgentProbe — Pairwise Comparisons (via API)")
    print("=" * 60)

    async with session_factory() as session:
        agents = (
            await session.execute(
                select(AgentConfig)
                .where(AgentConfig.is_active.is_(True))
                .order_by(AgentConfig.name)
            )
        ).scalars().all()
        scenarios = (
            await session.execute(select(Scenario).where(Scenario.is_active.is_(True)))
        ).scalars().all()

        if len(agents) < 2:
            print("Need at least 2 agents for pairwise comparison.")
            return

        agent_a, agent_b = agents[0], agents[1]
        print(f"Agent A: {agent_a.name}")
        print(f"Agent B: {agent_b.name}")
        print(f"Scenarios: {len(scenarios)}")

        # Pick a single active rubric to use for every comparison; if none exists,
        # the API falls back to DEFAULT_DIMENSIONS itself.
        from app.models.rubric import Rubric  # lazy import keeps top-level light
        rubric = (
            await session.execute(select(Rubric).where(Rubric.is_active.is_(True)))
        ).scalars().first()
        rubric_id = rubric.id if rubric else None

        comparison_count = 0
        async with httpx.AsyncClient(base_url=API_URL, timeout=120.0) as client:
            for scenario in scenarios:
                print(f"\n--- {scenario.name} ---")

                conv_a = await _first_completed_conversation(session, agent_a.id, scenario.id)
                conv_b = await _first_completed_conversation(session, agent_b.id, scenario.id)
                if not conv_a or not conv_b:
                    print("  Skipping — no completed conversations for one of the agents")
                    continue

                payload: dict = {
                    "conversation_id_a": conv_a.id,
                    "conversation_id_b": conv_b.id,
                }
                if rubric_id:
                    payload["rubric_id"] = rubric_id

                try:
                    resp = await client.post("/api/v1/evaluations/pairwise", json=payload)
                    resp.raise_for_status()
                    result = resp.json()
                except httpx.HTTPError as e:
                    print(f"  ERROR: {e}")
                    continue

                winner = result["winner"]
                winner_name = (
                    agent_a.name if winner == "a"
                    else agent_b.name if winner == "b"
                    else "Draw"
                )
                conf = result.get("confidence", 0.0)
                print(f"  Winner: {winner_name} (confidence: {conf:.0%})")
                if result.get("reasoning"):
                    print(f"  Reasoning: {result['reasoning'][:120]}...")
                comparison_count += 1

        # Read back ELO via the same API the UI uses.
        async with httpx.AsyncClient(base_url=API_URL, timeout=30.0) as client:
            try:
                rankings_resp = await client.get("/api/v1/evaluations/rankings")
                rankings_resp.raise_for_status()
                rankings = rankings_resp.json().get("rankings", [])
            except httpx.HTTPError:
                rankings = []

        if rankings:
            print("\n" + "=" * 60)
            print("ELO Rankings")
            print("=" * 60)
            for r in rankings:
                name = r.get("agent_name") or r.get("agent_config_id", "")[:8]
                print(
                    f"  {name:30s}  ELO: {r['elo_rating']:.0f}  "
                    f"W:{r['wins']} L:{r['losses']} D:{r['draws']}"
                )

        print(f"\nDone. {comparison_count} pairwise comparisons stored.")


if __name__ == "__main__":
    asyncio.run(main())
