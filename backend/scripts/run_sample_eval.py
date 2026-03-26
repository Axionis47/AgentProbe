"""Run a sample evaluation across all seeded agents and scenarios.

Usage:
    docker compose exec api python -m scripts.run_sample_eval

Creates eval runs (agent x scenario x 3 conversations), triggers Celery
simulation tasks, polls for completion, and prints a summary.
"""

from __future__ import annotations

import asyncio
import sys
import time

from sqlalchemy import select

from app.config import settings
from app.db.session import async_session_factory
from app.models.agent_config import AgentConfig
from app.models.eval_run import EvalRun
from app.models.rubric import Rubric
from app.models.scenario import Scenario
from app.workers.simulation_tasks import run_simulation

NUM_CONVERSATIONS_PER_RUN = 3
POLL_INTERVAL_SECONDS = 10
TIMEOUT_SECONDS = 600  # 10 minutes

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "running_evaluation"})


async def create_eval_runs() -> list[str]:
    """Create eval runs for every (agent, scenario) pair and return their IDs."""
    async with async_session_factory() as session:
        # Load agents
        result = await session.execute(
            select(AgentConfig).where(AgentConfig.is_active.is_(True))
        )
        agents = list(result.scalars().all())
        if not agents:
            print("ERROR: No agent configs found. Run `make seed` first.", file=sys.stderr)
            sys.exit(1)

        # Load scenarios
        result = await session.execute(
            select(Scenario).where(Scenario.is_active.is_(True))
        )
        scenarios = list(result.scalars().all())
        if not scenarios:
            print("ERROR: No scenarios found. Run `make seed` first.", file=sys.stderr)
            sys.exit(1)

        # Load rubric (use the first active one)
        result = await session.execute(
            select(Rubric).where(Rubric.is_active.is_(True)).limit(1)
        )
        rubric = result.scalar_one_or_none()

        rubric_id = rubric.id if rubric else None

        print(f"Agents:    {len(agents)}")
        print(f"Scenarios: {len(scenarios)}")
        print(f"Rubric:    {rubric.name if rubric else '(none)'}")
        print(f"Conversations per run: {NUM_CONVERSATIONS_PER_RUN}")
        print(f"Total eval runs: {len(agents) * len(scenarios)}")
        print(f"Total conversations: {len(agents) * len(scenarios) * NUM_CONVERSATIONS_PER_RUN}")
        print()

        eval_run_ids: list[str] = []

        for agent in agents:
            for scenario in scenarios:
                run_name = f"{agent.name} / {scenario.name}"
                eval_run = EvalRun(
                    name=run_name,
                    agent_config_id=agent.id,
                    scenario_id=scenario.id,
                    rubric_id=rubric_id,
                    num_conversations=NUM_CONVERSATIONS_PER_RUN,
                    config={},
                    status="pending",
                )
                session.add(eval_run)
                await session.flush()
                eval_run_ids.append(eval_run.id)
                print(f"  Created eval run: {run_name} (id={eval_run.id})")

        await session.commit()

    return eval_run_ids


def dispatch_tasks(eval_run_ids: list[str]) -> None:
    """Send simulation tasks to Celery."""
    print()
    print("Dispatching Celery simulation tasks...")
    for run_id in eval_run_ids:
        run_simulation.delay(run_id)
        print(f"  Dispatched: {run_id}")
    print(f"  {len(eval_run_ids)} task(s) dispatched.")


async def poll_until_done(eval_run_ids: list[str]) -> dict[str, EvalRun]:
    """Poll eval run statuses until all are terminal or timeout."""
    print()
    print(f"Polling for completion (timeout={TIMEOUT_SECONDS}s, interval={POLL_INTERVAL_SECONDS}s)...")
    start = time.monotonic()

    while True:
        elapsed = time.monotonic() - start
        if elapsed > TIMEOUT_SECONDS:
            print(f"\nTIMEOUT after {TIMEOUT_SECONDS}s — some runs may still be in progress.")
            break

        async with async_session_factory() as session:
            result = await session.execute(
                select(EvalRun).where(EvalRun.id.in_(eval_run_ids))
            )
            runs = {r.id: r for r in result.scalars().all()}

        statuses = {r.status for r in runs.values()}
        pending_count = sum(1 for r in runs.values() if r.status not in TERMINAL_STATUSES)

        elapsed_int = int(elapsed)
        status_summary = ", ".join(f"{s}={sum(1 for r in runs.values() if r.status == s)}" for s in sorted(statuses))
        print(f"  [{elapsed_int:>4d}s] {status_summary}")

        if pending_count == 0:
            print("\nAll eval runs reached a terminal status.")
            break

        await asyncio.sleep(POLL_INTERVAL_SECONDS)

    # Final read with relationships
    async with async_session_factory() as session:
        result = await session.execute(
            select(EvalRun).where(EvalRun.id.in_(eval_run_ids))
        )
        return {r.id: r for r in result.scalars().all()}


def print_summary(runs: dict[str, EvalRun]) -> None:
    """Print a summary table of eval run results."""
    print()
    print("=" * 90)
    print("EVALUATION SUMMARY")
    print("=" * 90)
    print(
        f"{'Run Name':<45} {'Status':<22} {'Convos':<8} {'Run ID'}"
    )
    print("-" * 90)

    for run in sorted(runs.values(), key=lambda r: r.name or ""):
        name = (run.name or "(unnamed)")[:44]
        status = run.status
        convos = run.num_conversations
        run_id = run.id[:12] + "..."
        print(f"{name:<45} {status:<22} {convos:<8} {run_id}")

        if run.error_message:
            print(f"  ERROR: {run.error_message[:80]}")

    print("-" * 90)

    completed = sum(1 for r in runs.values() if r.status in ("completed", "running_evaluation"))
    failed = sum(1 for r in runs.values() if r.status == "failed")
    other = len(runs) - completed - failed

    print(f"Completed: {completed}  |  Failed: {failed}  |  Other: {other}  |  Total: {len(runs)}")
    print()

    # Streamlit URLs
    print("=" * 90)
    print("VIEW RESULTS")
    print("=" * 90)
    print(f"  Streamlit Dashboard: http://localhost:8501")
    print(f"  API (eval runs):     http://localhost:8080/api/v1/eval-runs")
    for run_id in runs:
        print(f"  Run detail:          http://localhost:8080/api/v1/eval-runs/{run_id}")
    print()


def main() -> None:
    print("=" * 60)
    print("AgentProbe — Demo Evaluation Runner")
    print("=" * 60)
    print(f"Database: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")
    print()

    try:
        # Step 1: Create eval runs
        eval_run_ids = asyncio.run(create_eval_runs())

        # Step 2: Dispatch Celery tasks
        dispatch_tasks(eval_run_ids)

        # Step 3: Poll for completion
        runs = asyncio.run(poll_until_done(eval_run_ids))

        # Step 4: Print summary
        print_summary(runs)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
