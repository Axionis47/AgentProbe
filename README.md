# AgentProbe

AgentProbe is a platform for evaluating multi-turn AI agent conversations. It simulates realistic user interactions, runs multiple evaluation strategies on the resulting conversations, and visualizes the results through dashboards and an interactive UI. The system is fully containerized (9 services) and uses an event-driven pipeline to decouple simulation, evaluation, and aggregation.

## Architecture

```
                                    +-----------------+
                                    |   Streamlit UI  |
                                    |   (port 8501)   |
                                    +--------+--------+
                                             |
                                             | HTTP
                                             v
+------------------+              +----------+---------+              +------------------+
|   Grafana        | <---SQL----> |   FastAPI Backend  | <--broker--> |  Celery Workers   |
|   (port 3001)    |              |   (port 8080)      |              |  (simulation +    |
+--------+---------+              +----+-----+----+----+              |   evaluation)     |
         |                             |     |    |                   +--------+----------+
         |                             |     |    |                            |
         |                             |     |    +---> Redis (6379)           |
         |                             |     |         - Job queue             |
         |                             |     |         - Cache                 |
         |                             |     |         - Idempotency keys      |
         |                             |     |                                 |
         +-----------------------------+     |                                 |
                       |                     |                                 |
                       v                     v                                 v
              +--------+--------+   +--------+--------+              +--------+---------+
              |   PostgreSQL    |   |    ChromaDB      |              |      Kafka       |
              |   (port 5432)   |   |   (port 8001)    |              |   (port 9092)    |
              |                 |   |                   |              |                  |
              |  - agent_configs|   |  - conversations  |              | Topics:          |
              |  - scenarios    |   |    (embeddings)   |              |  - conversation. |
              |  - rubrics      |   |                   |              |    completed     |
              |  - eval_runs    |   +-------------------+              |  - evaluation.   |
              |  - conversations|                                      |    completed     |
              |  - evaluations  |                                      |  - metrics.      |
              |  - metrics      |         +------------------+         |    aggregated    |
              |  - users        |         | Kafka Consumers  |<------->|  - pipeline.     |
              +-----------------+         | (3 consumers)    |         |    errors        |
                                          +------------------+         +------------------+
```

**Data flow:** A user triggers an eval run through the API or Streamlit. Celery workers simulate multi-turn conversations using configurable personas and tool calls. Completed conversations are published to Kafka, which fans out to evaluation consumers. Evaluators score the conversations, store results in PostgreSQL, and generate embeddings in ChromaDB. Grafana and Streamlit read from the database to display metrics, trends, and conversation details.

## Features

**Simulation**
- Multi-turn conversation orchestration between a simulated user and the agent under test
- LLM-powered user simulator with configurable persona (patience, technical skill, communication style)
- Tool call interception with configurable responses, failure injection, and latency simulation
- Adversarial prompt injection at specified turns

**Evaluation (8 evaluators)**
- Model-as-Judge: LLM scores conversations using structured `tool_use` output
- Rubric Grading: Deterministic scoring against explicit criteria per dimension
- Reference-Based: ROUGE-1, ROUGE-L, exact match against gold-standard answers
- Trajectory: Compares tool call sequences against expected sequences (precision, recall, order correlation)
- Pairwise Judge: LLM compares two conversations head-to-head with position-bias mitigation
- ELO Rankings: Standard ELO rating system for pairwise comparison results
- Calibration Analysis: Pearson/Spearman correlation, MAE, RMSE between model and human scores
- Interrater Reliability: Krippendorff's alpha for measuring agreement across evaluators

**Analysis**
- Automated metrics: token usage, latency, tool call counts, resolution rates
- Score aggregation with weighted dimension averages and z-score calibration
- Calibration curves comparing model judge scores to human ground truth
- Semantic search over conversation content via vector embeddings

**Infrastructure**
- Kafka event pipeline with idempotent consumers and dead letter queue
- Celery async task queue for simulation and evaluation workloads
- Protocol-based architecture for testability (all LLM calls are mockable without monkey-patching)
- Structured logging via structlog

## Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| API | FastAPI | Async endpoints, Pydantic validation |
| Database | PostgreSQL 16 | JSONB for conversation turns, 8 tables |
| Task Queue | Celery + Redis | Async simulation and evaluation tasks |
| Event Streaming | Kafka | Decouples pipeline stages, 4 topics |
| Vector Search | ChromaDB | Semantic search on conversation content |
| LLM | LiteLLM | Model-agnostic (Ollama, Claude, OpenAI) |
| UI | Streamlit | 10 interactive pages |
| Dashboards | Grafana | 4 observability dashboards |
| Containers | Docker Compose | 9 services with health checks |

## Quick Start

```bash
# Clone and configure
git clone https://github.com/your-username/agentprobe.git
cd agentprobe
cp .env.example .env

# Start all 9 services
make up

# Wait for services to be healthy
make health

# Seed sample data
make seed

# Run a sample evaluation
make demo
```

Once running:
- API: http://localhost:8080/api/v1/health
- Streamlit UI: http://localhost:8501
- Grafana: http://localhost:3001 (admin / agentprobe)

The default configuration uses Ollama for LLM calls. Install Ollama locally and pull the models specified in `.env.example`, or switch `AGENTPROBE_LLM_PROVIDER` to `anthropic` or `openai` and provide the appropriate API key.

## Project Structure

```
agentprobe/
  backend/
    app/
      api/v1/            # REST endpoints (6 route modules)
      engine/            # Simulation engine (scenario runner, user/tool simulators)
      evaluation/        # 8 evaluators + aggregation + types
      models/            # SQLAlchemy ORM models (8 tables)
      schemas/           # Pydantic request/response schemas
      services/          # Business logic (simulation, evaluation orchestration)
      pipeline/          # Kafka producers, consumers, event types
      workers/           # Celery task definitions
      core/              # Logging, middleware, exceptions
    tests/
      unit/              # 133 tests
      integration/
      e2e/
    alembic/             # Database migrations
  streamlit_app/
    pages/               # 10 Streamlit pages
    lib/api_client.py    # HTTP client for the backend API
  grafana/
    dashboards/          # 4 JSON dashboard definitions
    provisioning/        # Grafana datasource and dashboard provisioning
  scripts/               # Seed data and sample evaluation scripts
  docker-compose.yml     # 9 services
  Makefile               # Developer commands
```

## Testing

The suite splits into four tiers. Each one has a different cost / signal trade-off:

```bash
make test            # unit tests — fast, mocked DB/LLM, run on every push
make test-integration # integration tests — fast, mocked services, run in CI
make test-e2e        # end-to-end against a real docker-compose stack
make test-frontend   # Streamlit page smokes via streamlit.testing.v1.AppTest
make test-all        # everything inside the api container with coverage
```

**Unit (`backend/tests/unit/`)** — 330+ tests. Cover every API route handler,
the LLM client and embedding wrapper, all five evaluators, the engine
(scenario_runner, user_simulator, tool_simulator, adversarial), the Kafka
consumers, the Celery tasks, and the schemas. DB is mocked at the function
level, LLM calls go through a protocol so no external services are touched.
This is what every push runs.

**Integration (`backend/tests/integration/`)** — checks orchestration
between services with mocks: `AgentSimulationService` + `EvaluationService`
working together, status transitions, conversation persistence shape.
Still no real DB.

**End-to-end (`backend/tests/e2e/`)** — single test that boots the full
docker-compose stack (postgres, redis, kafka, chromadb, api, worker,
kafka-consumer) and runs an actual evaluation through the pipeline. Uses
the deterministic `FakeLLMClient` (selected via `AGENTPROBE_LLM_PROVIDER=fake`)
so the test is reproducible and runs without a Vertex / Anthropic / OpenAI
key. Marked with `@pytest.mark.e2e` so the regular `make test` skips it.

To run e2e locally:
```bash
AGENTPROBE_LLM_PROVIDER=fake docker compose up -d
make migrate
make test-e2e
```

**Frontend (`streamlit_app/tests/`)** — smoke tests for every Streamlit page
using `streamlit.testing.v1.AppTest`. The `AgentProbeClient` is patched
with a fake at module-import time so no real HTTP calls happen — these
exist to catch the schema-vs-page-code drift that's the most common silent
breakage between commits.

**CI** — every push hits the `backend` job (lint with real-bug ruff rules,
unit + integration tests against a fresh postgres + redis service, with
`AGENTPROBE_LLM_PROVIDER=fake`). PRs additionally trigger the `frontend`
job. The mypy --strict gate and full e2e job are deliberately kept off
master CI today; both need their own focused passes first.

**Pre-commit** (optional but recommended):
```bash
pip install pre-commit
pre-commit install
```
Runs the real-bug ruff rules + trailing whitespace + a `.DS_Store` guard
on every commit. Heavier checks stay in CI so commits stay fast.

## Streamlit Pages

1. **Eval Runs** - View, filter, and trigger evaluation runs
2. **Conversation Viewer** - Step through multi-turn conversations with evaluation scores
3. **Human Eval** - Submit human evaluation scores for conversations
4. **Rubric Editor** - Create, version, and edit evaluation rubrics
5. **Agent Configs** - Manage agent configurations (model, temperature, system prompt)
6. **Scenarios** - Manage test scenarios with reference answers and expected tool sequences
7. **Agent Comparison** - Compare agent performance side-by-side with bar, radar, and violin charts
8. **Metrics Dashboard** - Visualize automated metrics with histograms and correlation heatmaps
9. **ELO Rankings** - Pairwise agent comparison with ELO ratings and win/loss breakdowns
10. **Calibration & Reliability** - Judge calibration curves and interrater agreement analysis

## Grafana Dashboards

1. **Overview** - High-level metrics across all evaluation runs
2. **Evaluation Deep Dive** - Per-dimension score distributions and trends
3. **Performance & Latency** - Response times, token usage, and throughput
4. **Pipeline Health** - Kafka consumer lag, error rates, and processing times
