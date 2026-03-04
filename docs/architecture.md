# AgentProbe — System Architecture

## Overview

AgentProbe is a multi-turn agent evaluation platform that simulates AI agent conversations, evaluates them using multiple strategies, processes results through event-driven pipelines, and visualizes insights via Grafana dashboards and a Streamlit interactive UI.

## System Architecture Diagram

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

## Services (9 total)

| Service | Port | Purpose |
|---------|------|---------|
| **api** (FastAPI) | 8080 | REST API + WebSocket for all backend operations |
| **worker** (Celery) | — | Async task execution: simulation, evaluation, embedding |
| **kafka-consumer** | — | Event pipeline consumers: conversation, evaluation, metrics |
| **streamlit** | 8501 | Interactive UI: conversation viewer, rubric editor, search |
| **grafana** | 3001 | Metrics dashboards: scores, trends, pipeline health |
| **postgres** | 5432 | Primary structured data store (8 tables) |
| **redis** | 6379 | Job broker, cache, idempotency tracking |
| **kafka** (+zookeeper) | 9092 | Event streaming for pipeline decoupling |
| **chromadb** | 8001 | Vector store for semantic search on conversations |

## Data Flow

### 1. Simulation Flow
```
User triggers eval run via Streamlit/API
  → FastAPI creates eval_run record (PostgreSQL)
  → Celery task enqueued (Redis broker)
  → Worker executes ScenarioRunner:
      - Claude API call as simulated user (with persona)
      - Claude API call as agent under test
      - Tool calls intercepted by ToolSimulator
      - Loop until termination condition
  → Conversations stored in PostgreSQL
  → ConversationCompletedEvent published to Kafka
```

### 2. Evaluation Flow
```
Kafka consumer receives ConversationCompletedEvent
  → Enqueues Celery evaluation tasks (fan-out per conversation)
  → Worker runs evaluation strategies in parallel:
      - Model-as-Judge (Claude API with tool_use for structured output)
      - Rubric-based grading (Claude API with rubric criteria)
      - Automated metrics (pure Python computation)
  → Evaluation scores stored in PostgreSQL
  → Conversation embeddings generated and stored in ChromaDB
  → EvaluationScoreCompletedEvent published to Kafka
```

### 3. Aggregation Flow
```
Kafka consumer receives EvaluationScoreCompletedEvent
  → Aggregates scores across conversations in the run
  → Computes weighted averages, z-score calibration
  → MetricsAggregatedEvent published to Kafka
  → Eval run status updated to "completed"
```

### 4. Visualization Flow
```
Grafana queries PostgreSQL directly via SQL for charts/dashboards
Streamlit calls FastAPI REST endpoints for interactive features
WebSocket provides real-time eval progress updates
```

## Key Design Decisions

### Why Kafka (not just Celery)?
Celery handles task execution (simulation, evaluation). Kafka handles event-driven decoupling between pipeline stages. This means simulation, evaluation, and aggregation can scale independently. If evaluation becomes the bottleneck, add evaluation consumers without touching simulation code.

### Why JSONB for conversation turns?
Conversation turns match Claude's message format exactly. Storing as JSONB means single-query conversation loading and native PostgreSQL JSONB operators for analysis. A normalized turns table would require N+1 queries or complex joins.

### Why model-as-judge + rubric + human evaluation?
No single evaluation method is reliable. Model-as-judge scales but has biases. Rubric grading is consistent but misses nuance. Human evaluation is ground truth but doesn't scale. Using all three enables inter-rater reliability analysis and catches miscalibrated judges.

### Why Grafana + Streamlit (not custom frontend)?
Grafana is what infrastructure teams use for observability. Streamlit keeps the entire project in Python. Both are Docker containers requiring zero custom frontend code. This demonstrates pragmatism over over-engineering.

### Why ChromaDB for vector search?
When you have thousands of evaluated conversations, the most useful query isn't "show me run #47" — it's "find conversations similar to this failure case." Vector search on conversation content makes that possible with metadata filtering.
