# AgentProbe: Multi-Turn Agent Evaluation Platform

## Problem

AI agents are being deployed in customer support, code assistance, and internal tooling. These agents hold multi-turn conversations, call external tools, and make decisions over several exchanges. Evaluating them is harder than evaluating a single-prompt model.

Standard benchmarks like MMLU or HumanEval test isolated capabilities. They do not measure whether an agent recovers from misunderstandings, calls the right tools in the right order, or maintains coherence across ten turns of conversation. A customer support agent might score well on individual responses but fail to resolve a multi-step issue.

Teams need a way to simulate realistic conversations at scale, evaluate them with multiple complementary strategies, and surface the results in a format that supports comparison and debugging. That is what AgentProbe does.

## What I Built

AgentProbe is a fully containerized evaluation platform with 9 services orchestrated through Docker Compose. It covers the full pipeline: scenario definition, conversation simulation, multi-strategy evaluation, score aggregation, and visualization.

Key numbers:
- 8 evaluation strategies (model-as-judge, rubric grading, reference-based, trajectory, pairwise, ELO, calibration, interrater reliability)
- 133 unit tests
- 10 Streamlit pages for interactive analysis
- 4 Grafana dashboards for observability
- 8 database tables
- 4 Kafka topics connecting 3 pipeline stages

## System Design

### Simulation Engine

The core of the platform is the ScenarioRunner. It orchestrates multi-turn conversations between two LLM instances: one acting as the user and one as the agent under test.

The user simulator takes a configurable persona that controls patience level, technical skill, and communication style. A frustrated non-technical user behaves very differently from a patient engineer, and the persona system captures that.

Tool calls from the agent are intercepted by a ToolSimulator. This component returns configurable responses, injects failures at a specified rate, and adds artificial latency. It lets you test how an agent handles tool timeouts or unexpected errors without connecting to real external services.

An adversarial injection system can insert prompt injection attempts at specified turns. This tests whether the agent leaks system prompt content or follows injected instructions.

### Evaluation Framework

The platform runs 8 evaluation strategies. Each one targets a different aspect of conversation quality.

**Model-as-Judge.** An LLM reads the full conversation and scores it across configurable dimensions (helpfulness, accuracy, safety, coherence). The judge uses Claude's `tool_use` feature to return structured scores. This is more reliable than asking the model to output JSON in its response text, because tool_use enforces the schema at the API level.

**Rubric Grading.** A deterministic evaluator that scores conversations against explicit criteria. Each rubric dimension has a list of criteria with associated score ranges. This provides consistent baselines that do not vary between runs.

**Reference-Based.** Compares agent responses to gold-standard answers using three metrics: ROUGE-1 (unigram overlap), ROUGE-L (longest common subsequence), and exact match. All three are implemented in pure Python with no NLP library dependencies. Scenario authors include expected responses in the scenario template, and the evaluator pairs them with actual agent responses.

**Trajectory.** Evaluates whether the agent called the right tools in the right order. The scenario defines an expected tool sequence (for example: search_knowledge_base, lookup_order, create_ticket). The evaluator computes precision (fraction of called tools that were correct), recall (fraction of expected tools that were called), and an order correlation score based on concordant pairs.

**Pairwise Judge.** An LLM compares two conversations side by side and declares a winner. To mitigate position bias (LLMs tend to favor whichever conversation is presented first), the system randomly swaps the A/B labels before presenting them to the judge and corrects the result afterward.

**ELO Rankings.** Pairwise comparison results feed into a standard ELO rating system (K-factor 32, initial rating 1500). Over many comparisons, agents converge to ratings that reflect their relative quality. This is the same approach used by Chatbot Arena for ranking LLMs.

**Calibration Analysis.** Measures agreement between the model judge and human evaluators. Computes Pearson correlation, Spearman rank correlation, mean absolute error, root mean square error, and systematic bias. A calibration curve bins model scores and compares the average model score per bin to the average human score. Perfect calibration means the two lines overlap.

**Interrater Reliability.** Implements Krippendorff's alpha for interval data. This measures whether multiple evaluators (human or automated) agree with each other beyond what chance would predict. An alpha of 1.0 means perfect agreement. Below 0.67 indicates the evaluators are not reliable enough to draw conclusions from.

### Event Pipeline

The pipeline has three stages connected by Kafka topics.

1. **Simulation** produces `ConversationCompletedEvent` when a conversation finishes.
2. **Evaluation** consumers pick up the event, run all applicable evaluators, and produce `EvaluationScoreCompletedEvent`.
3. **Aggregation** consumers compute run-level statistics and update the eval run status.

Each stage scales independently. If evaluation becomes the bottleneck, you add more evaluation consumers without changing the simulation code. Kafka also provides durability: if a consumer crashes, it resumes from where it left off.

Consumers are idempotent. Each event carries a unique ID, and Redis tracks which events have been processed. Duplicate delivery (which Kafka guarantees at-least-once) does not cause duplicate evaluations.

Failed events go to a dead letter topic for manual inspection.

### Data Model

The database has 8 tables: agent_configs, scenarios, rubrics, eval_runs, conversations, evaluations, metrics, and users.

Conversation turns are stored as a JSONB array. This matches the LLM message format directly, so loading a conversation is a single query with no joins. The alternative (a normalized turns table with a foreign key to conversations) would require N+1 queries or a complex join for every conversation load.

New evaluator types do not require schema migrations. Evaluation scores and metadata are stored in JSONB fields. When we added reference-based and trajectory evaluation, no Alembic migration was needed. The existing `Evaluation.metadata_` and `Scenario.constraints` fields absorbed the new data.

ChromaDB stores vector embeddings of conversation content. This supports semantic search: "find conversations similar to this failure case" is a useful query when you have thousands of evaluated conversations.

## Key Design Decisions

**Kafka + Celery instead of just one or the other.**
Celery handles task execution (running a simulation, scoring a conversation). Kafka handles event routing between pipeline stages. Using both means simulation workers do not need to know about evaluation, and evaluation does not need to know about aggregation. Each stage has a clear contract: produce an event when done, consume events from the previous stage.

**JSONB for conversation turns instead of a normalized turns table.**
Conversation turns are always loaded and stored as a complete array. There is no use case for querying a single turn independently. JSONB avoids the join overhead and keeps the conversation format identical to what the LLM API expects. The trade-off is that you cannot run SQL queries on individual turn content without JSONB operators, but PostgreSQL's JSONB support handles this well enough.

**Multiple evaluation strategies instead of just model-as-judge.**
Model-as-judge is fast and scalable, but it has known biases (verbosity bias, position bias, self-preference). Rubric grading provides consistency. Human evaluation provides ground truth. Reference-based and trajectory evaluation provide objective metrics that do not depend on LLM judgment at all. Using all of them together lets you cross-check results and identify when the judge is miscalibrated.

**Protocol-based architecture instead of inheritance.**
All major components (LLM client, evaluators, simulators) are defined as Python Protocols. This means tests can substitute lightweight mock implementations without inheriting from base classes or using monkey-patching. A mock LLM client that returns canned responses implements the same Protocol as the real one. No test-only flags, no conditional logic in production code.

**LiteLLM for model abstraction.**
LiteLLM wraps multiple LLM providers (Ollama, Anthropic, OpenAI, Google) behind a single interface. Switching from a local Ollama model to Claude requires changing one environment variable. No code changes, no conditional imports. The trade-off is an extra dependency, but it eliminates provider-specific API handling throughout the codebase.

**Grafana + Streamlit instead of a custom frontend.**
Grafana handles time-series dashboards and alerting out of the box. Streamlit keeps the entire codebase in Python. Neither requires writing JavaScript, CSS, or maintaining a build pipeline. The trade-off is limited customization compared to a React app, but for an evaluation tool used by engineers, the functionality is sufficient.

## Testing Strategy

The test suite has 133 unit tests covering all evaluators, the simulation engine, Kafka consumers, and score aggregation logic.

Every evaluator is tested against known inputs with deterministic expected outputs. The reference evaluator tests verify exact ROUGE-1 and ROUGE-L scores for specific string pairs. The trajectory evaluator tests verify precision, recall, and order scores for specific tool sequences. The ELO tests verify rating deltas for specific match outcomes.

LLM calls are mocked using protocol-based dependency injection. The test creates a mock object that implements `LLMClientProtocol` and returns a predetermined response. The evaluator receives this mock through its constructor. There is no monkey-patching, no `@mock.patch`, and no test-only code paths in production.

Kafka consumers are tested by calling the handler method directly with a constructed event envelope. The test verifies that the handler produces the correct database writes and downstream events.

## What I Would Change

**More integration test coverage.** The unit tests are thorough, but the integration tests that verify the full Kafka pipeline (API call triggers simulation, which triggers evaluation, which triggers aggregation) need more scenarios. Right now they cover the happy path but not edge cases like partial failures or consumer restarts.

**KRaft instead of Zookeeper.** The current Kafka setup uses Zookeeper for cluster coordination. Kafka has supported KRaft mode (removing the Zookeeper dependency) since version 3.3. Switching would simplify the Docker Compose setup from 9 services to 8 and reduce memory usage.

**A proper frontend for non-technical users.** Streamlit works well for engineers who want to explore data. If this tool needed to support product managers or QA analysts, a React frontend with better navigation, saved views, and role-based access would be worth the added complexity.

## Skills Demonstrated

- **Distributed systems**: Kafka event pipeline with 3 consumer types, Celery task queue, idempotent processing with Redis-backed deduplication, dead letter queue for failed events
- **Data modeling**: PostgreSQL schema design with JSONB trade-offs, Alembic migration management, 8-table relational model
- **API design**: RESTful FastAPI endpoints with Pydantic request/response validation, structured error handling, health check endpoints
- **Testing**: 133 unit tests, protocol-based mocking without monkey-patching, async test patterns with pytest-asyncio
- **Observability**: Structured logging with structlog, 4 Grafana dashboards, health checks across all services
- **Containerization**: Docker Compose orchestration of 9 services with health checks, volume management, and environment-based configuration
- **ML/AI engineering**: LLM-as-judge evaluation with structured output via tool_use, prompt engineering for user simulation, model-agnostic abstraction via LiteLLM, ELO ranking, calibration analysis
