"""Fixtures for the end-to-end test tier.

These tests assume the full docker-compose stack is already running on the
default ports — postgres, redis, kafka, chromadb, the api container, the
celery worker, and the kafka-consumer container — and that
AGENTPROBE_LLM_PROVIDER=fake is set on the api + worker so no real Vertex
calls happen.

CI starts the stack before running these. Locally:

    AGENTPROBE_LLM_PROVIDER=fake docker compose up -d
    make migrate
    pytest backend/tests/e2e -v -m e2e
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import httpx
import pytest


@pytest.fixture(scope="session")
def api_base_url() -> str:
    """Base URL of the running API container."""
    return os.getenv("AGENTPROBE_API_URL", "http://localhost:8080")


@pytest.fixture(scope="session")
def http(api_base_url: str) -> Iterator[httpx.Client]:
    """Sync httpx client; e2e tests use it for short polls and asserts."""
    with httpx.Client(base_url=api_base_url, timeout=60.0) as client:
        yield client
