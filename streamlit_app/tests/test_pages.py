"""Smoke tests for every Streamlit page.

streamlit.testing.v1.AppTest runs the script in this same process, so
the patched_client fixture (in conftest.py) replaces AgentProbeClient
with a fake before AppTest.run() — no real HTTP calls.

These tests catch the most common silent breakage: a Pydantic schema
changes shape and a page that reads response['something_renamed'] now
either errors or shows the wrong thing. We don't assert on rendered
content; we assert no uncaught exception.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


PAGES = [
    "app.py",
    "pages/01_Results.py",
    "pages/02_Conversations.py",
    "pages/03_Rate.py",
    "pages/04_Compare.py",
    "pages/05_Metrics.py",
    "pages/06_Admin.py",
    "pages/07_Advanced.py",
]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page, patched_client):
    """Each page must render against the fake API without raising."""
    at = AppTest.from_file(str(ROOT / page), default_timeout=15.0)
    at.run()
    assert not at.exception, (
        f"{page} raised: {[str(e) for e in at.exception]}"
    )


def test_results_page_disables_autorefresh_when_no_running_runs(patched_client):
    """The polling loop must not fire when all runs are terminal (otherwise the
    test would re-run forever)."""
    at = AppTest.from_file(str(ROOT / "pages/01_Results.py"), default_timeout=15.0)
    at.run()
    assert not at.exception
    # No running_simulation / running_evaluation in the fake data → no auto-refresh
    # checkbox is rendered.
    checkboxes = [c for c in at.checkbox if c.label.startswith("Auto-refresh")]
    assert checkboxes == []


def test_compare_page_renders_with_two_agents_in_fake(patched_client):
    """Compare page expects ≥2 agent configs — fake returns exactly two."""
    at = AppTest.from_file(str(ROOT / "pages/04_Compare.py"), default_timeout=15.0)
    at.run()
    assert not at.exception
