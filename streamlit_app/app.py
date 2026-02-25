import streamlit as st

st.set_page_config(
    page_title="AgentProbe",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AgentProbe")
st.subheader("Multi-Turn Agent Evaluation Platform")

st.markdown("""
Welcome to AgentProbe. Use the sidebar to navigate between pages:

1. **Eval Runs** — View, filter, and trigger evaluation runs
2. **Conversation Viewer** — Step through multi-turn conversations with scores
3. **Human Eval** — Submit human evaluation scores for conversations
4. **Rubric Editor** — Create, version, and edit evaluation rubrics
5. **Agent Configs** — Manage agent configurations (create, edit, deactivate)
6. **Scenarios** — Manage test scenarios (create, edit, deactivate)
7. **Agent Comparison** — Compare agent performance side-by-side with charts
8. **Metrics Dashboard** — Visualize automated metrics with histograms and correlations
9. **ELO Rankings** — Pairwise agent comparison with ELO ratings
10. **Calibration & Reliability** — Judge calibration curves and interrater agreement
""")

# API health check
from lib.api_client import AgentProbeClient

try:
    status = AgentProbeClient().health()
    st.success(f"API Status: {status.get('status', 'connected')}")
except Exception:
    st.warning("API is not reachable. Start the backend to enable all features.")
