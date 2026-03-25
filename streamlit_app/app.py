import streamlit as st

st.set_page_config(
    page_title="AgentProbe",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AgentProbe")
st.subheader("Multi-Turn Agent Evaluation Platform")

from lib.api_client import AgentProbeClient

client = AgentProbeClient()

# --- API Health Check ---
api_ok = False
try:
    status = client.health()
    st.success(f"API Status: {status.get('status', 'connected')}")
    api_ok = True
except Exception:
    st.warning("API is not reachable. Start the backend to enable all features.")

# --- Quick Stats ---
if api_ok:
    st.divider()
    st.subheader("Quick Stats")
    try:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            configs = client.list_agent_configs(limit=1)
            st.metric("Agent Configs", configs.get("total", 0))
        with col2:
            scenarios = client.list_scenarios(limit=1)
            st.metric("Scenarios", scenarios.get("total", 0))
        with col3:
            runs = client.list_eval_runs(limit=1)
            st.metric("Eval Runs", runs.get("total", 0))
        with col4:
            convs = client.list_conversations(limit=1)
            st.metric("Conversations", convs.get("total", 0))
    except Exception:
        st.info("Could not load stats. The API may still be initializing.")

# --- Quick Links ---
st.divider()
st.subheader("Quick Links")

col1, col2, col3 = st.columns(3)
with col1:
    st.page_link("pages/01_eval_runs.py", label="Eval Runs", icon="🚀")
    st.page_link("pages/02_conversation_viewer.py", label="Conversation Viewer", icon="💬")
    st.page_link("pages/03_human_eval.py", label="Human Evaluation", icon="👤")
    st.page_link("pages/04_rubric_editor.py", label="Rubric Editor", icon="📋")
with col2:
    st.page_link("pages/05_agent_configs.py", label="Agent Configs", icon="🤖")
    st.page_link("pages/06_scenarios.py", label="Scenarios", icon="🎭")
    st.page_link("pages/07_agent_comparison.py", label="Agent Comparison", icon="📊")
with col3:
    st.page_link("pages/08_metrics_dashboard.py", label="Metrics Dashboard", icon="📈")
    st.page_link("pages/09_elo_rankings.py", label="ELO Rankings", icon="🏆")
    st.page_link("pages/10_calibration.py", label="Calibration & Reliability", icon="🎯")

# --- Getting Started ---
st.divider()
st.subheader("Getting Started")

st.markdown("""
**AgentProbe** evaluates multi-turn AI agent conversations using automated model judges,
human evaluators, and pairwise comparisons. Here is the recommended demo flow:

1. **Review Agent Configs** -- Go to *Agent Configs* to see the pre-configured agents
   (e.g., different models or system prompts).

2. **Review Scenarios** -- Go to *Scenarios* to see the customer support test scenarios
   with their turn templates, personas, and constraints.

3. **Check Eval Runs** -- Go to *Eval Runs* to see completed evaluation runs.
   Each run pairs an agent config with a scenario and produces multiple conversations.

4. **Browse Conversations** -- Go to *Conversation Viewer* to step through individual
   conversations turn-by-turn, seeing tool calls and evaluation scores.

5. **Compare Agents** -- Go to *Agent Comparison* for the key visualization page.
   See radar charts, bar charts, and violin plots comparing agent performance
   across rubric dimensions.

6. **Explore Metrics** -- Go to *Metrics Dashboard* for automated metrics:
   latency, token usage, tool success rates, and correlation heatmaps.

7. **ELO Rankings** -- Go to *ELO Rankings* to run pairwise comparisons
   and see head-to-head agent rankings.

8. **Calibration** -- Go to *Calibration & Reliability* to measure how well
   the model judge aligns with human evaluators.
""")
