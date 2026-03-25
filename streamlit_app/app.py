import streamlit as st

st.set_page_config(
    page_title="AgentProbe",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

from lib.api_client import AgentProbeClient

client = AgentProbeClient()

# --- Health check (subtle) ---
api_ok = False
try:
    status = client.health()
    api_ok = True
except Exception:
    pass

st.title("AgentProbe -- Multi-Turn Agent Evaluation")
st.caption("Test how your AI agent handles real conversations. Statistically.")

if api_ok:
    st.markdown(
        '<span style="color: #2ecc71; font-size: 0.85em;">&#9679; API connected</span>',
        unsafe_allow_html=True,
    )
else:
    st.warning("API is not reachable. Start the backend to enable all features.")

# --- Big metric cards ---
if api_ok:
    try:
        runs_data = client.list_eval_runs(limit=1)
        convs_data = client.list_conversations(limit=1)
        configs_data = client.list_agent_configs(limit=1)

        # Count evaluations: iterate completed runs and sum evaluations
        eval_count = 0
        all_runs = client.list_eval_runs(limit=100)
        for run in all_runs.get("items", []):
            if run.get("status") != "completed":
                continue
            run_convs = client.list_conversations(eval_run_id=run["id"], limit=100)
            for conv in run_convs.get("items", []):
                try:
                    evals = client.get_conversation_evaluations(conv["id"])
                    eval_count += len(evals.get("items", []))
                except Exception:
                    pass

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Eval Runs", runs_data.get("total", 0))
        with col2:
            st.metric("Conversations", convs_data.get("total", 0))
        with col3:
            st.metric("Evaluations", eval_count)
        with col4:
            st.metric("Agents Tested", configs_data.get("total", 0))
    except Exception:
        st.info("Could not load stats. The API may still be initializing.")

# --- What to do ---
st.divider()
st.subheader("What to do")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1. See Results")
    st.write("View evaluation run results across all agents and scenarios.")
    st.page_link("pages/01_Results.py", label="Open Results Dashboard", icon="📊")

with col2:
    st.markdown("### 2. Compare Agents")
    st.write("Head-to-head radar charts and score breakdowns.")
    st.page_link("pages/04_Compare.py", label="Open Agent Comparison", icon="⚖️")

with col3:
    st.markdown("### 3. Inspect Conversations")
    st.write("Step through multi-turn conversations turn by turn.")
    st.page_link("pages/02_Conversations.py", label="Open Conversation Inspector", icon="💬")
