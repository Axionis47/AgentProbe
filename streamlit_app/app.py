import streamlit as st

st.set_page_config(
    page_title="AgentProbe",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import plotly.graph_objects as go
from lib.api_client import AgentProbeClient

client = AgentProbeClient()

# --- Health check ---
api_ok = False
try:
    client.health()
    api_ok = True
except Exception:
    pass

st.title("AgentProbe")

if not api_ok:
    st.error("Backend API is not reachable. Start it with `make up`.")
    st.stop()

st.markdown(
    "**What this does:** We threw simulated customer conversations at two AI agents "
    "and scored every conversation with 3 judges. Below are the results."
)

# --- Load all data ---
try:
    all_runs = client.list_eval_runs(limit=100).get("items", [])
    completed_runs = [r for r in all_runs if r.get("status") == "completed"]
except Exception:
    st.warning("Could not load data.")
    st.stop()

if not completed_runs:
    st.info("No completed evaluation runs yet. Run `make seed && make demo` first.")
    st.stop()

# --- Collect scores per run ---
rows = []
for run in completed_runs:
    try:
        convs = client.list_conversations(eval_run_id=run["id"], limit=100).get("items", [])
        for conv in convs:
            evals = client.get_conversation_evaluations(conv["id"]).get("items", [])
            for ev in evals:
                rows.append({
                    "agent": run["name"].split(" / ")[0].strip(),
                    "scenario": run["name"].split(" / ")[1].strip() if " / " in run["name"] else run["name"],
                    "evaluator": ev.get("evaluator_type", ""),
                    "overall_score": ev.get("overall_score", 0) or 0,
                    "scores": ev.get("scores", {}),
                    "conversation_id": conv["id"],
                })
    except Exception:
        continue

if not rows:
    st.info("No evaluation data found.")
    st.stop()

df = pd.DataFrame(rows)

# ===================================================================
# SECTION 1: The Headline
# ===================================================================
st.divider()

agents = df["agent"].unique().tolist()
if len(agents) >= 2:
    a1, a2 = agents[0], agents[1]
    avg1 = df[df["agent"] == a1]["overall_score"].mean()
    avg2 = df[df["agent"] == a2]["overall_score"].mean()
    gap = avg1 - avg2

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(a1, f"{avg1:.1f} / 10")
    with col2:
        st.metric(a2, f"{avg2:.1f} / 10")
    with col3:
        winner = a1 if gap > 0 else a2
        st.metric("Winner", winner, delta=f"{abs(gap):.1f} pts ahead")
else:
    agent = agents[0]
    avg = df["overall_score"].mean()
    st.metric(agent, f"{avg:.1f} / 10")

# ===================================================================
# SECTION 2: Score by Scenario (the interesting part)
# ===================================================================
st.divider()
st.subheader("Scores by Scenario")
st.caption("Each bar = average score across multiple conversations. Look for where agents differ.")

# Pivot: agent × scenario, grouped by evaluator
judge_df = df[df["evaluator"] == "model_judge"]
if judge_df.empty:
    judge_df = df

scenario_avg = judge_df.groupby(["agent", "scenario"])["overall_score"].agg(["mean", "std", "count"]).reset_index()
scenario_avg.columns = ["Agent", "Scenario", "Avg Score", "Std Dev", "N"]

fig = go.Figure()
for agent in agents:
    agent_data = scenario_avg[scenario_avg["Agent"] == agent]
    fig.add_trace(go.Bar(
        name=agent,
        x=agent_data["Scenario"],
        y=agent_data["Avg Score"],
        error_y=dict(type="data", array=agent_data["Std Dev"].fillna(0), visible=True),
        text=[f"{v:.1f}" for v in agent_data["Avg Score"]],
        textposition="outside",
    ))
fig.update_layout(
    barmode="group",
    yaxis=dict(range=[0, 11], title="Score (0-10)"),
    xaxis=dict(title=""),
    height=400,
    margin=dict(t=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

# ===================================================================
# SECTION 3: Evaluator Breakdown
# ===================================================================
st.divider()
st.subheader("What the Judges Say")
st.caption("3 independent judges scored each conversation. Do they agree?")

evaluator_labels = {
    "model_judge": "AI Judge (Gemini scores the conversation)",
    "rubric_grader": "Rule-Based Grader (heuristic checks, no AI)",
    "trajectory": "Tool Sequence Checker (did it call the right tools?)",
}

eval_summary = df.groupby(["agent", "evaluator"])["overall_score"].agg(["mean", "std"]).reset_index()
eval_summary.columns = ["Agent", "Evaluator", "Mean", "Std"]

for evaluator in df["evaluator"].unique():
    label = evaluator_labels.get(evaluator, evaluator)
    edata = eval_summary[eval_summary["Evaluator"] == evaluator]
    cols = st.columns(len(agents) + 1)
    with cols[0]:
        st.markdown(f"**{label}**")
    for i, agent in enumerate(agents):
        row = edata[edata["Agent"] == agent]
        if not row.empty:
            mean = row["Mean"].values[0]
            std = row["Std"].values[0]
            with cols[i + 1]:
                st.metric(agent, f"{mean:.1f}", delta=f"± {std:.1f} std dev", delta_color="off")

# ===================================================================
# SECTION 4: Score Distribution
# ===================================================================
st.divider()
st.subheader("Score Distribution")
st.caption("Not a single number — a spread. Higher variance = less reliable agent.")

fig2 = go.Figure()
for agent in agents:
    scores = df[df["agent"] == agent]["overall_score"]
    fig2.add_trace(go.Histogram(
        x=scores,
        name=agent,
        opacity=0.7,
        xbins=dict(start=0, end=10, size=1),
    ))
fig2.update_layout(
    barmode="overlay",
    xaxis=dict(title="Score", range=[0, 10.5]),
    yaxis=dict(title="Count"),
    height=300,
    margin=dict(t=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig2, use_container_width=True)

# ===================================================================
# SECTION 5: Quick Links
# ===================================================================
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.page_link("pages/02_Conversations.py", label="Read the actual conversations", icon="💬")
with col2:
    st.page_link("pages/04_Compare.py", label="Radar chart comparison", icon="📊")
with col3:
    st.page_link("pages/05_Metrics.py", label="Token & latency metrics", icon="⚡")
