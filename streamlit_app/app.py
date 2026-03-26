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

try:
    all_runs = client.list_eval_runs(limit=100).get("items", [])
    completed_runs = [r for r in all_runs if r.get("status") == "completed"]
except Exception:
    st.warning("Could not load data.")
    st.stop()

if not completed_runs:
    st.info("No completed evaluation runs yet. Run `make seed && make demo` first.")
    st.stop()

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

# --- Metric Cards ---
st.markdown("")

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

# --- Scores by Scenario ---
st.markdown("")
st.subheader("Scores by Scenario")

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

# --- Evaluator Breakdown (table) ---
st.markdown("")
st.subheader("Evaluator Breakdown")

eval_summary = df.groupby(["agent", "evaluator"])["overall_score"].agg(["mean", "std"]).reset_index()
eval_summary.columns = ["Agent", "Evaluator", "Mean", "Std"]
eval_summary["Mean"] = eval_summary["Mean"].round(1)
eval_summary["Std"] = eval_summary["Std"].round(1)
st.dataframe(eval_summary, use_container_width=True, hide_index=True)

# --- Score Distribution ---
st.markdown("")
st.subheader("Score Distribution")

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

# --- Quick Links ---
st.markdown("")
col1, col2, col3 = st.columns(3)
with col1:
    st.page_link("pages/02_Conversations.py", label="Conversations", icon="💬")
with col2:
    st.page_link("pages/04_Compare.py", label="Compare Agents", icon="📊")
with col3:
    st.page_link("pages/05_Metrics.py", label="Performance Metrics", icon="⚡")
