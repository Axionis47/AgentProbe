import streamlit as st

st.set_page_config(page_title="AgentProbe", page_icon="🔍", layout="wide")

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
    st.error("Backend not reachable. Run `make up`.")
    st.stop()

try:
    all_runs = client.list_eval_runs(limit=100).get("items", [])
    completed_runs = [r for r in all_runs if r.get("status") == "completed"]
except Exception:
    st.warning("Could not load data.")
    st.stop()

if not completed_runs:
    st.info("No data. Run `make seed && make demo`.")
    st.stop()

# --- Load all evaluation data ---
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
                    "reasoning": ev.get("reasoning", ""),
                })
    except Exception:
        continue

if not rows:
    st.info("No evaluation data.")
    st.stop()

df = pd.DataFrame(rows)
agents = df["agent"].unique().tolist()
traj_df = df[df["evaluator"] == "trajectory"]

# =====================================================
# SECTION 1: Tool Calling Analysis (THE LEAD)
# =====================================================
st.subheader("Tool Calling Analysis")

if not traj_df.empty:
    # Parse actual vs expected from reasoning text
    traj_rows = []
    for _, row in traj_df.iterrows():
        reasoning = row["reasoning"]
        scores = row["scores"] if isinstance(row["scores"], dict) else {}

        actual = ""
        expected = ""
        if "Actual tools:" in reasoning:
            actual = reasoning.split("Actual tools:")[1].split(".")[0].strip()
        if "Expected:" in reasoning:
            expected = reasoning.split("Expected:")[1].split(".")[0].strip()

        passed = scores.get("recall", 0) == 1.0 and scores.get("precision", 0) >= 0.5 and scores.get("order_score", 0) == 1.0

        traj_rows.append({
            "Agent": row["agent"],
            "Scenario": row["scenario"],
            "Expected": expected,
            "Actual": actual,
            "Recall": f"{scores.get('recall', 0):.0%}",
            "Precision": f"{scores.get('precision', 0):.0%}",
            "Order": f"{scores.get('order_score', 0):.0%}",
            "Result": "PASS" if passed else "FAIL",
        })

    traj_table = pd.DataFrame(traj_rows)

    # Pass rate per agent
    col1, col2 = st.columns(2)
    for i, agent in enumerate(agents):
        agent_traj = traj_table[traj_table["Agent"] == agent]
        total = len(agent_traj)
        passed = len(agent_traj[agent_traj["Result"] == "PASS"])
        rate = (passed / total * 100) if total > 0 else 0
        with [col1, col2][i % 2]:
            st.metric(agent, f"{rate:.0f}% pass rate", delta=f"{passed}/{total} runs")

    # Detail table
    display_cols = ["Agent", "Scenario", "Expected", "Actual", "Result"]
    styled = traj_table[display_cols].style.map(
        lambda v: "color: #2ecc71; font-weight: bold" if v == "PASS" else ("color: #e74c3c; font-weight: bold" if v == "FAIL" else ""),
        subset=["Result"],
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Failure analysis
    failures = traj_table[traj_table["Result"] == "FAIL"]
    if not failures.empty:
        st.markdown("**Failure Modes**")
        failure_counts = failures.groupby(["Agent", "Scenario"]).size().reset_index(name="Failures")
        st.dataframe(failure_counts, use_container_width=True, hide_index=True)

else:
    st.info("No trajectory data.")

# =====================================================
# SECTION 2: Overall Scores
# =====================================================
st.markdown("")
st.subheader("Overall Scores")

judge_df = df[df["evaluator"] == "model_judge"]
if judge_df.empty:
    judge_df = df

scenario_avg = judge_df.groupby(["agent", "scenario"])["overall_score"].agg(["mean", "std"]).reset_index()
scenario_avg.columns = ["Agent", "Scenario", "Avg", "Std"]

fig = go.Figure()
for agent in agents:
    ad = scenario_avg[scenario_avg["Agent"] == agent]
    fig.add_trace(go.Bar(
        name=agent, x=ad["Scenario"], y=ad["Avg"],
        error_y=dict(type="data", array=ad["Std"].fillna(0), visible=True),
        text=[f"{v:.1f}" for v in ad["Avg"]], textposition="outside",
    ))
fig.update_layout(
    barmode="group", yaxis=dict(range=[0, 11]), height=350, margin=dict(t=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

# =====================================================
# SECTION 3: Evaluator Summary
# =====================================================
st.markdown("")
st.subheader("Evaluator Summary")

eval_summary = df.groupby(["agent", "evaluator"])["overall_score"].agg(["mean", "std"]).reset_index()
eval_summary.columns = ["Agent", "Evaluator", "Mean", "Std"]
eval_summary["Mean"] = eval_summary["Mean"].round(1)
eval_summary["Std"] = eval_summary["Std"].round(1)
st.dataframe(eval_summary, use_container_width=True, hide_index=True)

# =====================================================
# Links
# =====================================================
st.markdown("")
col1, col2, col3 = st.columns(3)
with col1:
    st.page_link("pages/02_Conversations.py", label="Conversations", icon="💬")
with col2:
    st.page_link("pages/04_Compare.py", label="Compare Agents", icon="📊")
with col3:
    st.page_link("pages/05_Metrics.py", label="Metrics", icon="⚡")
