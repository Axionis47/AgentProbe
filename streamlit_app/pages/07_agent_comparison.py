import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lib.api_client import AgentProbeClient

st.set_page_config(page_title="Agent Comparison - AgentProbe", layout="wide")
st.title("Agent Comparison")

client = AgentProbeClient()


@st.cache_data(ttl=300)
def load_agent_data(agent_id: str, agent_name: str) -> dict:
    """Load all eval data for a single agent."""
    runs = client.list_eval_runs(limit=100)
    agent_runs = [r for r in runs.get("items", []) if r.get("agent_config_id") == agent_id]

    all_evals = []
    all_metrics = []
    for run in agent_runs:
        convs = client.list_conversations(eval_run_id=run["id"], limit=100)
        for conv in convs.get("items", []):
            if conv["status"] != "completed":
                continue
            try:
                evals = client.get_conversation_evaluations(conv["id"])
                for ev in evals.get("items", []):
                    ev["agent"] = agent_name
                    all_evals.append(ev)
            except Exception:
                pass
            try:
                metrics = client.get_conversation_metrics(conv["id"])
                for m in metrics.get("items", []):
                    m["agent"] = agent_name
                    all_metrics.append(m)
            except Exception:
                pass

    return {"evals": all_evals, "metrics": all_metrics}


# --- Select agents ---
try:
    configs_data = client.list_agent_configs(is_active=True, limit=50)
    configs = configs_data.get("items", [])
except Exception as e:
    st.error(f"Failed to load agent configs: {e}")
    st.stop()

if len(configs) < 2:
    st.warning("Need at least 2 active agent configs for comparison.")
    st.stop()

config_map = {c["id"]: c["name"] for c in configs}
selected_ids = st.multiselect(
    "Select Agents to Compare (2+)",
    options=list(config_map.keys()),
    format_func=lambda x: config_map[x],
    default=list(config_map.keys())[:2],
)

if len(selected_ids) < 2:
    st.info("Select at least 2 agents.")
    st.stop()

# --- Load data ---
with st.spinner("Loading agent evaluation data..."):
    agent_data = {}
    for aid in selected_ids:
        agent_data[aid] = load_agent_data(aid, config_map[aid])

all_evals = []
all_metrics = []
for ad in agent_data.values():
    all_evals.extend(ad["evals"])
    all_metrics.extend(ad["metrics"])

if not all_evals:
    st.warning("No evaluations found for selected agents. Run some eval pipelines first.")
    st.stop()

evals_df = pd.DataFrame(all_evals)
metrics_df = pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()

# --- Overall Score Bar Chart ---
st.subheader("Overall Score by Agent")
if "overall_score" in evals_df.columns:
    score_by_agent = evals_df.groupby("agent")["overall_score"].mean().reset_index()
    fig = px.bar(score_by_agent, x="agent", y="overall_score", color="agent", title="Average Overall Score")
    fig.update_layout(yaxis_range=[0, 10])
    st.plotly_chart(fig, use_container_width=True)

# --- Dimension Scores Grouped Bar ---
st.subheader("Dimension Scores")
dim_rows = []
for _, ev in evals_df.iterrows():
    if isinstance(ev.get("scores"), dict):
        for dim, score in ev["scores"].items():
            try:
                dim_rows.append({"agent": ev["agent"], "dimension": dim, "score": float(score)})
            except (ValueError, TypeError):
                pass

if dim_rows:
    dim_df = pd.DataFrame(dim_rows)
    dim_agg = dim_df.groupby(["agent", "dimension"])["score"].mean().reset_index()
    fig = px.bar(dim_agg, x="dimension", y="score", color="agent", barmode="group", title="Average Dimension Scores")
    fig.update_layout(yaxis_range=[0, 10])
    st.plotly_chart(fig, use_container_width=True)

    # --- Radar Chart ---
    st.subheader("Radar Comparison")
    dimensions = sorted(dim_agg["dimension"].unique())
    fig = go.Figure()
    for agent in dim_agg["agent"].unique():
        agent_scores = dim_agg[dim_agg["agent"] == agent]
        values = []
        for d in dimensions:
            match = agent_scores[agent_scores["dimension"] == d]["score"]
            values.append(match.values[0] if len(match) > 0 else 0)
        values.append(values[0])  # close the polygon
        fig.add_trace(go.Scatterpolar(r=values, theta=dimensions + [dimensions[0]], fill="toself", name=agent))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 10])), title="Dimension Radar")
    st.plotly_chart(fig, use_container_width=True)

# --- Violin Plot of Overall Scores ---
st.subheader("Score Distribution (Violin)")
if "overall_score" in evals_df.columns and len(evals_df) > 1:
    fig = px.violin(evals_df, x="agent", y="overall_score", color="agent", box=True, points="all", title="Overall Score Distribution")
    fig.update_layout(yaxis_range=[0, 10])
    st.plotly_chart(fig, use_container_width=True)

# --- Performance Metrics Box Plots ---
if not metrics_df.empty and "value" in metrics_df.columns:
    st.subheader("Performance Metrics")
    metric_names = sorted(metrics_df["metric_name"].unique())
    selected_metric = st.selectbox("Select Metric", metric_names)
    filtered = metrics_df[metrics_df["metric_name"] == selected_metric]
    if not filtered.empty:
        fig = px.box(filtered, x="agent", y="value", color="agent", points="all", title=f"{selected_metric} Distribution")
        st.plotly_chart(fig, use_container_width=True)
