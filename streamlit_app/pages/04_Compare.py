import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lib.api_client import AgentProbeClient

st.set_page_config(page_title="Agent Comparison - AgentProbe", layout="wide")

st.header("Agent Comparison")
st.caption("Head-to-head: which agent is better?")

client = AgentProbeClient()


@st.cache_data(ttl=300)
def load_agent_data(agent_id: str, agent_name: str) -> dict:
    """Load all eval data for a single agent."""
    runs = client.list_eval_runs(limit=100)
    agent_runs = [r for r in runs.get("items", []) if r.get("agent_config_id") == agent_id]

    all_evals = []
    for run in agent_runs:
        convs = client.list_conversations(eval_run_id=run["id"], limit=100)
        for conv in convs.get("items", []):
            if conv["status"] != "completed":
                continue
            try:
                evals = client.get_conversation_evaluations(conv["id"])
                for ev in evals.get("items", []):
                    if ev.get("evaluator_type") == "pairwise_judge":
                        continue
                    ev["agent"] = agent_name
                    all_evals.append(ev)
            except Exception:
                pass

    return {"evals": all_evals}


# --- Select agents ---
try:
    configs_data = client.list_agent_configs(is_active=True, limit=50)
    configs = configs_data.get("items", [])
except Exception as e:
    st.error(f"Failed to load agent configs: {e}")
    st.stop()

if not configs:
    st.info("No agent configs found. Create agents in the Admin page first.")
    st.stop()

config_map = {c["id"]: c["name"] for c in configs}

if len(configs) < 2:
    st.warning(
        "Only one agent has data. Add a second agent configuration and run an "
        "evaluation to compare them head-to-head."
    )
    # Still show single agent data below

selected_ids = st.multiselect(
    "Select agents to compare",
    options=list(config_map.keys()),
    format_func=lambda x: config_map[x],
    default=list(config_map.keys())[:2],
)

if not selected_ids:
    st.info("Select at least one agent above.")
    st.stop()

# --- Load data ---
with st.spinner("Loading evaluation data..."):
    agent_data = {}
    for aid in selected_ids:
        agent_data[aid] = load_agent_data(aid, config_map[aid])

all_evals = []
for ad in agent_data.values():
    all_evals.extend(ad["evals"])

if not all_evals:
    st.warning("No evaluations found for selected agents. Run some evaluations first.")
    st.stop()

evals_df = pd.DataFrame(all_evals)

# --- Overall Score Bar Chart ---
st.subheader("Overall Score")
if "overall_score" in evals_df.columns:
    score_by_agent = evals_df.groupby("agent")["overall_score"].mean().reset_index()
    fig = px.bar(
        score_by_agent, x="agent", y="overall_score", color="agent",
        title="Average Overall Score",
        labels={"agent": "Agent", "overall_score": "Score"},
    )
    fig.update_layout(yaxis_range=[0, 10], showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# --- Radar Chart ---
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

    st.subheader("Dimension Radar")
    dimensions = sorted(dim_agg["dimension"].unique())

    RADAR_COLORS = [
        "rgba(31, 119, 180, 0.5)",
        "rgba(255, 127, 14, 0.5)",
        "rgba(44, 160, 44, 0.5)",
        "rgba(214, 39, 40, 0.5)",
        "rgba(148, 103, 189, 0.5)",
    ]

    fig = go.Figure()
    for idx, agent in enumerate(dim_agg["agent"].unique()):
        agent_scores = dim_agg[dim_agg["agent"] == agent]
        values = []
        for d in dimensions:
            match = agent_scores[agent_scores["dimension"] == d]["score"]
            values.append(match.values[0] if len(match) > 0 else 0)
        values.append(values[0])  # close polygon
        color = RADAR_COLORS[idx % len(RADAR_COLORS)]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=[d.replace("_", " ").title() for d in dimensions] + [dimensions[0].replace("_", " ").title()],
            fill="toself",
            fillcolor=color,
            name=agent,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Dimension comparison table ---
    st.subheader("Dimension Breakdown")
    agents = sorted(dim_agg["agent"].unique())
    table_rows = []
    for d in dimensions:
        row = {"Dimension": d.replace("_", " ").title()}
        scores_for_dim = {}
        for agent in agents:
            match = dim_agg[(dim_agg["agent"] == agent) & (dim_agg["dimension"] == d)]["score"]
            val = round(match.values[0], 2) if len(match) > 0 else None
            row[agent] = val
            if val is not None:
                scores_for_dim[agent] = val

        if len(scores_for_dim) >= 2:
            best = max(scores_for_dim, key=scores_for_dim.get)
            row["Winner"] = best
        elif len(scores_for_dim) == 1:
            row["Winner"] = "--"
        else:
            row["Winner"] = "--"
        table_rows.append(row)

    table_df = pd.DataFrame(table_rows)
    st.dataframe(table_df, use_container_width=True, hide_index=True)
else:
    st.info("No per-dimension scores found in the evaluation data.")
