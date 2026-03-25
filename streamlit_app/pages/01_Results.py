import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from lib.api_client import AgentProbeClient

st.set_page_config(page_title="Results Dashboard - AgentProbe", layout="wide")

st.header("Results Dashboard")
st.caption("Evaluation run results across all agents and scenarios.")

client = AgentProbeClient()

STATUS_COLORS = {
    "completed": "🟢",
    "pending": "🟡",
    "running_simulation": "🔵",
    "running_evaluation": "🔵",
    "failed": "🔴",
    "cancelled": "⚪",
}

# Pre-fetch agent and scenario names
agent_names: dict[str, str] = {}
scenario_names: dict[str, str] = {}
try:
    configs_data = client.list_agent_configs(limit=100)
    agent_names = {c["id"]: c["name"] for c in configs_data.get("items", [])}
except Exception:
    pass
try:
    scenarios_data = client.list_scenarios(limit=100)
    scenario_names = {s["id"]: s["name"] for s in scenarios_data.get("items", [])}
except Exception:
    pass

# Fetch all runs
try:
    data = client.list_eval_runs(limit=100)
    runs = data.get("items", [])
except Exception as e:
    st.error(f"Failed to connect to API: {e}")
    st.stop()

if not runs:
    st.info("No eval runs found. Go to the Admin page to create one.")
    st.stop()


@st.cache_data(ttl=120)
def compute_run_scores(run_id: str) -> dict:
    """Get conversation count and average score for a run."""
    try:
        convs_data = client.list_conversations(eval_run_id=run_id, limit=100)
        convs = convs_data.get("items", [])
    except Exception:
        return {"num_conversations": 0, "avg_score": None, "conversations": []}

    scores = []
    conv_details = []
    for conv in convs:
        conv_info = {
            "id": conv["id"],
            "sequence": conv.get("sequence_num", "?"),
            "status": conv["status"],
            "turns": conv["turn_count"],
            "tokens": conv["total_tokens"],
        }
        if conv["status"] == "completed":
            try:
                evals = client.get_conversation_evaluations(conv["id"])
                eval_items = evals.get("items", [])
                conv_scores = [
                    e["overall_score"]
                    for e in eval_items
                    if e.get("overall_score") is not None
                    and e.get("evaluator_type") != "pairwise_judge"
                ]
                if conv_scores:
                    avg = sum(conv_scores) / len(conv_scores)
                    scores.append(avg)
                    conv_info["avg_score"] = round(avg, 1)
                else:
                    conv_info["avg_score"] = None
            except Exception:
                conv_info["avg_score"] = None
        else:
            conv_info["avg_score"] = None
        conv_details.append(conv_info)

    return {
        "num_conversations": len(convs),
        "avg_score": round(sum(scores) / len(scores), 2) if scores else None,
        "conversations": conv_details,
    }


# Build summary table
with st.spinner("Loading results..."):
    table_rows = []
    run_details = {}
    for run in runs:
        run_id = run["id"]
        agent = agent_names.get(run.get("agent_config_id", ""), run.get("agent_config_id", "")[:8])
        scenario = scenario_names.get(run.get("scenario_id", ""), run.get("scenario_id", "")[:8])
        status = run["status"]
        badge = STATUS_COLORS.get(status, "⚪")

        scores_data = compute_run_scores(run_id)
        run_details[run_id] = scores_data

        avg_score_str = f"{scores_data['avg_score']:.1f}/10" if scores_data["avg_score"] is not None else "--"

        table_rows.append({
            "Agent": agent,
            "Scenario": scenario,
            "Status": f"{badge} {status}",
            "Conversations": scores_data["num_conversations"],
            "Avg Score": avg_score_str,
            "_run_id": run_id,
        })

# ============================================================
# Quick Stats & Score Distribution
# ============================================================

# Collect per-agent scores from completed runs
agent_scores: dict[str, list[float]] = {}
for row in table_rows:
    run_id = row["_run_id"]
    agent_name = row["Agent"]
    details = run_details[run_id]
    for conv in details["conversations"]:
        if conv.get("avg_score") is not None:
            agent_scores.setdefault(agent_name, []).append(conv["avg_score"])

pro_scores = agent_scores.get("Customer Support Pro", [])
baseline_scores = agent_scores.get("Generic Baseline", [])

if pro_scores or baseline_scores:
    st.subheader("Quick Stats")
    col1, col2, col3 = st.columns(3)
    pro_mean = sum(pro_scores) / len(pro_scores) if pro_scores else 0.0
    baseline_mean = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
    gap = pro_mean - baseline_mean

    col1.metric(
        label="Avg Score (Pro)",
        value=f"{pro_mean:.2f}" if pro_scores else "--",
    )
    col2.metric(
        label="Avg Score (Baseline)",
        value=f"{baseline_mean:.2f}" if baseline_scores else "--",
    )
    col3.metric(
        label="Score Gap",
        value=f"{abs(gap):.2f}" if (pro_scores and baseline_scores) else "--",
        delta=f"{gap:+.2f}" if (pro_scores and baseline_scores) else None,
    )

    # Score Distribution histogram
    st.subheader("Score Distribution")
    bins = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
    bin_labels = ["0-2", "2-4", "4-6", "6-8", "8-10"]

    def _bin_scores(scores: list[float]) -> list[int]:
        counts = [0] * len(bins)
        for s in scores:
            for i, (lo, hi) in enumerate(bins):
                if lo <= s < hi or (i == len(bins) - 1 and s == hi):
                    counts[i] += 1
                    break
        return counts

    fig = go.Figure()
    if pro_scores:
        fig.add_trace(go.Bar(
            x=bin_labels,
            y=_bin_scores(pro_scores),
            name="Customer Support Pro",
            marker_color="#636EFA",
            opacity=0.75,
        ))
    if baseline_scores:
        fig.add_trace(go.Bar(
            x=bin_labels,
            y=_bin_scores(baseline_scores),
            name="Generic Baseline",
            marker_color="#EF553B",
            opacity=0.75,
        ))
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Score Range",
        yaxis_title="Count",
        height=320,
        margin=dict(t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

# Display summary table
st.subheader("All Eval Runs")
display_df = pd.DataFrame(table_rows)
if not display_df.empty:
    st.dataframe(
        display_df[["Agent", "Scenario", "Status", "Conversations", "Avg Score"]],
        use_container_width=True,
        hide_index=True,
    )

# Expandable per-run details
st.subheader("Per-Run Breakdown")
st.caption("Expand any run below to see individual conversation scores.")

for row in table_rows:
    run_id = row["_run_id"]
    label = f"{row['Status']}  {row['Agent']} / {row['Scenario']} -- Avg: {row['Avg Score']}"

    with st.expander(label):
        details = run_details[run_id]
        convs = details["conversations"]
        if not convs:
            st.write("No conversations in this run.")
            continue

        conv_df = pd.DataFrame(convs)
        conv_df = conv_df.rename(columns={
            "sequence": "#",
            "status": "Status",
            "turns": "Turns",
            "tokens": "Tokens",
            "avg_score": "Score",
        })
        display_cols = ["#", "Status", "Turns", "Tokens", "Score"]
        existing_cols = [c for c in display_cols if c in conv_df.columns]
        st.dataframe(conv_df[existing_cols], use_container_width=True, hide_index=True)
