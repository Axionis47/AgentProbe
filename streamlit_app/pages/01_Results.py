import streamlit as st
import pandas as pd
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
