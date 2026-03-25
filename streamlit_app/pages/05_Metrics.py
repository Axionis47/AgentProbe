import streamlit as st
import pandas as pd
import plotly.express as px
from lib.api_client import AgentProbeClient

st.set_page_config(page_title="Performance Metrics - AgentProbe", layout="wide")

st.header("Performance Metrics")
st.caption("Token usage, latency, and tool calling statistics.")

client = AgentProbeClient()

# --- Pre-fetch names ---
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

# --- Select eval run ---
try:
    runs_data = client.list_eval_runs(limit=100)
    runs = [r for r in runs_data.get("items", []) if r.get("status") == "completed"]
except Exception as e:
    st.error(f"Failed to load eval runs: {e}")
    st.stop()

if not runs:
    st.info("No completed eval runs found. Run an evaluation first.")
    st.stop()

run_options = {}
for r in runs:
    agent = agent_names.get(r.get("agent_config_id", ""), "Unknown")
    scenario = scenario_names.get(r.get("scenario_id", ""), "Unknown")
    run_options[r["id"]] = f"{agent} / {scenario}"

selected_run_id = st.selectbox(
    "Pick an eval run",
    options=list(run_options.keys()),
    format_func=lambda x: run_options[x],
)


@st.cache_data(ttl=300)
def load_run_metrics(run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all conversations and their metrics for a run."""
    convs_data = client.list_conversations(eval_run_id=run_id, limit=100)
    convs = convs_data.get("items", [])

    conv_rows = []
    metric_rows = []
    for c in convs:
        conv_rows.append({
            "id": c["id"],
            "sequence": c.get("sequence_num", 0),
            "status": c["status"],
            "turns": c["turn_count"],
            "tokens": c["total_tokens"],
            "latency_ms": c["total_latency_ms"],
        })
        if c["status"] == "completed":
            try:
                metrics = client.get_conversation_metrics(c["id"])
                for m in metrics.get("items", []):
                    metric_rows.append({
                        "conversation_id": c["id"],
                        "sequence": c.get("sequence_num", 0),
                        "metric_name": m["metric_name"],
                        "value": m["value"],
                        "unit": m.get("unit", ""),
                    })
            except Exception:
                pass

    return pd.DataFrame(conv_rows), pd.DataFrame(metric_rows)


with st.spinner("Loading metrics..."):
    conv_df, metrics_df = load_run_metrics(selected_run_id)

if conv_df.empty:
    st.warning("No conversations found in this run.")
    st.stop()

# --- Big 4 metric cards ---
st.divider()

KEY_METRICS = {
    "tokens_per_turn": "Avg Tokens/Turn",
    "avg_latency_ms": "Avg Latency (ms)",
    "tool_success_rate": "Tool Success Rate",
    "turns_to_resolution": "Turns to Resolution",
}

if not metrics_df.empty:
    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]
    for i, (metric_key, label) in enumerate(KEY_METRICS.items()):
        subset = metrics_df[metrics_df["metric_name"] == metric_key]
        if not subset.empty:
            avg_val = subset["value"].mean()
            unit = subset.iloc[0]["unit"]
            if metric_key == "tool_success_rate":
                display = f"{avg_val:.0%}" if avg_val <= 1 else f"{avg_val:.1f}"
            else:
                display = f"{avg_val:.1f}"
                if unit:
                    display += f" {unit}"
            cols[i].metric(label, display)
        else:
            cols[i].metric(label, "--")
else:
    st.info("No metrics data available for this run.")
    st.stop()

# --- Bar chart of metrics per conversation ---
st.divider()
st.subheader("Metrics by Conversation")

available_metrics = sorted(metrics_df["metric_name"].unique())
selected_metric = st.selectbox("Select a metric", available_metrics)

filtered = metrics_df[metrics_df["metric_name"] == selected_metric].sort_values("sequence")
if not filtered.empty:
    fig = px.bar(
        filtered, x="sequence", y="value",
        title=f"{selected_metric.replace('_', ' ').title()} by Conversation",
        labels={"sequence": "Conversation #", "value": selected_metric.replace("_", " ").title()},
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Histograms (clean) ---
st.divider()
st.subheader("Distributions")

METRIC_CATEGORIES = {
    "Token Usage": ["tokens_per_turn", "output_input_ratio"],
    "Latency": ["avg_latency_ms", "p95_latency_ms"],
    "Tool Calls": ["tool_call_count", "tool_success_rate"],
    "Resolution": ["turns_to_resolution", "conversation_completed"],
}

for category, metric_names in METRIC_CATEGORIES.items():
    cat_df = metrics_df[metrics_df["metric_name"].isin(metric_names)]
    if cat_df.empty:
        continue
    fig = px.histogram(
        cat_df, x="value", color="metric_name",
        barmode="overlay", nbins=15,
        title=f"{category}",
        labels={"value": "Value", "metric_name": "Metric"},
    )
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)
