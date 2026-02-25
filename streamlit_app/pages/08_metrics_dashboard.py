import streamlit as st
import pandas as pd
import plotly.express as px
from lib.api_client import AgentProbeClient

st.set_page_config(page_title="Metrics Dashboard - AgentProbe", layout="wide")
st.title("Metrics Dashboard")

client = AgentProbeClient()

# --- Select eval run ---
try:
    runs_data = client.list_eval_runs(limit=100)
    runs = [r for r in runs_data.get("items", []) if r.get("status") == "completed"]
except Exception as e:
    st.error(f"Failed to load eval runs: {e}")
    st.stop()

if not runs:
    st.info("No completed eval runs found. Run an evaluation pipeline first.")
    st.stop()

run_options = {r["id"]: f"{r.get('name', r['id'][:8])} ({r['status']})" for r in runs}
selected_run_id = st.selectbox("Select Completed Eval Run", options=list(run_options.keys()), format_func=lambda x: run_options[x])


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


with st.spinner("Loading metrics data..."):
    conv_df, metrics_df = load_run_metrics(selected_run_id)

if conv_df.empty:
    st.warning("No conversations found in this run.")
    st.stop()

# --- Conversation Summary ---
st.subheader("Conversation Summary")
st.dataframe(conv_df, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Total Conversations", len(conv_df))
col2.metric("Completed", len(conv_df[conv_df["status"] == "completed"]))
col3.metric("Avg Turns", round(conv_df["turns"].mean(), 1))

if metrics_df.empty:
    st.warning("No metrics available for this run.")
    st.stop()

# --- Aggregated Metrics Table ---
st.subheader("Aggregated Metrics")
agg = metrics_df.groupby("metric_name")["value"].agg(["mean", "std", "min", "max", "count"]).round(3).reset_index()
agg.columns = ["Metric", "Mean", "Std", "Min", "Max", "Count"]
st.dataframe(agg, use_container_width=True)

# --- Histograms by Category ---
st.subheader("Metric Distributions")

METRIC_CATEGORIES = {
    "Token": ["tokens_per_turn", "output_input_ratio"],
    "Latency": ["avg_latency_ms", "p95_latency_ms"],
    "Resolution": ["turns_to_resolution", "conversation_completed"],
    "Tool": ["tool_call_count", "tool_success_rate"],
}

for category, metric_names in METRIC_CATEGORIES.items():
    cat_df = metrics_df[metrics_df["metric_name"].isin(metric_names)]
    if cat_df.empty:
        continue
    st.write(f"**{category} Metrics**")
    fig = px.histogram(cat_df, x="value", color="metric_name", barmode="overlay", nbins=20, title=f"{category} Metric Distribution")
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)

# --- Correlation Heatmap ---
st.subheader("Metric Correlation Heatmap")
pivot = metrics_df.pivot_table(index="conversation_id", columns="metric_name", values="value")
if pivot.shape[1] >= 2:
    corr = pivot.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Metric Correlations")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough metrics for correlation analysis.")

# --- Per-Conversation Metric Chart ---
st.subheader("Per-Conversation Metrics")
available_metrics = sorted(metrics_df["metric_name"].unique())
selected_metric = st.selectbox("Select Metric", available_metrics)
filtered = metrics_df[metrics_df["metric_name"] == selected_metric].sort_values("sequence")
if not filtered.empty:
    fig = px.bar(filtered, x="sequence", y="value", title=f"{selected_metric} by Conversation", labels={"sequence": "Conversation #", "value": selected_metric})
    st.plotly_chart(fig, use_container_width=True)
