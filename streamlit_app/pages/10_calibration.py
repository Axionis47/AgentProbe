import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lib.api_client import AgentProbeClient

st.set_page_config(page_title="Calibration & Reliability - AgentProbe", layout="wide")
st.title("Calibration & Interrater Reliability")

client = AgentProbeClient()

tab_calibration, tab_reliability = st.tabs(["Calibration Analysis", "Interrater Reliability"])

# ---- Shared: select eval run ----
try:
    runs_data = client.list_eval_runs(limit=100)
    runs = runs_data.get("items", [])
except Exception as e:
    st.error(f"Failed to load eval runs: {e}")
    st.stop()

if not runs:
    st.info("No eval runs found.")
    st.stop()

run_options = {r["id"]: f"{r.get('name', r['id'][:8])} ({r['status']})" for r in runs}


# ---- Calibration Tab ----
with tab_calibration:
    st.subheader("Model Judge vs Human Score Agreement")
    st.write("Measures how well automated model-judge scores predict human scores.")

    cal_run_id = st.selectbox(
        "Select Eval Run", list(run_options.keys()),
        format_func=lambda x: run_options[x], key="cal_run",
    )

    if st.button("Compute Calibration", key="cal_btn"):
        with st.spinner("Computing calibration metrics..."):
            try:
                data = client.get_calibration(cal_run_id)

                # KPI row
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Pearson r", f"{data['pearson_r']:.3f}")
                col2.metric("Spearman \u03c1", f"{data['spearman_rho']:.3f}")
                col3.metric("MAE", f"{data['mae']:.3f}")
                col4.metric("RMSE", f"{data['rmse']:.3f}")
                col5.metric("Bias", f"{data['bias']:+.3f}")

                st.write(f"Based on **{data['n']}** paired human + model evaluations.")

                # Interpretation
                r = data["pearson_r"]
                if r > 0.8:
                    st.success("Strong correlation — model judge is well-calibrated with humans.")
                elif r > 0.5:
                    st.warning("Moderate correlation — model judge partially agrees with humans.")
                else:
                    st.error("Weak correlation — model judge scores diverge from human scores.")

                # Calibration curve
                curve = data.get("calibration_curve", [])
                if curve:
                    curve_df = pd.DataFrame(curve)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=curve_df["avg_model"], y=curve_df["avg_human"],
                        mode="markers+lines", name="Actual",
                        marker=dict(size=curve_df["count"] * 3 + 5),
                    ))
                    # Perfect calibration line
                    min_val = min(curve_df["avg_model"].min(), curve_df["avg_human"].min())
                    max_val = max(curve_df["avg_model"].max(), curve_df["avg_human"].max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val], y=[min_val, max_val],
                        mode="lines", name="Perfect Calibration",
                        line=dict(dash="dash", color="gray"),
                    ))
                    fig.update_layout(
                        title="Calibration Curve", xaxis_title="Model Judge Score",
                        yaxis_title="Human Score",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Per-dimension metrics
                per_dim = data.get("per_dimension", {})
                if per_dim:
                    st.subheader("Per-Dimension Calibration")
                    dim_rows = []
                    for dim, metrics in per_dim.items():
                        dim_rows.append({"Dimension": dim, **metrics})
                    dim_df = pd.DataFrame(dim_rows)
                    st.dataframe(dim_df, use_container_width=True, hide_index=True)

                    fig2 = px.bar(
                        dim_df, x="Dimension", y="pearson_r",
                        title="Per-Dimension Pearson Correlation",
                        color="pearson_r",
                        color_continuous_scale="RdYlGn",
                        range_color=[-1, 1],
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"Calibration analysis failed: {e}")
                st.info("Ensure the selected run has both human AND model judge evaluations.")


# ---- Reliability Tab ----
with tab_reliability:
    st.subheader("Interrater Reliability (Krippendorff's Alpha)")
    st.write("Measures agreement among multiple human evaluators scoring the same conversations.")

    rel_run_id = st.selectbox(
        "Select Eval Run", list(run_options.keys()),
        format_func=lambda x: run_options[x], key="rel_run",
    )

    if st.button("Compute Reliability", key="rel_btn"):
        with st.spinner("Computing interrater reliability..."):
            try:
                data = client.get_reliability(rel_run_id)

                alpha = data["alpha"]
                num_items = data["num_items"]
                num_raters = data["num_raters"]

                # Overall alpha with interpretation
                col1, col2, col3 = st.columns(3)
                col1.metric("Krippendorff's Alpha", f"{alpha:.3f}")
                col2.metric("Conversations Rated", num_items)
                col3.metric("Number of Raters", num_raters)

                if alpha >= 0.8:
                    st.success(f"Alpha = {alpha:.3f} — **Excellent** agreement. Evaluations are highly reliable.")
                elif alpha >= 0.67:
                    st.warning(f"Alpha = {alpha:.3f} — **Good** agreement. Suitable for tentative conclusions.")
                else:
                    st.error(f"Alpha = {alpha:.3f} — **Poor** agreement. Evaluations may be unreliable.")

                # Per-dimension alpha
                per_dim = data.get("per_dimension_alpha", {})
                if per_dim:
                    st.subheader("Per-Dimension Agreement")
                    dim_df = pd.DataFrame([
                        {"Dimension": dim, "Alpha": val}
                        for dim, val in sorted(per_dim.items())
                    ])
                    st.dataframe(dim_df, use_container_width=True, hide_index=True)

                    fig = px.bar(
                        dim_df, x="Dimension", y="Alpha",
                        title="Krippendorff's Alpha by Dimension",
                        color="Alpha",
                        color_continuous_scale="RdYlGn",
                        range_color=[-0.5, 1.0],
                    )
                    fig.add_hline(y=0.8, line_dash="dash", line_color="green", annotation_text="Excellent (0.8)")
                    fig.add_hline(y=0.67, line_dash="dash", line_color="orange", annotation_text="Good (0.67)")
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Reliability analysis failed: {e}")
                st.info("Ensure the selected run has multiple human evaluations per conversation.")
