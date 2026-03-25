import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lib.api_client import AgentProbeClient

st.set_page_config(page_title="Advanced - AgentProbe", layout="wide")

st.header("Advanced Analysis")
st.caption("ELO rankings, calibration, and interrater reliability.")

st.warning(
    "These features require human ratings and pairwise comparisons to produce meaningful results. "
    "If you haven't submitted human ratings on the Rate page or run pairwise comparisons, "
    "the results here will be empty or incomplete."
)

client = AgentProbeClient()

tab_elo, tab_calibration, tab_reliability = st.tabs([
    "ELO Rankings", "Calibration", "Interrater Reliability",
])

# Shared: load eval runs
try:
    runs_data = client.list_eval_runs(limit=100)
    runs = runs_data.get("items", [])
except Exception as e:
    st.error(f"Failed to load eval runs: {e}")
    st.stop()

run_options = {r["id"]: f"{r.get('name', r['id'][:8])} ({r['status']})" for r in runs}


# ============================================================
# ELO RANKINGS TAB
# ============================================================
with tab_elo:
    st.subheader("ELO Rankings")
    st.caption("Head-to-head agent rankings based on pairwise LLM-judge comparisons.")

    # Scenario filter
    try:
        scenarios_data = client.list_scenarios(limit=100, is_active=True)
        scenario_items = scenarios_data.get("items", [])
    except Exception:
        scenario_items = []

    scenario_options = {"__all__": "All Scenarios"}
    scenario_options.update({s["id"]: s["name"] for s in scenario_items})
    selected_scenario = st.selectbox(
        "Filter by Scenario",
        options=list(scenario_options.keys()),
        format_func=lambda x: scenario_options[x],
    )

    scenario_id = None if selected_scenario == "__all__" else selected_scenario

    try:
        data = client.get_rankings(scenario_id=scenario_id)
        rankings = data.get("rankings", [])
        total_matches = data.get("total_matches", 0)
    except Exception as e:
        st.error(f"Failed to load rankings: {e}")
        rankings = []
        total_matches = 0

    st.metric("Total Matches", total_matches)

    if not rankings:
        st.info(
            "No pairwise comparisons found. To populate this page, run pairwise comparisons "
            "by selecting two conversations from different agents and letting the LLM judge decide."
        )
    else:
        df = pd.DataFrame([
            {
                "Rank": i + 1,
                "Agent": r.get("agent_name") or r["agent_config_id"][:8],
                "ELO Rating": r["elo_rating"],
                "Matches": r["matches_played"],
                "W": r["wins"],
                "L": r["losses"],
                "D": r["draws"],
            }
            for i, r in enumerate(rankings)
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        fig = px.bar(
            df, x="Agent", y="ELO Rating", color="Agent",
            title="ELO Ratings by Agent", text="ELO Rating",
        )
        fig.update_layout(yaxis_range=[
            min(1300, df["ELO Rating"].min() - 50),
            max(1700, df["ELO Rating"].max() + 50),
        ])
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# CALIBRATION TAB
# ============================================================
with tab_calibration:
    st.subheader("Model Judge vs Human Score Agreement")
    st.caption("Measures how well automated model-judge scores predict human scores.")

    if not runs:
        st.info("No eval runs found.")
    else:
        cal_run_id = st.selectbox(
            "Select Eval Run", list(run_options.keys()),
            format_func=lambda x: run_options[x], key="cal_run",
        )

        if st.button("Compute Calibration", key="cal_btn"):
            with st.spinner("Computing calibration metrics..."):
                try:
                    data = client.get_calibration(cal_run_id)

                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Pearson r", f"{data['pearson_r']:.3f}")
                    col2.metric("Spearman rho", f"{data['spearman_rho']:.3f}")
                    col3.metric("MAE", f"{data['mae']:.3f}")
                    col4.metric("RMSE", f"{data['rmse']:.3f}")
                    col5.metric("Bias", f"{data['bias']:+.3f}")

                    st.write(f"Based on **{data['n']}** paired human + model evaluations.")

                    r = data["pearson_r"]
                    if r > 0.8:
                        st.success("Strong correlation -- model judge is well-calibrated with humans.")
                    elif r > 0.5:
                        st.warning("Moderate correlation -- model judge partially agrees with humans.")
                    else:
                        st.error("Weak correlation -- model judge scores diverge from human scores.")

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
                        min_val = min(curve_df["avg_model"].min(), curve_df["avg_human"].min())
                        max_val = max(curve_df["avg_model"].max(), curve_df["avg_human"].max())
                        fig.add_trace(go.Scatter(
                            x=[min_val, max_val], y=[min_val, max_val],
                            mode="lines", name="Perfect Calibration",
                            line=dict(dash="dash", color="gray"),
                        ))
                        fig.update_layout(
                            title="Calibration Curve",
                            xaxis_title="Model Judge Score",
                            yaxis_title="Human Score",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    error_msg = str(e)
                    if "paired" in error_msg.lower() or "human" in error_msg.lower() or "400" in error_msg:
                        st.warning(
                            "Not enough paired human + model judge evaluations found. "
                            "Go to the Rate page and score some conversations first, then try again."
                        )
                    else:
                        st.error(f"Calibration analysis failed: {e}")


# ============================================================
# INTERRATER RELIABILITY TAB
# ============================================================
with tab_reliability:
    st.subheader("Interrater Reliability (Krippendorff's Alpha)")
    st.caption("Measures agreement among multiple human evaluators scoring the same conversations.")

    if not runs:
        st.info("No eval runs found.")
    else:
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

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Krippendorff's Alpha", f"{alpha:.3f}")
                    col2.metric("Conversations Rated", num_items)
                    col3.metric("Number of Raters", num_raters)

                    if alpha >= 0.8:
                        st.success(f"Alpha = {alpha:.3f} -- Excellent agreement.")
                    elif alpha >= 0.67:
                        st.warning(f"Alpha = {alpha:.3f} -- Good agreement.")
                    else:
                        st.error(f"Alpha = {alpha:.3f} -- Poor agreement.")

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
                    error_msg = str(e)
                    if "human" in error_msg.lower() or "404" in error_msg or "not found" in error_msg.lower():
                        st.warning(
                            "No human evaluations found. Have multiple evaluators score "
                            "conversations on the Rate page first, then try again."
                        )
                    else:
                        st.error(f"Reliability analysis failed: {e}")
