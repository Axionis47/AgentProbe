import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lib.api_client import AgentProbeClient

st.set_page_config(page_title="Advanced - AgentProbe", layout="wide")

st.header("Advanced Analysis")

client = AgentProbeClient()

tab_elo, tab_calibration, tab_reliability = st.tabs([
    "ELO Rankings", "Calibration", "Interrater Reliability",
])

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
    try:
        scenarios_data = client.list_scenarios(limit=100, is_active=True)
        scenario_items = scenarios_data.get("items", [])
    except Exception:
        scenario_items = []

    scenario_options = {"__all__": "All Scenarios"}
    scenario_options.update({s["id"]: s["name"] for s in scenario_items})
    selected_scenario = st.selectbox(
        "Scenario",
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
        st.info("No pairwise comparisons found.")
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
            df, x="Agent", y="ELO Rating", color="Agent", text="ELO Rating",
        )
        fig.update_layout(yaxis_range=[
            min(1300, df["ELO Rating"].min() - 50),
            max(1700, df["ELO Rating"].max() + 50),
        ], showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# CALIBRATION TAB
# ============================================================
with tab_calibration:
    if not runs:
        st.info("No eval runs found.")
    else:
        cal_run_id = st.selectbox(
            "Eval Run", list(run_options.keys()),
            format_func=lambda x: run_options[x], key="cal_run",
        )

        if st.button("Compute", key="cal_btn"):
            with st.spinner("Computing..."):
                try:
                    data = client.get_calibration(cal_run_id)

                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Pearson r", f"{data['pearson_r']:.3f}")
                    col2.metric("Spearman rho", f"{data['spearman_rho']:.3f}")
                    col3.metric("MAE", f"{data['mae']:.3f}")
                    col4.metric("RMSE", f"{data['rmse']:.3f}")
                    col5.metric("Bias", f"{data['bias']:+.3f}")

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
                            mode="lines", name="Perfect",
                            line=dict(dash="dash", color="gray"),
                        ))
                        fig.update_layout(
                            xaxis_title="Model Score",
                            yaxis_title="Human Score",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    error_msg = str(e)
                    if "paired" in error_msg.lower() or "human" in error_msg.lower() or "400" in error_msg:
                        st.warning("Not enough paired human + model evaluations.")
                    else:
                        st.error(f"Failed: {e}")


# ============================================================
# INTERRATER RELIABILITY TAB
# ============================================================
with tab_reliability:
    if not runs:
        st.info("No eval runs found.")
    else:
        rel_run_id = st.selectbox(
            "Eval Run", list(run_options.keys()),
            format_func=lambda x: run_options[x], key="rel_run",
        )

        if st.button("Compute", key="rel_btn"):
            with st.spinner("Computing..."):
                try:
                    data = client.get_reliability(rel_run_id)

                    alpha = data["alpha"]
                    num_items = data["num_items"]
                    num_raters = data["num_raters"]

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Krippendorff's Alpha", f"{alpha:.3f}")
                    col2.metric("Conversations", num_items)
                    col3.metric("Raters", num_raters)

                    per_dim = data.get("per_dimension_alpha", {})
                    if per_dim:
                        dim_df = pd.DataFrame([
                            {"Dimension": dim, "Alpha": val}
                            for dim, val in sorted(per_dim.items())
                        ])
                        st.dataframe(dim_df, use_container_width=True, hide_index=True)

                        fig = px.bar(
                            dim_df, x="Dimension", y="Alpha",
                            color="Alpha",
                            color_continuous_scale="RdYlGn",
                            range_color=[-0.5, 1.0],
                        )
                        fig.add_hline(y=0.8, line_dash="dash", line_color="green")
                        fig.add_hline(y=0.67, line_dash="dash", line_color="orange")
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    error_msg = str(e)
                    if "human" in error_msg.lower() or "404" in error_msg or "not found" in error_msg.lower():
                        st.warning("No human evaluations found.")
                    else:
                        st.error(f"Failed: {e}")
