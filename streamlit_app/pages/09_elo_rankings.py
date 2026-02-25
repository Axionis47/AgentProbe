import streamlit as st
import pandas as pd
import plotly.express as px
from lib.api_client import AgentProbeClient

st.set_page_config(page_title="ELO Rankings - AgentProbe", layout="wide")
st.title("ELO Rankings & Pairwise Comparison")

client = AgentProbeClient()

tab_rankings, tab_compare = st.tabs(["Rankings", "Run Comparison"])

# ---- Rankings Tab ----
with tab_rankings:
    # Optional scenario filter
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
        st.info("No pairwise comparisons found. Run some comparisons in the Compare tab first.")
    else:
        # Rankings table
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

        # Bar chart
        fig = px.bar(
            df, x="Agent", y="ELO Rating", color="Agent",
            title="ELO Ratings by Agent",
            text="ELO Rating",
        )
        fig.update_layout(yaxis_range=[
            min(1300, df["ELO Rating"].min() - 50),
            max(1700, df["ELO Rating"].max() + 50),
        ])
        st.plotly_chart(fig, use_container_width=True)

        # Win/Loss breakdown
        fig2 = px.bar(
            df, x="Agent", y=["W", "L", "D"],
            title="Win/Loss/Draw Breakdown",
            barmode="stack",
            color_discrete_map={"W": "#2ecc71", "L": "#e74c3c", "D": "#95a5a6"},
        )
        st.plotly_chart(fig2, use_container_width=True)


# ---- Compare Tab ----
with tab_compare:
    st.subheader("Run Pairwise Comparison")
    st.write("Select two conversations to compare. The LLM judge will determine which agent performed better.")

    try:
        runs_data = client.list_eval_runs(limit=100)
        runs = runs_data.get("items", [])
    except Exception as e:
        st.error(f"Failed to load runs: {e}")
        st.stop()

    if not runs:
        st.info("No eval runs found.")
        st.stop()

    # Select run to pick conversations from
    run_options = {r["id"]: f"{r.get('name', r['id'][:8])} ({r['status']})" for r in runs}

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Agent A**")
        run_a = st.selectbox("Run A", list(run_options.keys()), format_func=lambda x: run_options[x], key="run_a")
        try:
            convs_a = client.list_conversations(eval_run_id=run_a, status="completed", limit=50).get("items", [])
        except Exception:
            convs_a = []
        conv_a_opts = {c["id"]: f"#{c.get('sequence_num', '?')} ({c['turn_count']} turns)" for c in convs_a}
        conv_a_id = st.selectbox("Conversation A", list(conv_a_opts.keys()), format_func=lambda x: conv_a_opts[x], key="conv_a") if conv_a_opts else None

    with col2:
        st.write("**Agent B**")
        run_b = st.selectbox("Run B", list(run_options.keys()), format_func=lambda x: run_options[x], key="run_b")
        try:
            convs_b = client.list_conversations(eval_run_id=run_b, status="completed", limit=50).get("items", [])
        except Exception:
            convs_b = []
        conv_b_opts = {c["id"]: f"#{c.get('sequence_num', '?')} ({c['turn_count']} turns)" for c in convs_b}
        conv_b_id = st.selectbox("Conversation B", list(conv_b_opts.keys()), format_func=lambda x: conv_b_opts[x], key="conv_b") if conv_b_opts else None

    if conv_a_id and conv_b_id:
        if conv_a_id == conv_b_id:
            st.warning("Select two different conversations.")
        elif st.button("Run Comparison"):
            with st.spinner("Running pairwise comparison..."):
                try:
                    result = client.create_pairwise_comparison({
                        "conversation_id_a": conv_a_id,
                        "conversation_id_b": conv_b_id,
                    })
                    winner_label = {"a": "Agent A Wins", "b": "Agent B Wins", "draw": "Draw"}
                    st.success(f"Result: **{winner_label.get(result['winner'], result['winner'])}** (confidence: {result['confidence']:.0%})")
                    st.write(f"**Reasoning:** {result['reasoning']}")

                    if result.get("dimension_preferences"):
                        st.write("**Per-Dimension Preferences:**")
                        for dim, pref in result["dimension_preferences"].items():
                            icon = {"a": "A", "b": "B", "draw": "="}
                            st.write(f"  - {dim}: **{icon.get(pref, pref)}**")
                except Exception as e:
                    st.error(f"Comparison failed: {e}")
    else:
        st.info("Select conversations from both sides to compare.")
