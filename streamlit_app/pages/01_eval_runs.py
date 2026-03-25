import streamlit as st
from lib.api_client import AgentProbeClient

st.set_page_config(page_title="Eval Runs - AgentProbe", layout="wide")
st.title("Eval Runs")

client = AgentProbeClient()

STATUS_BADGES = {
    "pending": "🟡",
    "running_simulation": "🔵",
    "running_evaluation": "🔵",
    "completed": "🟢",
    "failed": "🔴",
    "cancelled": "⚪",
}

tab_browse, tab_create = st.tabs(["Browse Runs", "Create New Run"])

# ---- Browse Tab ----
with tab_browse:
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox(
            "Status",
            ["All", "pending", "running_simulation", "running_evaluation", "completed", "failed"],
        )
    with col2:
        limit = st.number_input("Results per page", min_value=5, max_value=100, value=20)

    # Pre-fetch agent configs and scenarios for name resolution
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

    # Fetch runs
    try:
        params: dict = {"limit": limit}
        if status_filter != "All":
            params["status"] = status_filter
        data = client.list_eval_runs(**params)

        st.metric("Total Runs", data["total"])

        if data["items"]:
            for run in data["items"]:
                badge = STATUS_BADGES.get(run["status"], "⚪")
                run_name = run.get("name") or run["id"][:8]
                agent = agent_names.get(run.get("agent_config_id", ""), run.get("agent_config_id", "")[:8])
                scenario = scenario_names.get(run.get("scenario_id", ""), run.get("scenario_id", "")[:8])

                with st.expander(
                    f"{badge} {run_name} -- Agent: {agent} | Scenario: {scenario} | Status: {run['status']}"
                ):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.write(f"**Agent:** {agent}")
                    col2.write(f"**Scenario:** {scenario}")
                    col3.write(f"**Conversations:** {run.get('num_conversations', '?')}")
                    col4.write(f"**Status:** {badge} {run['status']}")

                    st.write(f"**ID:** `{run['id']}`")
                    st.write(f"**Created:** {run['created_at']}")
                    if run.get("started_at"):
                        st.write(f"**Started:** {run['started_at']}")
                    if run.get("completed_at"):
                        st.write(f"**Completed:** {run['completed_at']}")
                    if run.get("error_message"):
                        st.error(f"**Error:** {run['error_message']}")
                    if run.get("rubric_id"):
                        st.write(f"**Rubric ID:** `{run['rubric_id']}`")
                    if run.get("config"):
                        with st.expander("Run Config"):
                            st.json(run["config"])
        else:
            st.info("No eval runs found.")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")

# ---- Create New Run Tab ----
with tab_create:
    st.subheader("Create New Eval Run")

    # Load agent configs for selection
    try:
        configs_data = client.list_agent_configs(is_active=True, limit=100)
        configs = configs_data.get("items", [])
    except Exception as e:
        st.error(f"Failed to load agent configs: {e}")
        configs = []

    # Load scenarios for selection
    try:
        scenarios_data = client.list_scenarios(is_active=True, limit=100)
        scenarios = scenarios_data.get("items", [])
    except Exception as e:
        st.error(f"Failed to load scenarios: {e}")
        scenarios = []

    # Load rubrics for selection
    try:
        rubrics_data = client.list_rubrics(is_active=True, limit=100)
        rubrics = rubrics_data.get("items", [])
    except Exception as e:
        st.error(f"Failed to load rubrics: {e}")
        rubrics = []

    if not configs:
        st.warning("No active agent configs found. Create one in the Agent Configs page first.")
    elif not scenarios:
        st.warning("No active scenarios found. Create one in the Scenarios page first.")
    else:
        with st.form("create_eval_run"):
            run_name = st.text_input("Run Name (optional)", placeholder="e.g. Mistral vs GPT-4o on billing")

            config_options = {c["id"]: f"{c['name']} ({c['model']})" for c in configs}
            selected_config = st.selectbox(
                "Agent Config",
                options=list(config_options.keys()),
                format_func=lambda x: config_options[x],
            )

            scenario_options = {s["id"]: f"{s['name']} ({s.get('difficulty', '?')})" for s in scenarios}
            selected_scenario = st.selectbox(
                "Scenario",
                options=list(scenario_options.keys()),
                format_func=lambda x: scenario_options[x],
            )

            rubric_options = {"__none__": "Default (all dimensions)"}
            rubric_options.update({r["id"]: f"{r['name']} v{r['version']}" for r in rubrics})
            selected_rubric = st.selectbox(
                "Rubric (optional)",
                options=list(rubric_options.keys()),
                format_func=lambda x: rubric_options[x],
            )

            num_conversations = st.number_input(
                "Number of Conversations", min_value=1, max_value=100, value=5,
            )

            if st.form_submit_button("Create Eval Run"):
                payload: dict = {
                    "agent_config_id": selected_config,
                    "scenario_id": selected_scenario,
                    "num_conversations": num_conversations,
                }
                if run_name:
                    payload["name"] = run_name
                if selected_rubric != "__none__":
                    payload["rubric_id"] = selected_rubric

                try:
                    result = client.create_eval_run(payload)
                    st.success(
                        f"Eval run created! ID: {result['id'][:8]}... Status: {result['status']}. "
                        "The simulation will run asynchronously."
                    )
                except Exception as e:
                    st.error(f"Failed to create eval run: {e}")
