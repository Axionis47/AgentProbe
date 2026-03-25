import json

import streamlit as st
from lib.api_client import AgentProbeClient

st.set_page_config(page_title="Admin - AgentProbe", layout="wide")

st.header("Admin")
st.caption("Manage agent configs, scenarios, rubrics, and launch new evaluation runs.")

client = AgentProbeClient()

tab_agents, tab_scenarios, tab_rubrics, tab_run = st.tabs([
    "Agent Configs", "Scenarios", "Rubrics", "Run Evaluation",
])

# ============================================================
# AGENT CONFIGS TAB
# ============================================================
with tab_agents:
    st.subheader("Agent Configurations")

    # --- List existing ---
    try:
        data = client.list_agent_configs(limit=50)
        configs = data.get("items", [])
    except Exception as e:
        st.error(f"Failed to load configs: {e}")
        configs = []

    if configs:
        for cfg in configs:
            status_icon = "Active" if cfg["is_active"] else "Inactive"
            with st.expander(f"{cfg['name']} -- {cfg['model']} ({status_icon})"):
                st.write(f"**Temperature:** {cfg['temperature']} | **Max Tokens:** {cfg['max_tokens']}")
                st.write(f"**Created:** {cfg['created_at']}")
                with st.expander("System Prompt"):
                    st.code(cfg["system_prompt"], language="text")
                if cfg.get("tools"):
                    with st.expander("Tools"):
                        st.json(cfg["tools"])
    else:
        st.info("No agent configs found.")

    # --- Create new ---
    st.divider()
    st.subheader("Create New Agent Config")
    with st.form("create_agent"):
        name = st.text_input("Name")
        description = st.text_area("Description")
        model = st.selectbox("Model", [
            "ollama/mistral:7b-instruct",
            "ollama/llama3:8b",
            "ollama/codellama:7b",
            "claude-sonnet-4-20250514",
            "gpt-4o",
            "vertex_ai/gemini-1.5-pro",
            "vertex_ai/gemini-1.5-flash",
            "vertex_ai/gemini-2.0-flash",
            "vertex_ai/gemini-2.5-pro-preview-05-06",
        ], index=0)
        system_prompt = st.text_area("System Prompt", height=200, placeholder="You are a helpful assistant...")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.number_input("Max Tokens", min_value=1, max_value=200000, value=4096)
        tools_json = st.text_area("Tools (JSON array)", value="[]", height=100)

        if st.form_submit_button("Create Agent Config"):
            if not name or not system_prompt:
                st.error("Name and system prompt are required.")
            else:
                try:
                    tools = json.loads(tools_json)
                    result = client.create_agent_config({
                        "name": name,
                        "description": description or None,
                        "model": model,
                        "system_prompt": system_prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "tools": tools,
                    })
                    st.success(f"Created: {result['name']} (ID: {result['id'][:8]}...)")
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON in tools field.")
                except Exception as e:
                    st.error(f"Failed: {e}")


# ============================================================
# SCENARIOS TAB
# ============================================================
with tab_scenarios:
    st.subheader("Test Scenarios")

    # --- List existing ---
    try:
        data = client.list_scenarios(limit=50)
        scenarios = data.get("items", [])
    except Exception as e:
        st.error(f"Failed to load scenarios: {e}")
        scenarios = []

    if scenarios:
        for sc in scenarios:
            status = "Active" if sc["is_active"] else "Inactive"
            tags_str = ", ".join(sc.get("tags", []))
            label = f"{sc['name']} -- {sc.get('difficulty', '?')} ({status})"
            if tags_str:
                label += f" [{tags_str}]"
            with st.expander(label):
                st.write(f"**Category:** {sc.get('category') or 'N/A'}")
                st.write(f"**Description:** {sc.get('description') or 'N/A'}")
                with st.expander("Turns Template"):
                    st.json(sc.get("turns_template", []))
                with st.expander("User Persona"):
                    st.json(sc.get("user_persona", {}))
                with st.expander("Constraints"):
                    st.json(sc.get("constraints", {}))
    else:
        st.info("No scenarios found.")

    # --- Create new ---
    st.divider()
    st.subheader("Create New Scenario")
    with st.form("create_scenario"):
        name = st.text_input("Name", key="sc_name")
        description = st.text_area("Description", key="sc_desc")
        category = st.text_input("Category", placeholder="e.g. customer_support", key="sc_cat")
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1, key="sc_diff")
        tags = st.text_input("Tags (comma-separated)", placeholder="multi-turn, tool-use", key="sc_tags")
        turns_template = st.text_area(
            "Turns Template (JSON array)",
            value=json.dumps([{"role": "user", "content": "Hello, I need help with..."}], indent=2),
            height=200,
            key="sc_turns",
        )
        user_persona = st.text_area("User Persona (JSON)", value="{}", height=100, key="sc_persona")
        constraints = st.text_area("Constraints (JSON)", value="{}", height=100, key="sc_constraints")

        if st.form_submit_button("Create Scenario"):
            if not name:
                st.error("Name is required.")
            else:
                try:
                    result = client.create_scenario({
                        "name": name,
                        "description": description or None,
                        "category": category or None,
                        "turns_template": json.loads(turns_template),
                        "user_persona": json.loads(user_persona),
                        "constraints": json.loads(constraints),
                        "difficulty": difficulty,
                        "tags": [t.strip() for t in tags.split(",") if t.strip()],
                    })
                    st.success(f"Created: {result['name']} (ID: {result['id'][:8]}...)")
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON in one of the JSON fields.")
                except Exception as e:
                    st.error(f"Failed: {e}")


# ============================================================
# RUBRICS TAB
# ============================================================
with tab_rubrics:
    st.subheader("Evaluation Rubrics")

    # --- List existing ---
    try:
        data = client.list_rubrics(limit=50)
        rubrics = data.get("items", [])
    except Exception as e:
        st.error(f"Failed to load rubrics: {e}")
        rubrics = []

    if rubrics:
        for rubric in rubrics:
            label = f"{rubric['name']} (v{rubric['version']}) -- {'Active' if rubric['is_active'] else 'Inactive'}"
            with st.expander(label):
                st.write(f"**Description:** {rubric.get('description') or 'N/A'}")
                st.write("**Dimensions:**")
                for dim in rubric.get("dimensions", []):
                    st.write(f"- **{dim.get('name', '?')}** (weight: {dim.get('weight', '?')}): {dim.get('description', '')}")
    else:
        st.info("No rubrics found.")

    # --- Create new ---
    st.divider()
    st.subheader("Create New Rubric")
    with st.form("create_rubric"):
        name = st.text_input("Rubric Name", key="rb_name")
        description = st.text_area("Description", key="rb_desc")
        num_dims = st.number_input("Number of Dimensions", min_value=1, max_value=20, value=5, key="rb_ndims")

        st.write("**Define Dimensions:**")
        dimensions = []
        for i in range(int(num_dims)):
            st.write(f"--- Dimension {i + 1} ---")
            c1, c2 = st.columns([3, 1])
            with c1:
                dim_name = st.text_input("Name", key=f"rb_dim_name_{i}", placeholder="e.g. helpfulness")
                dim_desc = st.text_input("Description", key=f"rb_dim_desc_{i}", placeholder="How helpful is the response?")
            with c2:
                dim_weight = st.number_input(
                    "Weight", min_value=0.0, max_value=1.0,
                    value=round(1.0 / num_dims, 2), step=0.05,
                    key=f"rb_dim_weight_{i}",
                )
            dim_criteria = st.text_input(
                "Criteria (comma-separated)", key=f"rb_dim_criteria_{i}",
                placeholder="addresses question, provides examples",
            )
            dimensions.append({
                "name": dim_name,
                "description": dim_desc,
                "weight": dim_weight,
                "criteria": [c.strip() for c in dim_criteria.split(",") if c.strip()] if dim_criteria else [],
            })

        if st.form_submit_button("Create Rubric"):
            if not name:
                st.error("Name is required.")
            elif not all(d["name"] for d in dimensions):
                st.error("All dimensions must have a name.")
            else:
                try:
                    result = client.create_rubric({
                        "name": name,
                        "description": description or None,
                        "dimensions": dimensions,
                    })
                    st.success(f"Rubric created: {result['name']} v{result['version']} (ID: {result['id'][:8]}...)")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to create rubric: {e}")


# ============================================================
# RUN EVALUATION TAB
# ============================================================
with tab_run:
    st.subheader("Launch New Evaluation Run")
    st.info(
        "An eval run pairs an agent config with a scenario, simulates multiple conversations, "
        "and scores them with automated judges. It runs asynchronously in the background."
    )

    # Load options
    try:
        configs_data = client.list_agent_configs(is_active=True, limit=100)
        run_configs = configs_data.get("items", [])
    except Exception:
        run_configs = []

    try:
        scenarios_data = client.list_scenarios(is_active=True, limit=100)
        run_scenarios = scenarios_data.get("items", [])
    except Exception:
        run_scenarios = []

    try:
        rubrics_data = client.list_rubrics(is_active=True, limit=100)
        run_rubrics = rubrics_data.get("items", [])
    except Exception:
        run_rubrics = []

    if not run_configs:
        st.warning("No active agent configs found. Create one in the Agent Configs tab first.")
    elif not run_scenarios:
        st.warning("No active scenarios found. Create one in the Scenarios tab first.")
    else:
        with st.form("create_eval_run"):
            run_name = st.text_input("Run Name (optional)", placeholder="e.g. Mistral vs GPT-4o on billing")

            config_options = {c["id"]: f"{c['name']} ({c['model']})" for c in run_configs}
            selected_config = st.selectbox(
                "Agent Config",
                options=list(config_options.keys()),
                format_func=lambda x: config_options[x],
            )

            scenario_options = {s["id"]: f"{s['name']} ({s.get('difficulty', '?')})" for s in run_scenarios}
            selected_scenario = st.selectbox(
                "Scenario",
                options=list(scenario_options.keys()),
                format_func=lambda x: scenario_options[x],
            )

            rubric_options = {"__none__": "Default (all dimensions)"}
            rubric_options.update({r["id"]: f"{r['name']} v{r['version']}" for r in run_rubrics})
            selected_rubric = st.selectbox(
                "Rubric (optional)",
                options=list(rubric_options.keys()),
                format_func=lambda x: rubric_options[x],
            )

            num_conversations = st.number_input(
                "Number of Conversations", min_value=1, max_value=100, value=5,
            )

            if st.form_submit_button("Launch Eval Run"):
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
