import json

import streamlit as st
from lib.api_client import AgentProbeClient

st.set_page_config(page_title="Admin - AgentProbe", layout="wide")

st.header("Admin")

client = AgentProbeClient()

tab_agents, tab_scenarios, tab_rubrics, tab_run = st.tabs([
    "Agent Configs", "Scenarios", "Rubrics", "Run Evaluation",
])

# ============================================================
# AGENT CONFIGS TAB
# ============================================================
with tab_agents:
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
                st.code(cfg["system_prompt"], language="text")
                if cfg.get("tools"):
                    st.json(cfg["tools"])

    st.markdown("")
    with st.form("create_agent"):
        st.subheader("New Agent Config")
        name = st.text_input("Name")
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
        system_prompt = st.text_area("System Prompt", height=200)
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.number_input("Max Tokens", min_value=1, max_value=200000, value=4096)
        tools_json = st.text_area("Tools (JSON)", value="[]", height=100)

        if st.form_submit_button("Create"):
            if not name or not system_prompt:
                st.error("Name and system prompt required.")
            else:
                try:
                    tools = json.loads(tools_json)
                    result = client.create_agent_config({
                        "name": name,
                        "model": model,
                        "system_prompt": system_prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "tools": tools,
                    })
                    st.success(f"Created: {result['name']}")
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON in tools field.")
                except Exception as e:
                    st.error(f"Failed: {e}")


# ============================================================
# SCENARIOS TAB
# ============================================================
with tab_scenarios:
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
                st.json(sc.get("turns_template", []))

    st.markdown("")
    with st.form("create_scenario"):
        st.subheader("New Scenario")
        name = st.text_input("Name", key="sc_name")
        category = st.text_input("Category", key="sc_cat")
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1, key="sc_diff")
        tags = st.text_input("Tags (comma-separated)", key="sc_tags")
        turns_template = st.text_area(
            "Turns Template (JSON)",
            value=json.dumps([{"role": "user", "content": "Hello, I need help with..."}], indent=2),
            height=200,
            key="sc_turns",
        )
        user_persona = st.text_area("User Persona (JSON)", value="{}", height=100, key="sc_persona")
        constraints = st.text_area("Constraints (JSON)", value="{}", height=100, key="sc_constraints")

        if st.form_submit_button("Create"):
            if not name:
                st.error("Name is required.")
            else:
                try:
                    result = client.create_scenario({
                        "name": name,
                        "category": category or None,
                        "turns_template": json.loads(turns_template),
                        "user_persona": json.loads(user_persona),
                        "constraints": json.loads(constraints),
                        "difficulty": difficulty,
                        "tags": [t.strip() for t in tags.split(",") if t.strip()],
                    })
                    st.success(f"Created: {result['name']}")
                    st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid JSON.")
                except Exception as e:
                    st.error(f"Failed: {e}")


# ============================================================
# RUBRICS TAB
# ============================================================
with tab_rubrics:
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
                for dim in rubric.get("dimensions", []):
                    st.write(f"- **{dim.get('name', '?')}** (weight: {dim.get('weight', '?')})")

    st.markdown("")
    with st.form("create_rubric"):
        st.subheader("New Rubric")
        name = st.text_input("Name", key="rb_name")
        description = st.text_area("Description", key="rb_desc")
        num_dims = st.number_input("Dimensions", min_value=1, max_value=20, value=5, key="rb_ndims")

        dimensions = []
        for i in range(int(num_dims)):
            c1, c2 = st.columns([3, 1])
            with c1:
                dim_name = st.text_input(f"Dim {i+1} Name", key=f"rb_dim_name_{i}")
                dim_desc = st.text_input(f"Dim {i+1} Description", key=f"rb_dim_desc_{i}")
            with c2:
                dim_weight = st.number_input(
                    "Weight", min_value=0.0, max_value=1.0,
                    value=round(1.0 / num_dims, 2), step=0.05,
                    key=f"rb_dim_weight_{i}",
                )
            dim_criteria = st.text_input(
                f"Dim {i+1} Criteria (comma-separated)", key=f"rb_dim_criteria_{i}",
            )
            dimensions.append({
                "name": dim_name,
                "description": dim_desc,
                "weight": dim_weight,
                "criteria": [c.strip() for c in dim_criteria.split(",") if c.strip()] if dim_criteria else [],
            })

        if st.form_submit_button("Create"):
            if not name:
                st.error("Name is required.")
            elif not all(d["name"] for d in dimensions):
                st.error("All dimensions need a name.")
            else:
                try:
                    result = client.create_rubric({
                        "name": name,
                        "description": description or None,
                        "dimensions": dimensions,
                    })
                    st.success(f"Created: {result['name']} v{result['version']}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")


# ============================================================
# RUN EVALUATION TAB
# ============================================================
with tab_run:
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
        st.warning("No active agent configs.")
    elif not run_scenarios:
        st.warning("No active scenarios.")
    else:
        with st.form("create_eval_run"):
            run_name = st.text_input("Run Name (optional)")

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

            rubric_options = {"__none__": "Default"}
            rubric_options.update({r["id"]: f"{r['name']} v{r['version']}" for r in run_rubrics})
            selected_rubric = st.selectbox(
                "Rubric",
                options=list(rubric_options.keys()),
                format_func=lambda x: rubric_options[x],
            )

            num_conversations = st.number_input(
                "Conversations", min_value=1, max_value=100, value=5,
            )

            if st.form_submit_button("Launch"):
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
                    st.success(f"Created run {result['id'][:8]}... ({result['status']})")
                except Exception as e:
                    st.error(f"Failed: {e}")
