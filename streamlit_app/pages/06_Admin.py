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

                col_edit, col_del = st.columns([1, 1])
                with col_edit:
                    edit_open = st.checkbox(
                        "Edit",
                        key=f"agent_edit_open_{cfg['id']}",
                        value=False,
                    )
                with col_del:
                    confirm_key = f"agent_confirm_delete_{cfg['id']}"
                    if st.session_state.get(confirm_key):
                        if st.button("Confirm delete (deactivate)", key=f"agent_del_yes_{cfg['id']}"):
                            try:
                                client.delete_agent_config(cfg["id"])
                                st.session_state.pop(confirm_key, None)
                                st.success("Deactivated.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")
                        if st.button("Cancel", key=f"agent_del_no_{cfg['id']}"):
                            st.session_state.pop(confirm_key, None)
                            st.rerun()
                    else:
                        if st.button("Delete", key=f"agent_del_{cfg['id']}"):
                            st.session_state[confirm_key] = True
                            st.rerun()

                if edit_open:
                    with st.form(f"agent_edit_form_{cfg['id']}"):
                        e_name = st.text_input("Name", value=cfg["name"])
                        e_model = st.text_input("Model", value=cfg["model"])
                        e_system = st.text_area(
                            "System Prompt", value=cfg["system_prompt"], height=200
                        )
                        e_temp = st.slider(
                            "Temperature", 0.0, 2.0, float(cfg["temperature"]), 0.1
                        )
                        e_maxtok = st.number_input(
                            "Max Tokens", min_value=1, max_value=200000,
                            value=int(cfg["max_tokens"]),
                        )
                        e_tools = st.text_area(
                            "Tools (JSON)", value=json.dumps(cfg.get("tools", []), indent=2),
                            height=120,
                        )
                        e_active = st.checkbox("Active", value=bool(cfg["is_active"]))
                        if st.form_submit_button("Save"):
                            try:
                                update = {
                                    "name": e_name,
                                    "model": e_model,
                                    "system_prompt": e_system,
                                    "temperature": e_temp,
                                    "max_tokens": e_maxtok,
                                    "tools": json.loads(e_tools),
                                    "is_active": e_active,
                                }
                                client.update_agent_config(cfg["id"], update)
                                st.success("Saved.")
                                st.rerun()
                            except json.JSONDecodeError:
                                st.error("Invalid JSON in tools.")
                            except Exception as e:
                                st.error(f"Failed: {e}")

    st.markdown("")
    with st.form("create_agent"):
        st.subheader("New Agent Config")
        name = st.text_input("Name")
        agent_type_label = st.radio(
            "Agent Type",
            options=["Built-in (LLM)", "External (your API)"],
            index=0,
            key="agent_type_radio",
        )
        agent_type = "builtin" if agent_type_label == "Built-in (LLM)" else "external"
        endpoint_url = ""
        if agent_type == "external":
            endpoint_url = st.text_input("Endpoint URL", placeholder="https://your-api.example.com/chat")
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
            elif agent_type == "external" and not endpoint_url:
                st.error("Endpoint URL is required for external agents.")
            else:
                try:
                    tools = json.loads(tools_json)
                    payload = {
                        "name": name,
                        "model": model,
                        "system_prompt": system_prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "tools": tools,
                        "agent_type": agent_type,
                    }
                    if endpoint_url:
                        payload["endpoint_url"] = endpoint_url
                    result = client.create_agent_config(payload)
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

                col_edit, col_del = st.columns([1, 1])
                with col_edit:
                    sc_edit_open = st.checkbox(
                        "Edit", key=f"scenario_edit_open_{sc['id']}", value=False
                    )
                with col_del:
                    sc_confirm_key = f"scenario_confirm_delete_{sc['id']}"
                    if st.session_state.get(sc_confirm_key):
                        if st.button("Confirm delete", key=f"sc_del_yes_{sc['id']}"):
                            try:
                                client.delete_scenario(sc["id"])
                                st.session_state.pop(sc_confirm_key, None)
                                st.success("Deactivated.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")
                        if st.button("Cancel", key=f"sc_del_no_{sc['id']}"):
                            st.session_state.pop(sc_confirm_key, None)
                            st.rerun()
                    else:
                        if st.button("Delete", key=f"sc_del_{sc['id']}"):
                            st.session_state[sc_confirm_key] = True
                            st.rerun()

                if sc_edit_open:
                    with st.form(f"sc_edit_form_{sc['id']}"):
                        e_name = st.text_input("Name", value=sc["name"])
                        e_cat = st.text_input("Category", value=sc.get("category") or "")
                        e_diff = st.selectbox(
                            "Difficulty",
                            options=["easy", "medium", "hard"],
                            index=["easy", "medium", "hard"].index(
                                sc.get("difficulty") or "medium"
                            ),
                        )
                        e_tags = st.text_input(
                            "Tags (comma-separated)",
                            value=", ".join(sc.get("tags", [])),
                        )
                        e_turns = st.text_area(
                            "Turns Template (JSON)",
                            value=json.dumps(sc.get("turns_template", []), indent=2),
                            height=180,
                        )
                        e_persona = st.text_area(
                            "User Persona (JSON)",
                            value=json.dumps(sc.get("user_persona", {}), indent=2),
                            height=80,
                        )
                        e_constraints = st.text_area(
                            "Constraints (JSON)",
                            value=json.dumps(sc.get("constraints", {}), indent=2),
                            height=80,
                        )
                        e_active = st.checkbox("Active", value=bool(sc["is_active"]))
                        if st.form_submit_button("Save"):
                            try:
                                update = {
                                    "name": e_name,
                                    "category": e_cat or None,
                                    "difficulty": e_diff,
                                    "tags": [t.strip() for t in e_tags.split(",") if t.strip()],
                                    "turns_template": json.loads(e_turns),
                                    "user_persona": json.loads(e_persona),
                                    "constraints": json.loads(e_constraints),
                                    "is_active": e_active,
                                }
                                client.update_scenario(sc["id"], update)
                                st.success("Saved.")
                                st.rerun()
                            except json.JSONDecodeError:
                                st.error("Invalid JSON.")
                            except Exception as e:
                                st.error(f"Failed: {e}")

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

                col_edit, col_versions, col_del = st.columns([1, 1, 1])
                with col_edit:
                    rb_edit_open = st.checkbox(
                        "Edit", key=f"rubric_edit_open_{rubric['id']}", value=False
                    )
                with col_versions:
                    rb_ver_open = st.checkbox(
                        "Versions", key=f"rubric_ver_open_{rubric['id']}", value=False
                    )
                with col_del:
                    rb_confirm_key = f"rubric_confirm_delete_{rubric['id']}"
                    if st.session_state.get(rb_confirm_key):
                        if st.button("Confirm delete", key=f"rb_del_yes_{rubric['id']}"):
                            try:
                                client.delete_rubric(rubric["id"])
                                st.session_state.pop(rb_confirm_key, None)
                                st.success("Deleted.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")
                        if st.button("Cancel", key=f"rb_del_no_{rubric['id']}"):
                            st.session_state.pop(rb_confirm_key, None)
                            st.rerun()
                    else:
                        if st.button("Delete", key=f"rb_del_{rubric['id']}"):
                            st.session_state[rb_confirm_key] = True
                            st.rerun()

                if rb_ver_open:
                    try:
                        versions = client.list_rubric_versions(rubric["id"])
                        for v in versions:
                            v_active = "active" if v.get("is_active") else "archived"
                            st.write(
                                f"- v{v['version']} ({v_active}) — created {v.get('created_at', '')}"
                            )
                    except Exception as e:
                        st.warning(f"Could not load versions: {e}")

                if rb_edit_open:
                    with st.form(f"rb_edit_form_{rubric['id']}"):
                        e_name = st.text_input("Name", value=rubric["name"])
                        e_desc = st.text_area(
                            "Description", value=rubric.get("description") or ""
                        )
                        e_dims = st.text_area(
                            "Dimensions (JSON)",
                            value=json.dumps(rubric.get("dimensions", []), indent=2),
                            height=200,
                        )
                        e_active = st.checkbox(
                            "Active", value=bool(rubric["is_active"])
                        )
                        if st.form_submit_button("Save"):
                            try:
                                update = {
                                    "name": e_name,
                                    "description": e_desc or None,
                                    "dimensions": json.loads(e_dims),
                                    "is_active": e_active,
                                }
                                client.update_rubric(rubric["id"], update)
                                st.success("Saved.")
                                st.rerun()
                            except json.JSONDecodeError:
                                st.error("Invalid JSON.")
                            except Exception as e:
                                st.error(f"Failed: {e}")

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
