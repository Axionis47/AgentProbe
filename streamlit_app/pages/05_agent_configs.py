import json

import streamlit as st
from lib.api_client import AgentProbeClient

st.set_page_config(page_title="Agent Configs - AgentProbe", layout="wide")
st.title("Agent Configs")

client = AgentProbeClient()

tab_browse, tab_create = st.tabs(["Browse", "Create"])

# ---- Browse Tab ----
with tab_browse:
    col1, col2 = st.columns(2)
    with col1:
        active_filter = st.selectbox("Status", ["All", "Active", "Inactive"])
    with col2:
        model_filter = st.text_input("Filter by Model", placeholder="e.g. ollama/mistral:7b-instruct")

    params: dict = {"limit": 50}
    if active_filter == "Active":
        params["is_active"] = True
    elif active_filter == "Inactive":
        params["is_active"] = False
    if model_filter:
        params["model"] = model_filter

    try:
        data = client.list_agent_configs(**params)
        configs = data.get("items", [])
    except Exception as e:
        st.error(f"Failed to load configs: {e}")
        configs = []

    st.metric("Total Configs", data.get("total", 0) if configs or data else 0)

    for cfg in configs:
        status_icon = "Active" if cfg["is_active"] else "Inactive"
        with st.expander(f"{cfg['name']} — {cfg['model']} ({status_icon})"):
            st.write(f"**ID:** `{cfg['id']}`")
            st.write(f"**Temperature:** {cfg['temperature']} | **Max Tokens:** {cfg['max_tokens']}")
            st.write(f"**Created:** {cfg['created_at']}")

            with st.expander("System Prompt"):
                st.code(cfg["system_prompt"], language="text")

            if cfg.get("tools"):
                with st.expander("Tools"):
                    st.json(cfg["tools"])

            # Inline edit
            st.write("---")
            with st.form(f"edit_{cfg['id']}"):
                st.write("**Edit Config:**")
                edit_name = st.text_input("Name", value=cfg["name"], key=f"en_{cfg['id']}")
                edit_desc = st.text_area("Description", value=cfg.get("description") or "", key=f"ed_{cfg['id']}")
                edit_model = st.text_input("Model", value=cfg["model"], key=f"em_{cfg['id']}")
                edit_prompt = st.text_area("System Prompt", value=cfg["system_prompt"], height=150, key=f"ep_{cfg['id']}")
                edit_temp = st.slider("Temperature", 0.0, 2.0, cfg["temperature"], 0.1, key=f"et_{cfg['id']}")
                edit_max = st.number_input("Max Tokens", min_value=1, max_value=200000, value=cfg["max_tokens"], key=f"emx_{cfg['id']}")

                c1, c2 = st.columns(2)
                with c1:
                    if st.form_submit_button("Save Changes"):
                        try:
                            client.update_agent_config(cfg["id"], {
                                "name": edit_name,
                                "description": edit_desc or None,
                                "model": edit_model,
                                "system_prompt": edit_prompt,
                                "temperature": edit_temp,
                                "max_tokens": edit_max,
                            })
                            st.success("Config updated.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Update failed: {e}")

            if cfg["is_active"]:
                if st.button("Deactivate", key=f"deact_{cfg['id']}"):
                    try:
                        client.delete_agent_config(cfg["id"])
                        st.success("Config deactivated.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Deactivation failed: {e}")

# ---- Create Tab ----
with tab_create:
    st.subheader("Create Agent Config")

    with st.form("create_agent"):
        name = st.text_input("Name")
        description = st.text_area("Description")
        model = st.selectbox("Model", [
            "ollama/mistral:7b-instruct",
            "ollama/llama3:8b",
            "ollama/codellama:7b",
            "claude-sonnet-4-20250514",
            "gpt-4o",
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
                except json.JSONDecodeError:
                    st.error("Invalid JSON in tools field.")
                except Exception as e:
                    st.error(f"Failed: {e}")
