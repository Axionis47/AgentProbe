import json

import streamlit as st
from lib.api_client import AgentProbeClient

st.set_page_config(page_title="Scenarios - AgentProbe", layout="wide")
st.title("Scenarios")

client = AgentProbeClient()

tab_browse, tab_create = st.tabs(["Browse", "Create"])

# ---- Browse Tab ----
with tab_browse:
    col1, col2, col3 = st.columns(3)
    with col1:
        cat_filter = st.text_input("Category", placeholder="e.g. customer_support")
    with col2:
        diff_filter = st.selectbox("Difficulty", ["All", "easy", "medium", "hard"])
    with col3:
        active_filter = st.selectbox("Status", ["All", "Active", "Inactive"], key="scenario_status")

    params: dict = {"limit": 50}
    if cat_filter:
        params["category"] = cat_filter
    if diff_filter != "All":
        params["difficulty"] = diff_filter
    if active_filter == "Active":
        params["is_active"] = True
    elif active_filter == "Inactive":
        params["is_active"] = False

    try:
        data = client.list_scenarios(**params)
        scenarios = data.get("items", [])
    except Exception as e:
        st.error(f"Failed to load scenarios: {e}")
        scenarios = []
        data = {}

    st.metric("Total Scenarios", data.get("total", 0))

    for sc in scenarios:
        status = "Active" if sc["is_active"] else "Inactive"
        tags_str = ", ".join(sc.get("tags", []))
        with st.expander(f"{sc['name']} — {sc.get('difficulty', '?')} ({status}) {f'[{tags_str}]' if tags_str else ''}"):
            st.write(f"**ID:** `{sc['id']}`")
            st.write(f"**Category:** {sc.get('category') or 'N/A'}")
            st.write(f"**Description:** {sc.get('description') or 'N/A'}")
            st.write(f"**Created:** {sc['created_at']}")

            with st.expander("Turns Template"):
                st.json(sc.get("turns_template", []))
            with st.expander("User Persona"):
                st.json(sc.get("user_persona", {}))
            with st.expander("Constraints"):
                st.json(sc.get("constraints", {}))

            # Inline edit
            st.write("---")
            with st.form(f"edit_scenario_{sc['id']}"):
                st.write("**Edit Scenario:**")
                edit_name = st.text_input("Name", value=sc["name"], key=f"sn_{sc['id']}")
                edit_desc = st.text_area("Description", value=sc.get("description") or "", key=f"sd_{sc['id']}")
                edit_cat = st.text_input("Category", value=sc.get("category") or "", key=f"sc_{sc['id']}")
                edit_diff = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=["easy", "medium", "hard"].index(sc.get("difficulty", "medium")), key=f"sdf_{sc['id']}")
                edit_tags = st.text_input("Tags (comma-separated)", value=", ".join(sc.get("tags", [])), key=f"st_{sc['id']}")
                edit_turns = st.text_area("Turns Template (JSON)", value=json.dumps(sc.get("turns_template", []), indent=2), height=200, key=f"stt_{sc['id']}")
                edit_persona = st.text_area("User Persona (JSON)", value=json.dumps(sc.get("user_persona", {}), indent=2), height=100, key=f"sp_{sc['id']}")
                edit_constraints = st.text_area("Constraints (JSON)", value=json.dumps(sc.get("constraints", {}), indent=2), height=100, key=f"sco_{sc['id']}")

                if st.form_submit_button("Save Changes"):
                    try:
                        update_data = {
                            "name": edit_name,
                            "description": edit_desc or None,
                            "category": edit_cat or None,
                            "difficulty": edit_diff,
                            "tags": [t.strip() for t in edit_tags.split(",") if t.strip()],
                            "turns_template": json.loads(edit_turns),
                            "user_persona": json.loads(edit_persona),
                            "constraints": json.loads(edit_constraints),
                        }
                        client.update_scenario(sc["id"], update_data)
                        st.success("Scenario updated.")
                        st.rerun()
                    except json.JSONDecodeError:
                        st.error("Invalid JSON in one of the JSON fields.")
                    except Exception as e:
                        st.error(f"Update failed: {e}")

            if sc["is_active"]:
                if st.button("Deactivate", key=f"deact_sc_{sc['id']}"):
                    try:
                        client.delete_scenario(sc["id"])
                        st.success("Scenario deactivated.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Deactivation failed: {e}")

# ---- Create Tab ----
with tab_create:
    st.subheader("Create Scenario")

    # Format examples (shown above the form for reference)
    with st.expander("Reference Answer Format (for reference-based evaluation)"):
        st.markdown("""
Add `expected_response` to each turn in the **Turns Template** to enable
reference-based evaluation (ROUGE-1, ROUGE-L, exact-match scoring).
""")
        st.code(json.dumps([
            {"role": "user", "content": "What is the return policy?"},
            {"role": "user", "content": "Can I return after 30 days?",
             "expected_response": "Returns are accepted within 60 days of purchase with a valid receipt."},
        ], indent=2), language="json")

    with st.expander("Expected Tool Sequence Format (for trajectory evaluation)"):
        st.markdown("""
Add `expected_tool_sequence` to the **Constraints** JSON to enable trajectory
evaluation (precision, recall, order-matching of tool calls).
""")
        st.code(json.dumps({
            "expected_tool_sequence": ["search_knowledge_base", "lookup_order", "create_ticket"],
            "max_turns": 10,
        }, indent=2), language="json")

    with st.form("create_scenario"):
        name = st.text_input("Name")
        description = st.text_area("Description")
        category = st.text_input("Category", placeholder="e.g. customer_support")
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
        tags = st.text_input("Tags (comma-separated)", placeholder="multi-turn, tool-use")
        turns_template = st.text_area(
            "Turns Template (JSON array)",
            value=json.dumps([{"role": "user", "content": "Hello, I need help with..."}], indent=2),
            height=200,
        )
        user_persona = st.text_area("User Persona (JSON)", value="{}", height=100)
        constraints = st.text_area("Constraints (JSON)", value="{}", height=100)

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
                except json.JSONDecodeError:
                    st.error("Invalid JSON in one of the JSON fields.")
                except Exception as e:
                    st.error(f"Failed: {e}")
