import json

import streamlit as st
from lib.api_client import AgentProbeClient

st.set_page_config(page_title="Rubric Editor - AgentProbe", layout="wide")
st.title("Rubric Editor")

client = AgentProbeClient()

tab_browse, tab_create = st.tabs(["Browse", "Create"])

# ---- Browse Tab ----
with tab_browse:
    try:
        data = client.list_rubrics(limit=50)
        rubrics = data.get("items", [])
    except Exception as e:
        st.error(f"Failed to load rubrics: {e}")
        rubrics = []

    if not rubrics:
        st.info("No rubrics found. Create one in the Create tab.")
    else:
        st.metric("Total Rubrics", data["total"])
        for rubric in rubrics:
            label = f"{rubric['name']} (v{rubric['version']}) — {'Active' if rubric['is_active'] else 'Inactive'}"
            with st.expander(label):
                st.write(f"**Description:** {rubric.get('description') or 'N/A'}")
                st.write(f"**Created:** {rubric['created_at']}")
                st.write(f"**ID:** `{rubric['id']}`")

                # Show dimensions
                st.write("**Dimensions:**")
                for dim in rubric.get("dimensions", []):
                    st.write(f"- **{dim.get('name', '?')}** (weight: {dim.get('weight', '?')}): {dim.get('description', '')}")

                # Version history
                if st.button("Show Version History", key=f"versions_{rubric['id']}"):
                    try:
                        versions = client.get_rubric_versions(rubric["id"])
                        for v in versions:
                            st.write(f"  v{v['version']} — {v['created_at']} (`{v['id'][:8]}...`)")
                    except Exception as e:
                        st.warning(f"Could not load versions: {e}")

                # Create new version inline
                st.write("---")
                st.write("**Create New Version:**")
                with st.form(f"new_version_{rubric['id']}"):
                    new_name = st.text_input("Name", value=rubric["name"], key=f"vn_{rubric['id']}")
                    new_desc = st.text_area("Description", value=rubric.get("description") or "", key=f"vd_{rubric['id']}")
                    new_dims = st.text_area(
                        "Dimensions (JSON)",
                        value=json.dumps(rubric.get("dimensions", []), indent=2),
                        height=200,
                        key=f"vdims_{rubric['id']}",
                    )
                    if st.form_submit_button("Create New Version"):
                        try:
                            parsed_dims = json.loads(new_dims)
                            result = client.update_rubric(rubric["id"], {
                                "name": new_name,
                                "description": new_desc or None,
                                "dimensions": parsed_dims,
                            })
                            st.success(f"Created v{result['version']} (ID: {result['id'][:8]}...)")
                            st.rerun()
                        except json.JSONDecodeError:
                            st.error("Invalid JSON in dimensions field.")
                        except Exception as e:
                            st.error(f"Failed to create version: {e}")

# ---- Create Tab ----
with tab_create:
    st.subheader("Create New Rubric")

    with st.form("create_rubric"):
        name = st.text_input("Rubric Name")
        description = st.text_area("Description")
        num_dims = st.number_input("Number of Dimensions", min_value=1, max_value=20, value=5)

        st.write("**Define Dimensions:**")
        dimensions = []
        for i in range(int(num_dims)):
            st.write(f"--- Dimension {i + 1} ---")
            c1, c2 = st.columns([3, 1])
            with c1:
                dim_name = st.text_input("Name", key=f"dim_name_{i}", placeholder="e.g. helpfulness")
                dim_desc = st.text_input("Description", key=f"dim_desc_{i}", placeholder="How helpful is the response?")
            with c2:
                dim_weight = st.number_input("Weight", min_value=0.0, max_value=1.0, value=round(1.0 / num_dims, 2), step=0.05, key=f"dim_weight_{i}")
            dim_criteria = st.text_input("Criteria (comma-separated)", key=f"dim_criteria_{i}", placeholder="addresses question, provides examples")
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
                except Exception as e:
                    st.error(f"Failed to create rubric: {e}")
