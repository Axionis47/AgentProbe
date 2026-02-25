import streamlit as st
from lib.api_client import AgentProbeClient

st.set_page_config(page_title="Eval Runs - AgentProbe", layout="wide")
st.title("Eval Runs")

client = AgentProbeClient()

# Filters
col1, col2 = st.columns(2)
with col1:
    status_filter = st.selectbox("Status", ["All", "pending", "running_simulation", "running_evaluation", "completed", "failed"])
with col2:
    limit = st.number_input("Results per page", min_value=5, max_value=100, value=20)

# Fetch data
try:
    params = {"limit": limit}
    if status_filter != "All":
        params["status"] = status_filter
    data = client.list_eval_runs(**params)
    
    st.metric("Total Runs", data["total"])
    
    if data["items"]:
        for run in data["items"]:
            with st.expander(f"Run: {run.get('name', run['id'][:8])} — Status: {run['status']}"):
                st.json(run)
    else:
        st.info("No eval runs found.")
except Exception as e:
    st.error(f"Failed to connect to API: {e}")
