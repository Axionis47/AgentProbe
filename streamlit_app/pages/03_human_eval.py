import json

import streamlit as st
from lib.api_client import AgentProbeClient

st.set_page_config(page_title="Human Evaluation - AgentProbe", layout="wide")
st.title("Human Evaluation")

client = AgentProbeClient()

DIMENSIONS = ["helpfulness", "accuracy", "safety", "coherence", "tool_usage"]

# --- Step 1: Select eval run ---
try:
    runs_data = client.list_eval_runs(limit=100)
    runs = runs_data.get("items", [])
except Exception as e:
    st.error(f"Failed to load eval runs: {e}")
    st.stop()

if not runs:
    st.info("No eval runs found. Create one first.")
    st.stop()

run_options = {r["id"]: f"{r.get('name', r['id'][:8])} ({r['status']})" for r in runs}
selected_run_id = st.selectbox("Select Eval Run", options=list(run_options.keys()), format_func=lambda x: run_options[x])

# --- Step 2: Select conversation ---
try:
    convs_data = client.list_conversations(eval_run_id=selected_run_id, limit=100)
    convs = convs_data.get("items", [])
except Exception as e:
    st.error(f"Failed to load conversations: {e}")
    st.stop()

if not convs:
    st.info("No conversations in this run.")
    st.stop()

conv_options = {c["id"]: f"#{c.get('sequence_num', '?')} — {c['status']} ({c['turn_count']} turns)" for c in convs}
selected_conv_id = st.selectbox("Select Conversation", options=list(conv_options.keys()), format_func=lambda x: conv_options[x])

# --- Step 3: Display conversation ---
try:
    conv = client.get_conversation(selected_conv_id)
except Exception as e:
    st.error(f"Failed to load conversation: {e}")
    st.stop()

st.subheader(f"Conversation ({conv['turn_count']} turns)")

col1, col2, col3 = st.columns(3)
col1.metric("Total Tokens", conv["total_tokens"])
col2.metric("Latency (ms)", conv["total_latency_ms"])
col3.metric("Status", conv["status"])

st.divider()

for turn in conv.get("turns", []):
    role = turn.get("role", "unknown")
    content = turn.get("content", "")
    with st.chat_message("user" if role == "user" else "assistant"):
        st.write(content)
        if turn.get("tool_calls"):
            with st.expander("Tool Calls"):
                st.json(turn["tool_calls"])

# --- Step 4: Show existing evaluations ---
st.divider()
st.subheader("Existing Evaluations")
try:
    evals = client.get_conversation_evaluations(selected_conv_id)
    for ev in evals.get("items", []):
        with st.expander(f"{ev['evaluator_type']} — Score: {ev.get('overall_score', 'N/A')}"):
            if ev.get("reasoning"):
                st.write(ev["reasoning"])
            st.json(ev["scores"])
except Exception as e:
    st.warning(f"Could not load evaluations: {e}")

# --- Step 5: Human evaluation form ---
st.divider()
st.subheader("Submit Human Evaluation")

with st.form("human_eval_form"):
    scores = {}
    cols = st.columns(len(DIMENSIONS))
    for i, dim in enumerate(DIMENSIONS):
        with cols[i]:
            scores[dim] = st.slider(dim.replace("_", " ").title(), 0.0, 10.0, 5.0, 0.5, key=f"score_{dim}")

    overall = st.slider("Overall Score", 0.0, 10.0, 5.0, 0.5)
    reasoning = st.text_area("Reasoning", placeholder="Explain your evaluation...")
    evaluator_id = st.text_input("Your Name / ID", placeholder="reviewer-1")

    submitted = st.form_submit_button("Submit Evaluation")
    if submitted:
        payload = {
            "conversation_id": selected_conv_id,
            "scores": scores,
            "overall_score": overall,
            "reasoning": reasoning or None,
            "evaluator_id": evaluator_id or None,
        }
        try:
            result = client.create_human_evaluation(payload)
            st.success(f"Evaluation submitted (ID: {result['id'][:8]}...)")
        except Exception as e:
            st.error(f"Failed to submit: {e}")
