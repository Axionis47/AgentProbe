import streamlit as st
from lib.api_client import AgentProbeClient

st.set_page_config(page_title="Rate - AgentProbe", layout="wide")

st.header("Rate Conversations")

client = AgentProbeClient()

DIMENSIONS = ["helpfulness", "accuracy", "safety", "coherence", "tool_usage"]

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

try:
    runs_data = client.list_eval_runs(limit=100)
    runs = runs_data.get("items", [])
except Exception as e:
    st.error(f"Failed to load eval runs: {e}")
    st.stop()

if not runs:
    st.info("No eval runs found.")
    st.stop()

run_options = {}
for r in runs:
    agent = agent_names.get(r.get("agent_config_id", ""), "Unknown")
    scenario = scenario_names.get(r.get("scenario_id", ""), "Unknown")
    run_options[r["id"]] = f"{agent} / {scenario}"

selected_run_id = st.selectbox(
    "Eval Run",
    options=list(run_options.keys()),
    format_func=lambda x: run_options[x],
)

try:
    convs_data = client.list_conversations(eval_run_id=selected_run_id, limit=100)
    convs = convs_data.get("items", [])
except Exception as e:
    st.error(f"Failed to load conversations: {e}")
    st.stop()

if not convs:
    st.info("No conversations in this run.")
    st.stop()

conv_options = {
    c["id"]: f"#{c.get('sequence_num', '?')} -- {c['turn_count']} turns"
    for c in convs
}
selected_conv_id = st.selectbox(
    "Conversation",
    options=list(conv_options.keys()),
    format_func=lambda x: conv_options[x],
)

try:
    conv = client.get_conversation(selected_conv_id)
except Exception as e:
    st.error(f"Failed to load conversation: {e}")
    st.stop()

st.markdown("")

for turn in conv.get("turns", []):
    role = turn.get("role", "unknown")
    content = turn.get("content", "")
    with st.chat_message("user" if role == "user" else "assistant"):
        if content:
            st.write(content)
        if turn.get("tool_calls"):
            st.json(turn["tool_calls"])

st.markdown("")

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Automated Scores")
    try:
        evals = client.get_conversation_evaluations(selected_conv_id)
        eval_items = evals.get("items", [])
        if not eval_items:
            st.write("No evaluations yet.")
        else:
            for ev in eval_items:
                evaluator = ev.get("evaluator_type", "Unknown").replace("_", " ").title()
                score = ev.get("overall_score")
                score_str = f"{score:.1f}/10" if score is not None else "N/A"
                st.metric(evaluator, score_str)
    except Exception as e:
        st.warning(f"Could not load evaluations: {e}")

with right_col:
    st.subheader("Your Rating")
    with st.form("human_eval_form"):
        scores = {}
        for dim in DIMENSIONS:
            scores[dim] = st.slider(
                dim.replace("_", " ").title(),
                0.0, 10.0, 5.0, 0.5,
                key=f"score_{dim}",
            )

        overall = st.slider("Overall", 0.0, 10.0, 5.0, 0.5)
        reasoning = st.text_area("Notes", placeholder="Optional")
        evaluator_id = st.text_input("Your ID", placeholder="reviewer-1")

        submitted = st.form_submit_button("Submit")
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
                st.success(f"Submitted (ID: {result['id'][:8]}...)")
            except Exception as e:
                st.error(f"Failed to submit: {e}")
