import json

import streamlit as st
from lib.api_client import AgentProbeClient

st.set_page_config(page_title="Conversations - AgentProbe", layout="wide")

st.header("Conversations")

client = AgentProbeClient()

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
    agent = agent_names.get(r.get("agent_config_id", ""), "Unknown Agent")
    scenario = scenario_names.get(r.get("scenario_id", ""), "Unknown Scenario")
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

tab_labels = [f"#{c.get('sequence_num', i + 1)}" for i, c in enumerate(convs)]
tabs = st.tabs(tab_labels)

for tab, conv_summary in zip(tabs, convs):
    with tab:
        conv_id = conv_summary["id"]

        try:
            conv = client.get_conversation(conv_id)
        except Exception as e:
            st.error(f"Failed to load conversation: {e}")
            continue

        col1, col2, col3 = st.columns(3)
        col1.metric("Turns", conv["turn_count"])
        col2.metric("Tokens", f"{conv['total_tokens']:,}")
        col3.metric("Latency", f"{conv['total_latency_ms']:,} ms")

        st.markdown("")

        for turn in conv.get("turns", []):
            role = turn.get("role", "unknown")
            content = turn.get("content", "")

            if role == "user":
                with st.chat_message("user"):
                    st.write(content)

            elif role == "assistant":
                with st.chat_message("assistant"):
                    if content:
                        st.write(content)

                    if turn.get("tool_calls"):
                        for tc in turn["tool_calls"]:
                            func = tc.get("function", {})
                            tool_name = func.get("name", tc.get("name", "unknown"))
                            args_raw = func.get("arguments", "")

                            if isinstance(args_raw, str):
                                try:
                                    args_raw = json.loads(args_raw)
                                except (json.JSONDecodeError, TypeError):
                                    pass

                            st.code(f"{tool_name}({json.dumps(args_raw, indent=2) if isinstance(args_raw, dict) else args_raw})", language="json")

                    if not content and not turn.get("tool_calls"):
                        st.write("*(empty)*")

            elif role == "tool":
                tool_call_id = turn.get("tool_call_id", "")
                st.markdown(
                    f'<div style="background-color: #f0f2f6; padding: 0.75rem; '
                    f'border-radius: 0.5rem; margin: 0.5rem 0; border-left: 3px solid #6c757d;">'
                    f'<strong>Result</strong> <code>{tool_call_id}</code><br/>'
                    f'<pre style="white-space: pre-wrap; font-size: 0.85em;">{content[:2000]}</pre>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Evaluation Scores
        st.markdown("")
        st.subheader("Scores")
        try:
            evals = client.get_conversation_evaluations(conv_id)
            eval_items = evals.get("items", [])

            if not eval_items:
                st.write("No evaluations yet.")
            else:
                eval_cols = st.columns(len(eval_items))
                for col, ev in zip(eval_cols, eval_items):
                    with col:
                        score_display = (
                            f"{ev['overall_score']:.1f}/10"
                            if ev.get("overall_score") is not None
                            else "N/A"
                        )
                        st.metric(
                            ev.get("evaluator_type", "Unknown").replace("_", " ").title(),
                            score_display,
                        )

                        scores = ev.get("scores", {})
                        if scores and isinstance(scores, dict):
                            for dim, score in scores.items():
                                try:
                                    st.write(f"**{dim.replace('_', ' ').title()}:** {float(score):.1f}")
                                except (ValueError, TypeError):
                                    st.write(f"**{dim.replace('_', ' ').title()}:** {score}")

                        if ev.get("reasoning"):
                            with st.expander("Reasoning"):
                                st.write(ev["reasoning"])
        except Exception as e:
            st.warning(f"Could not load evaluations: {e}")
