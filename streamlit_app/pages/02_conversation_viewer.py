import json

import streamlit as st
from lib.api_client import AgentProbeClient

st.set_page_config(page_title="Conversation Viewer - AgentProbe", layout="wide")
st.title("Conversation Viewer")

client = AgentProbeClient()

# --- Selection Mode ---
mode = st.radio("Select conversation by:", ["Browse from Eval Runs", "Enter ID directly"], horizontal=True)

conv_id: str | None = None

if mode == "Enter ID directly":
    conv_id = st.text_input("Conversation ID") or None
else:
    # Browse mode: select eval run, then conversation
    try:
        runs_data = client.list_eval_runs(limit=100)
        runs = runs_data.get("items", [])
    except Exception as e:
        st.error(f"Failed to load eval runs: {e}")
        runs = []

    if not runs:
        st.info("No eval runs found. Create one in the Eval Runs page first.")
        st.stop()

    run_options = {r["id"]: f"{r.get('name', r['id'][:8])} ({r['status']})" for r in runs}
    selected_run_id = st.selectbox(
        "Select Eval Run",
        options=list(run_options.keys()),
        format_func=lambda x: run_options[x],
    )

    try:
        convs_data = client.list_conversations(eval_run_id=selected_run_id, limit=100)
        convs = convs_data.get("items", [])
    except Exception as e:
        st.error(f"Failed to load conversations: {e}")
        convs = []

    if not convs:
        st.info("No conversations in this eval run.")
        st.stop()

    conv_options = {
        c["id"]: f"#{c.get('sequence_num', '?')} -- {c['status']} ({c['turn_count']} turns, {c['total_tokens']} tokens)"
        for c in convs
    }
    conv_id = st.selectbox(
        "Select Conversation",
        options=list(conv_options.keys()),
        format_func=lambda x: conv_options[x],
    )

# --- Display Conversation ---
if conv_id:
    try:
        conv = client.get_conversation(conv_id)

        st.subheader(f"Conversation ({conv['turn_count']} turns)")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tokens", f"{conv['total_tokens']:,}")
        col2.metric("Input Tokens", f"{conv.get('total_input_tokens', 0):,}")
        col3.metric("Output Tokens", f"{conv.get('total_output_tokens', 0):,}")
        col4.metric("Latency", f"{conv['total_latency_ms']:,} ms")

        st.caption(f"Status: **{conv['status']}** | ID: `{conv['id']}`")

        st.divider()

        # Render turns
        for i, turn in enumerate(conv.get("turns", [])):
            role = turn.get("role", "unknown")
            content = turn.get("content", "")

            # Map roles to chat message types
            if role == "user":
                msg_role = "user"
            elif role == "tool":
                msg_role = "assistant"
            else:
                msg_role = "assistant"

            with st.chat_message(msg_role):
                # Show role label for tool responses
                if role == "tool":
                    st.caption(f"Tool Response ({turn.get('tool_call_id', 'unknown')})")

                if content:
                    st.write(content)
                elif role == "assistant" and not turn.get("tool_calls"):
                    st.write("*(empty response)*")

                # Tool calls in expandable sections
                if turn.get("tool_calls"):
                    for tc in turn["tool_calls"]:
                        tool_name = tc.get("function", {}).get("name", tc.get("name", "unknown"))
                        with st.expander(f"Tool Call: {tool_name}"):
                            if isinstance(tc.get("function", {}).get("arguments"), str):
                                try:
                                    args = json.loads(tc["function"]["arguments"])
                                    st.json(args)
                                except (json.JSONDecodeError, TypeError):
                                    st.code(tc["function"]["arguments"])
                            else:
                                st.json(tc)

                # Show per-turn metadata if present
                if turn.get("tokens"):
                    st.caption(f"Tokens: {turn['tokens']}")
                if turn.get("latency_ms"):
                    st.caption(f"Latency: {turn['latency_ms']} ms")

        # --- Show evaluations ---
        st.divider()
        st.subheader("Evaluations")
        try:
            evals = client.get_conversation_evaluations(conv_id)
            eval_items = evals.get("items", [])

            if not eval_items:
                st.info("No evaluations yet for this conversation.")
            else:
                for ev in eval_items:
                    score_display = f"{ev['overall_score']:.1f}/10" if ev.get("overall_score") is not None else "N/A"
                    with st.expander(f"{ev['evaluator_type']} -- Overall: {score_display}"):
                        if ev.get("reasoning"):
                            st.write("**Reasoning:**")
                            st.write(ev["reasoning"])

                        # Per-dimension breakdown
                        scores = ev.get("scores", {})
                        if scores and isinstance(scores, dict):
                            st.write("**Per-Dimension Scores:**")
                            score_cols = st.columns(min(len(scores), 5))
                            for idx, (dim, score) in enumerate(scores.items()):
                                with score_cols[idx % len(score_cols)]:
                                    try:
                                        st.metric(dim.replace("_", " ").title(), f"{float(score):.1f}")
                                    except (ValueError, TypeError):
                                        st.metric(dim.replace("_", " ").title(), str(score))

                        if ev.get("evaluator_id"):
                            st.caption(f"Evaluator: {ev['evaluator_id']}")
                        st.caption(f"Created: {ev.get('created_at', 'N/A')}")
        except Exception as e:
            st.warning(f"Could not load evaluations: {e}")

        # --- Show metrics ---
        st.divider()
        st.subheader("Automated Metrics")
        try:
            metrics = client.get_conversation_metrics(conv_id)
            metric_items = metrics.get("items", [])
            if not metric_items:
                st.info("No automated metrics for this conversation.")
            else:
                metric_cols = st.columns(min(len(metric_items), 4))
                for idx, m in enumerate(metric_items):
                    with metric_cols[idx % len(metric_cols)]:
                        unit = f" {m['unit']}" if m.get("unit") else ""
                        st.metric(m["metric_name"].replace("_", " ").title(), f"{m['value']:.2f}{unit}")
        except Exception as e:
            st.warning(f"Could not load metrics: {e}")

else:
    st.info("Select or enter a conversation to view it.")
