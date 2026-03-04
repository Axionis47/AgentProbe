import streamlit as st
from lib.api_client import AgentProbeClient

st.set_page_config(page_title="Conversation Viewer - AgentProbe", layout="wide")
st.title("Conversation Viewer")

client = AgentProbeClient()

conv_id = st.text_input("Conversation ID")

if conv_id:
    try:
        conv = client.get_conversation(conv_id)
        
        st.subheader(f"Conversation ({conv['turn_count']} turns)")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tokens", conv["total_tokens"])
        col2.metric("Latency (ms)", conv["total_latency_ms"])
        col3.metric("Status", conv["status"])
        
        st.divider()
        
        for turn in conv["turns"]:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            
            with st.chat_message("user" if role == "user" else "assistant"):
                st.write(content)
                
                if turn.get("tool_calls"):
                    with st.expander("Tool Calls"):
                        st.json(turn["tool_calls"])
        
        # Show evaluations
        st.divider()
        st.subheader("Evaluations")
        evals = client.get_conversation_evaluations(conv_id)
        for ev in evals.get("items", []):
            with st.expander(f"{ev['evaluator_type']} — Score: {ev.get('overall_score', 'N/A')}"):
                if ev.get("reasoning"):
                    st.write(ev["reasoning"])
                st.json(ev["scores"])
    except Exception as e:
        st.error(f"Error loading conversation: {e}")
else:
    st.info("Enter a conversation ID to view it.")
