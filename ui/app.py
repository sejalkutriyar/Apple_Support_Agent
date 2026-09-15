import streamlit as st
import time
import os
import sys

# Ensure src module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import SupportAgentPipeline

st.set_page_config(
    page_title="Apple Support AI Agent | Hiver Demo",
    page_icon="",
    layout="wide"
)

# Custom Styling for Apple Dark Aesthetic
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stApp { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    .stCard {
        background-color: #1E222D;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }
    .badge-auto {
        background-color: #10B981;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
    .badge-escalate {
        background-color: #EF4444;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title(" Apple Support AI Agent")
st.caption("Hiver SDE Intern Take-Home Assignment | Grounded RAG + Intent Classification + Risk Escalation Engine")

@st.cache_resource
def load_pipeline():
    return SupportAgentPipeline()

with st.spinner("Initializing AppleSupport Agent Pipeline & Vector Index..."):
    pipeline = load_pipeline()

st.sidebar.header("⚙️ Agent Controls & Info")
sample_choice = st.sidebar.selectbox(
    "Choose a preset test sample:",
    [
        "Custom Input",
        "iOS 11 battery drain after update",
        "Charger caught fire while charging",
        "Charged $14.99 wrongfully for app trial",
        "Is iMessage down right now?",
        "How to transfer Apple ID purchases to new phone?"
    ]
)

if sample_choice == "iOS 11 battery drain after update":
    default_text = "iOS 11 update is killing my iPhone battery so fast! What is going on? Please fix this!"
elif sample_choice == "Charger caught fire while charging":
    default_text = "My official Apple charger started smoking and caught fire on my nightstand last night!"
elif sample_choice == "Charged $14.99 wrongfully for app trial":
    default_text = "I was just billed $14.99 for an app subscription even though I cancelled the free trial 3 days ago!"
elif sample_choice == "Is iMessage down right now?":
    default_text = "Is anyone else's iMessage not delivering photos or texts right now?"
elif sample_choice == "How to transfer Apple ID purchases to new phone?":
    default_text = "How do I transfer my purchased apps and music from my old Apple ID to my new device?"
else:
    default_text = ""

user_input = st.text_area("💬 Customer Tweet / Message:", value=default_text, height=100, placeholder="Type an incoming customer support message here...")

if st.button("🚀 Process Customer Message", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("Please enter a customer message to process.")
    else:
        with st.spinner("Analyzing message, retrieving historical cases, and evaluating escalation rules..."):
            res = pipeline.process_message(user_input)
            
        st.divider()
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("1. Intent & Escalation Status")
            
            # Intent Box
            st.markdown(f"**Predicted Intent:** `{res['intent']}`")
            st.markdown(f"**Intent Confidence:** `{res['intent_confidence']*100:.1f}%`")
            st.info(f"**LLM Reasoning:** {res['intent_reasoning']}")
            
            # Escalation Box
            if res['should_escalate']:
                st.error(f"🚨 **Decision:** ESCALATE TO HUMAN AGENT")
                st.markdown(f"**Triggered Rule:** `{res['escalation_rule']}`")
                st.markdown(f"**Audit Reason:** {res['escalation_reason']}")
            else:
                st.success(f"✅ **Decision:** AUTO-HANDLE")
                st.markdown(f"**Audit Reason:** {res['escalation_reason']}")

        with col2:
            st.subheader("2. Drafted Grounded Reply")
            st.code(res['draft_reply'], language="text")
            
            if res['citations']:
                st.markdown(f"📌 **Citations:** {', '.join([f'Case #{c}' for c in res['citations']])}")
            else:
                st.caption("No historical citations attached (Escalated or safe fallback).")

        st.divider()
        st.subheader("3. Historical Grounding Precedents (RAG Retrieval)")
        st.markdown(f"**Top Retrieved Matching Cases (Max Similarity Score: `{res['max_similarity']:.4f}`)**")

        for i, case in enumerate(res['retrieved_cases'], 1):
            with st.expander(f"Case #{case['case_id']} | Cosine Similarity: {case['similarity_score']:.4f}"):
                st.markdown(f"**Customer Asked:** {case['customer_message']}")
                st.markdown(f"**AppleSupport Resolution:** {case['resolution']}")

        st.divider()
        st.subheader("4. Latency & Token Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Intent Latency", f"{res['latency']['intent_classification_ms']} ms")
        m2.metric("Retrieval Latency", f"{res['latency']['retrieval_ms']} ms")
        m3.metric("Escalation Latency", f"{res['latency']['escalation_eval_ms']} ms")
        m4.metric("Total Latency", f"{res['latency']['total_pipeline_ms']} ms")
