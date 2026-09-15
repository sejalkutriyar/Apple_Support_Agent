# pyrefly: ignore [missing-import]
import streamlit as st
import time
import os
import sys

# Ensure src module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import SupportAgentPipeline

# Page Config
st.set_page_config(
    page_title="Apple Support AI Agent | Enterprise Demo",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Pipeline
@st.cache_resource
def load_pipeline():
    return SupportAgentPipeline()

with st.spinner("⚡ Initializing AppleSupport AI Agent Pipeline & RAG Vector Index..."):
    pipeline = load_pipeline()

# Header Section
st.title(" Apple Support AI Agent")
st.caption("Autonomous Support Automation Platform — Powered by Few-Shot Llama 3.1 & Grounded RAG")

# Sidebar Configuration & Controls
with st.sidebar:
    st.header("⚙️ Agent Controls")
    
    sample_choice = st.selectbox(
        "Choose a Preset Customer Scenario:",
        [
            "Custom Input",
            "⚡ Battery Drain After Update (Auto-Handle)",
            "🔥 Charger Fire & Safety Hazard (Escalate)",
            "💳 Subscription Billing Dispute (Escalate)",
            "📡 iMessage Outage (Service Status)",
            "📲 Transfer Apple ID Purchases (Account How-To)"
        ]
    )
    
    st.divider()
    st.markdown("### 🛠️ System Specifications")
    st.markdown("""
    - **LLM Engine**: Local Ollama (`llama3.1:8b`)
    - **Vector Search**: `sentence-transformers` (`all-MiniLM-L6-v2`)
    - **Corpus Size**: 8,000 Historical Resolved Threads
    - **Escalation Threshold**: Cosine Similarity < `0.65` (OOD)
    """)

# Preset Text Pre-fill
if "Battery Drain" in sample_choice:
    default_text = "iOS 11 update is killing my iPhone battery so fast! What is going on? Please fix this!"
elif "Charger Fire" in sample_choice:
    default_text = "My official Apple charger started smoking and caught fire on my nightstand last night!"
elif "Billing Dispute" in sample_choice:
    default_text = "I was just billed $14.99 for an app subscription even though I cancelled the free trial 3 days ago!"
elif "iMessage Outage" in sample_choice:
    default_text = "Is anyone else's iMessage not delivering photos or texts right now?"
elif "Transfer Apple ID" in sample_choice:
    default_text = "How do I transfer my purchased apps and music from my old Apple ID to my new device?"
else:
    default_text = ""

# Main Input Form
with st.container():
    st.subheader("💬 Customer Input")
    user_input = st.text_area(
        "Incoming Customer Support Tweet / Query:",
        value=default_text,
        height=100,
        placeholder="Type or paste a customer query here..."
    )
    
    submit = st.button("🚀 Run AI Support Agent", type="primary", use_container_width=True)

# Main Processing & Display
if submit:
    if not user_input.strip():
        st.warning("⚠️ Please enter a customer message to process.")
    else:
        with st.status("🧠 Processing Customer Message...", expanded=True) as status:
            st.write("1️⃣ Running Llama 3.1 Few-Shot Intent Classification...")
            time.sleep(0.1)
            st.write("2️⃣ Querying SentenceTransformer Vector Index for Grounding Context...")
            time.sleep(0.1)
            st.write("3️⃣ Evaluating Hybrid Escalation & Risk Rules...")
            
            res = pipeline.process_message(user_input)
            status.update(label="✅ Processing Complete!", state="complete", expanded=False)

        st.divider()

        # Tabs Layout
        tab1, tab2, tab3 = st.tabs([
            "🎯 Agent Decision & Reply", 
            "📚 Historical Grounding Cases (RAG)", 
            "⚡ Latency & SLA Performance"
        ])

        # TAB 1: Decision & Reply
        with tab1:
            col1, col2 = st.columns([1, 1], gap="large")

            with col1:
                st.markdown("#### 1️⃣ Classification & Risk Decision")
                
                # Intent Box
                st.info(f"**Predicted Intent:** `{res['intent']}`\n\n"
                        f"**Confidence:** `{res['intent_confidence']*100:.1f}%`\n\n"
                        f"**LLM Reasoning:** {res['intent_reasoning']}")

                # Escalation Box
                if res['should_escalate']:
                    st.error(f"🚨 **DECISION: ESCALATE TO HUMAN AGENT**\n\n"
                             f"**Triggered Rule:** `{res['escalation_rule']}`\n\n"
                             f"**Audit Reason:** {res['escalation_reason']}")
                else:
                    st.success(f"✅ **DECISION: AUTO-HANDLE (SAFE TO REPLY)**\n\n"
                               f"**Audit Reason:** {res['escalation_reason']}")

            with col2:
                st.markdown("#### 2️⃣ Drafted Response & Citations")
                
                if res['should_escalate']:
                    st.warning("⚠️ Message flagged for human agent review. Draft reply shows handoff note below:")
                
                st.code(res['draft_reply'], language="text")
                
                if res['citations']:
                    st.success(f"📌 **Grounded Citations:** {', '.join([f'Case #{c}' for c in res['citations']])}")
                else:
                    st.caption("No historical citations attached (Escalated or safe fallback).")

        # TAB 2: Historical RAG Cases
        with tab2:
            st.markdown(f"#### Top Retrieved Historical Precedents (Max Similarity: `{res['max_similarity']:.4f}`)")
            st.caption("The reply generator uses these exact historically resolved AppleSupport threads for zero-hallucination grounding:")
            
            for i, case in enumerate(res['retrieved_cases'], 1):
                with st.expander(f"📌 Case #{case['case_id']} — Cosine Similarity: {case['similarity_score']:.4f}", expanded=(i==1)):
                    st.markdown(f"**Customer Asked:**\n> {case['customer_message']}")
                    st.markdown(f"**AppleSupport Resolution:**\n`{case['resolution']}`")

        # TAB 3: Latency Performance
        with tab3:
            st.markdown("#### Millisecond Latency Breakdown per Pipeline Stage")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🎯 Intent Latency", f"{res['latency']['intent_classification_ms']} ms")
            m2.metric("🔎 RAG Retrieval Latency", f"{res['latency']['retrieval_ms']} ms")
            m3.metric("🚨 Escalation Rule Latency", f"{res['latency']['escalation_eval_ms']} ms")
            m4.metric("⚡ Total Pipeline Latency", f"{res['latency']['total_pipeline_ms']} ms")
