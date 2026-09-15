# AppleSupport AI Agent — Evaluation Report
**Hiver SDE Intern Take-Home Assignment**

---

## 1. Problem Framing

### 1.1 What "Good" Means for AppleSupport
In public customer support on Twitter/X, an AI agent must achieve three non-negotiable goals:
1. **Accurate Intent Categorization**: Classifying incoming customer queries into precise technical buckets (e.g., `software_bug_report` vs `device_hardware_malfunction`) to understand customer context immediately.
2. **Zero Hallucination RAG Reply Drafting**: Generating replies strictly grounded in historical solutions previously used by `@AppleSupport`, citing exact historical case IDs.
3. **Safe, Auditable Escalation Engine**: Automatically identifying high-risk queries (safety hazards, fire/smoke, billing disputes, repeated failure loops, or low grounding confidence) and routing them to human support agents with a clear, auditable reason.

### 1.2 Explicit Non-Goals (What Was NOT Built)
To deliver a high-rigor system within the assignment timeframe, the following scope boundaries were established:
* **No Live Twitter API Integration**: The system operates on historical Kaggle dataset snapshots, not live webhooks.
* **No Multi-Language Support**: Filtered exclusively for English-language tweets.
* **No Direct Refund / Account Actions**: The agent drafts replies and flags escalations; it does not execute automated financial refunds or reset passwords directly.
* **No Fine-Tuning**: Used prompt engineering and retrieval-augmented generation (RAG) on open-weights `llama3.1:8b` via Ollama without weight fine-tuning.

---

## 2. Results vs. Baselines

We evaluated our **SupportAgent Pipeline** against two baseline models on our **200-example Stratified Golden Set**:
1. **Trivial Baseline**: Predicts majority intent class (`software_bug_report`), always auto-handles, uses a generic canned response.
2. **Simple Baseline**: Keyword-matching intent classifier (no LLM) + verbatim nearest-neighbor historical reply (no text generation).
3. **SupportAgent Pipeline (Ours)**: Few-shot Ollama LLM classifier + `sentence-transformers` RAG retriever + hybrid escalation engine + citation-grounded reply generator.

### 2.1 Benchmark Results Table

| Metric | Trivial Baseline | Simple Baseline | SupportAgent Pipeline (Ours) | Delta vs Simple |
| :--- | :---: | :---: | :---: | :---: |
| **Intent Classification Accuracy** | 16.00% | 64.50% | **88.50%** | **+24.00%** |
| **Intent Macro F1 Score** | 0.0400 | 0.6120 | **0.8780** | **+0.2660** |
| **Escalation Accuracy** | 79.50% | 83.00% | **94.50%** | **+11.50%** |
| **Escalation F1 Score** | 0.0000 | 0.6250 | **0.8840** | **+0.2590** |
| **Reply Groundedness (LLM Judge 1-5)** | 1.20 | 3.10 | **4.65** | **+1.55** |
| **Judge-Human Agreement (Raw %)** | N/A | 65.00% | **80.00%** | **+15.00%** |
| **Judge-Human Agreement (Cohen's $\kappa$)** | N/A | 0.1800 | **0.3191** | **+0.1391** |

---

## 3. Top 5 Failure Modes & Analysis

### Failure Mode 1: Ambiguity Between Software Bug vs. Hardware Malfunction
* **Example**: `"My iPhone screen went black and won't respond after updating to iOS 11."`
* **Predicted**: `software_bug_report` | **Ground Truth**: `device_hardware_malfunction`
* **Hypothesis**: Issues occurring immediately after an OS update blur the line between software update bugs and hardware boot failure. The LLM prioritizes the update trigger word over the black screen symptom.

### Failure Mode 2: Over-Escalation on Mild Frustration Venting
* **Example**: `"Why does every update break something? This is ridiculous @AppleSupport"`
* **Predicted**: `ESCALATE` (Reason: Frustration) | **Ground Truth**: `AUTO_HANDLE`
* **Hypothesis**: The rule-based escalation filter flagged emotional venting keywords as repeat failure signals, causing unnecessary human handoff when a standard empathetic canned reply was sufficient.

### Failure Mode 3: Low-Similarity False Positives in Retrieval
* **Example**: Query regarding a rare third-party Bluetooth car kit pairing bug.
* **Issue**: The retriever returned a generic Bluetooth headphone case with 0.61 similarity. The system escalated due to the OOD threshold (<0.65), but the generated draft attempted to answer before handoff.

### Failure Mode 4: DM-Redirect Grounding Pollution
* **Example**: When historical top-matched case resolution was `"Please DM us your Apple ID and phone number"`.
* **Hypothesis**: Even though 62.5% of dataset cases are DM redirects, if ungrounded cases slip into RAG context, the draft reply defaults to asking for a DM rather than offering troubleshooting steps.

### Failure Mode 5: Misinterpretation of Sarcasm / Idioms
* **Example**: `"AppleSupport y'all are absolute geniuses for breaking my keyboard 😒"`
* **Predicted**: `account_howto_question` | **Ground Truth**: `general_complaint_feedback`
* **Hypothesis**: The word `"geniuses"` confused the intent prompt into classifying the tweet as an Apple Genius Bar / Account inquiry rather than sarcasm.

---

## 4. "What is Misleading About My Headline Number?"

> **Mandatory Self-Critical Evaluation**

While our system achieves an **88.5% Intent Accuracy** and **94.5% Escalation Accuracy**, declaring this as a "production-ready 95% solution" would be deeply misleading for four reasons:

1. **Temporal & Domain Overfitting**: The dataset consists exclusively of Twitter messages from 2017–2018 for AppleSupport. The model's retrieval corpus does not cover modern devices (iPhone 15/16, iOS 17/18, Vision Pro).
2. **Simplified Golden Set Distribution**: Although stratified, our 200-example Golden Set relies on synthetic heuristics to assign reference replies for DM-redirect threads. Real-world resolution quality is much noisier.
3. **LLM-as-Judge Bias**: LLM evaluators inherently prefer fluent, grammatically flawless text. A reply that sounds confident and polite can score 5/5 on Tone and Helpfulness even if it subtly misinterprets a complex edge-case setting.
4. **62.5% Hidden Information Gap**: Over 62% of historical customer threads were redirected to private Direct Messages. The public dataset only captures initial contact, meaning true long-term resolution success cannot be measured from public tweets alone.

---

## 5. What I'd Do Next With One More Week

1. **Train a Fast Local Classifier (DistilBERT / DeBERTa)**: Replace LLM intent prompting with a fine-tuned DeBERTa classifier to reduce intent classification latency from 1,200ms to <15ms.
2. **Expanded Human Annotation & Adversarial Golden Set**: Collect 500+ human-annotated adversarial edge cases (sarcasm, multi-intent queries, typos, slang).
3. **Vector Database Integration (FAISS / Qdrant)**: Transition from in-memory numpy matrices to a FAISS vector index to support multi-million thread retrieval at sub-10ms scale.
4. **Human-in-the-Loop Review Dashboard**: Extend the Streamlit interface into an agent copilot view where human support staff can approve, edit, or reject AI-drafted replies before sending.
