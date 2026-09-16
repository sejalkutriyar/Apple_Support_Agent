# AppleSupport AI Agent

> **Autonomous Customer Support Agent for @AppleSupport** 

An end-to-end AI Support Agent built for `@AppleSupport` using historical Twitter customer support threads. The system classifies customer intents, retrieves grounded historical resolutions via RAG, drafts citation-backed replies, and executes automated risk escalation logic for high-hazard queries.

---

## Quick Start & Reproduction Guide (< 15 Minutes)

### 1. Prerequisites & Environment Setup
Ensure Python 3.10+ and [Ollama](https://ollama.com/) are installed.

```bash
# Clone the repository
git clone https://github.com/sejalkutriyar/Apple_Support_Agent.git
cd Apple_Support_Agent

# Install dependencies
python -m pip install -r requirements.txt

# Pull the required local LLM model
ollama pull llama3.1:8b
```

### 2. Run End-to-End Pipeline (CLI Test)
```bash
python src/pipeline.py
```

### 3. Launch Interactive Web App (Streamlit Demo UI)
```bash
python -m streamlit run ui/app.py
```
*Navigates automatically to `http://localhost:8501` to test custom customer queries.*

### 4. Run Evaluation Suite & Metrics Benchmark
```bash
# Evaluate automated metrics vs Baselines on Golden Set (200 examples)
python eval/metrics.py

# Evaluate LLM-as-Judge vs Human Agreement (Cohen's Kappa Score)
python eval/human_agreement.py
```

---

## Technical Architecture & Key Differentiators

1. **Citation-Grounded Replies**: Every auto-handled reply includes explicit `[Grounded in Case #ID]` tags linking directly to the historical dataset precedent.
2. **Out-of-Distribution (OOD) Escalation Safety Net**: Vector Cosine Similarity scores below `0.65` trigger statistical safety escalations before hallucination can occur.
3. **Statistical Inter-Annotator Reliability**: Evaluated LLM-as-Judge against manual human annotations using **Cohen’s Kappa ($\kappa = 0.3191$, 80.00% Raw Agreement)** on empirical test runs.
4. **Interactive Streamlit Web Dashboard**: Live visual demonstration of intent classification, top-3 RAG context cards, citations, and stage-by-stage latency breakdowns.
5. **Cost & Latency Instrumentation**: Sub-millisecond tracking across pipeline stages for SLA production readiness.

---

## Repository Structure

```
Apple_Support_Agent/
├── README.md                  # Setup instructions and project overview
├── requirements.txt           # Python dependency requirements
├── decision_log.md            # 12 technical engineering decisions & rationale
├── data/
│   ├── apple_support_sample.csv  # 8,000-row extracted historical dataset
│   ├── processed/             # Cached embeddings & index metadata
│   └── golden/
│       └── golden_set.csv     # 200 stratified hand-labeled evaluation examples
├── src/
│   ├── intents.py             # Intent classifier (Llama 3.1 structured JSON)
│   ├── retrieval.py           # SentenceTransformers RAG vector search
│   ├── reply_generator.py     # Grounded reply draft generator with citations
│   ├── escalation.py          # Rule-based + OOD threshold escalation engine
│   ├── pipeline.py            # End-to-End Orchestrator + Latency tracking
│   └── baselines.py           # Trivial baseline & Simple keyword baseline
├── eval/
│   ├── build_golden_set.py    # Stratified golden set generator
│   ├── metrics.py             # Intent & Escalation Accuracy/F1 evaluator
│   ├── judge.py               # Ollama LLM-as-Judge rubric evaluator
│   └── human_agreement.py     # Cohen's Kappa Judge vs Human agreement
├── ui/
│   └── app.py                 # Streamlit interactive web dashboard UI
└── report/
    └── REPORT.md              # 6-page comprehensive project evaluation report
```

---

## Evaluation Benchmark Results

| Benchmark Metric | Trivial Baseline | Simple Baseline | SupportAgent Pipeline (Ours) |
| :--- | :---: | :---: | :---: |
| **Intent Classification Accuracy** | 14.00% | 45.00% | **88.50%** |
| **Intent Macro F1 Score** | 0.0351 | 0.4094 | **0.8780** |
| **Escalation Accuracy** | 79.50% | 81.50% | **94.50%** |
| **Escalation F1 Score** | 0.0000 | 0.1778 | **0.8840** |
| **Reply Groundedness (LLM Judge 1-5)** | 1.20 | 3.10 | **4.65** |
| **Judge-Human Agreement (Raw %)** | N/A | 65.00% | **80.00%** |
| **Judge-Human Agreement (Cohen's $\kappa$)** | N/A | 0.1800 | **0.3191** |

---

## Documentation & Submission Links

- **Evaluation Report**: [`report/REPORT.md`](report/REPORT.md)
- **Decision Log**: [`decision_log.md`](decision_log.md)
