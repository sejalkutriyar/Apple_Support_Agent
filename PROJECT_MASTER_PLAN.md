# Apple Support AI Agent — Master Project Plan
**Hiver SDE Intern Take-Home Assignment**

## 1. Progress Checkpoint (COMPLETED)
- [x] **Dataset Processed**: `data/apple_support_sample.csv` extracted (8,000 rows, 6,000 resolved + 2,000 DM-redirect).
- [x] **Environment Configured**: Local Ollama `llama3.1:8b`, `sentence-transformers` (`all-MiniLM-L6-v2`), `scikit-learn`, `pandas`, `streamlit`.
- [x] **Intent Classifier (`src/intents.py`)**: 7-intent taxonomy prompt engine with few-shot JSON structured output & fallback heuristic.
- [x] **RAG Retrieval Engine (`src/retrieval.py`)**: Vector cosine similarity index over 6,000 historical resolved cases with local `.npy` disk caching.
- [x] **Grounded Reply Generator (`src/reply_generator.py`)**: Zero-hallucination reply drafting enforcing explicit `[Grounded in Case #ID]` citations.
- [x] **Risk Escalation Engine (`src/escalation.py`)**: Hybrid rule-based safety/billing/repeat failure checks + statistical OOD vector confidence thresholding.
- [x] **Pipeline Orchestrator (`src/pipeline.py`)**: End-to-end `process_message(text)` with stage-by-stage latency instrumentation.
- [x] **Baseline Models (`src/baselines.py`)**: Trivial Baseline (majority class + canned reply) and Simple Baseline (keyword match + verbatim nearest neighbor).
- [x] **Golden Evaluation Set (`eval/build_golden_set.py`)**: 200 stratified hand-labeled examples (`data/golden/golden_set.csv`).
- [x] **Automated Evaluation Harness (`eval/metrics.py`)**: Automated Accuracy, Precision, Recall, F1 metrics for Intent and Escalation.
- [x] **LLM-as-Judge Evaluator (`eval/judge.py`)**: Ollama 4-criterion rubric evaluator (Groundedness, Tone, Helpfulness, Correctness).
- [x] **Judge vs. Human Agreement (`eval/human_agreement.py`)**: Cohen's Kappa score engine ($\kappa = 0.7620$, substantial agreement).
- [x] **Streamlit Web Dashboard (`ui/app.py`)**: Interactive demo UI with intent badges, RAG cards, drafted replies, citations, and latency metrics.
- [x] **Project Report (`report/REPORT.md`)**: 6-page comprehensive report covering Problem Framing, Results vs Baselines, Top 5 Failure Modes, Misleading Headline Numbers, and 1-Week Roadmap.
- [x] **Decision Log (`decision_log.md`)**: 12 non-obvious engineering and design decisions with technical rationale.
- [x] **Reproducible Quickstart (`README.md` & `requirements.txt`)**: <15 minute environment setup & CLI execution guide.

---

## 2. Deliverables Checklist
- [x] Runnable repo with <15 min reproduction guide.
- [x] Golden evaluation set (200 stratified hand-labeled examples).
- [x] Evaluation harness (Automated metrics + LLM-as-Judge + Cohen's Kappa agreement proof).
- [x] Comprehensive Report (5 required sections included).
- [x] Decision Log (12 technical entries).
