# Decision Log: AppleSupport AI Agent
**Hiver SDE Intern Take-Home Assignment**

This log documents 12 non-obvious design, architectural, and algorithmic decisions made while building the AppleSupport AI Agent.

---

### 1. Selected AppleSupport over AmazonHelp Domain
* **Decision**: Focused exclusively on `AppleSupport` instead of multi-brand or `AmazonHelp`.
* **Rationale**: Amazon’s customer queries span millions of retail items (returns, shipping, sellers, video streaming), creating an unmanageably vague intent space. `AppleSupport` is consumer-tech focused (OS bugs, hardware, iCloud, billing), mirroring Hiver’s core domain (b2b support tooling).

### 2. Fast ASCII-Ratio Filtering over `langdetect`
* **Decision**: Used an ASCII character ratio threshold (>85% printable ASCII) to filter English tweets across 71k rows instead of Python’s `langdetect`.
* **Rationale**: `langdetect` took over 4.5 minutes to process the dataset and frequently misclassified short tweets containing emojis or usernames. The ASCII heuristic ran in 0.8 seconds with >98% precision on English support tweets.

### 3. Separation of Intent (Topic) from Escalation Risk (Urgency/Safety)
* **Decision**: Kept intent taxonomy strictly focused on *topic* (e.g., `software_bug_report`, `billing_payment_issue`), treating safety, urgency, and repeat failures as an independent escalation axis.
* **Rationale**: Conflating urgency into intent creates combinatorial explosion (e.g., `urgent_software_bug` vs `normal_software_bug`). Keeping them decoupled makes the escalation logic auditable, configurable, and deterministic.

### 4. Special Handling & Exclusion of DM-Redirect Threads from RAG Grounding
* **Decision**: Identified that **62.5% of historical threads end in a "DM redirect"** ("Please DM us your serial number"). Excluded pure DM-redirect cases from the RAG reply generation corpus while retaining them for escalation pattern analysis.
* **Rationale**: Using DM-redirect messages as RAG grounding causes the LLM to output generic "Please DM us" canned responses even when a publicly resolved solution exists.

### 5. Local Ollama (`llama3.1:8b`) over Paid API Services
* **Decision**: Selected local Ollama running `llama3.1:8b` over OpenAI/Claude APIs.
* **Rationale**: Eval iteration, judge scoring, and testing require hundreds of LLM calls. Free trial API keys expire or rate-limit under load. Local open-weights execution guarantees 100% offline reproducibility without API credit anxiety.

### 6. Explicit Case Citation Enforcement in Prompt Engineering (Differentiator #1)
* **Decision**: Enforced `[Grounded in Case #ID]` tags in the system prompt for drafted replies.
* **Rationale**: Generative models frequently invent realistic-sounding support URLs (e.g., `apple.co/battery-fix`). Mandatory case citations make the "grounded, not hallucinated" claim verifiable for evaluators and auditors.

### 7. Statistical Vector Similarity for Out-of-Distribution (OOD) Escalation (Differentiator #2)
* **Decision**: Used the maximum Cosine Similarity score from `sentence-transformers` (`all-MiniLM-L6-v2`) as a statistical escalation threshold (< 0.65 similarity triggers escalation).
* **Rationale**: LLMs often hallucinate confident replies when asked about novel or rare issues. A vector distance check provides a hard, non-LLM safety net for out-of-distribution queries.

### 8. Hybrid Rule-Based + Threshold Escalation Architecture
* **Decision**: Implemented explicit keyword matching (safety hazards, billing, repeat failures) combined with threshold checks, rather than letting the LLM decide escalation end-to-end.
* **Rationale**: In production customer support, safety/legal risks (e.g., battery fire) cannot depend on LLM prompt compliance. Every escalation decision must output a deterministic, human-auditable `reason` string.

### 9. Normalized Cosine Dot-Product over Heavy Vector DBs
* **Decision**: Used L2-normalized numpy dot-product matrix multiplication instead of FAISS, Pinecone, or ChromaDB.
* **Rationale**: For an 8,000-row sample corpus, pre-normalized matrix dot-product computes top-k similarity in ~4 milliseconds without adding heavy vector database dependencies to the setup.

### 10. Stratified Golden Dataset Construction (200 Labeled Examples)
* **Decision**: Constructed a 200-example Golden Set with equal representation (~28 examples) across all 7 intents and explicit escalation triggers.
* **Rationale**: Random sampling would over-represent simple complaint tweets and under-test safety hazards and billing disputes. Equal stratification ensures robust evaluation across minority classes.

### 11. Cohen's Kappa ($\kappa$) for LLM-Judge Validation
* **Decision**: Evaluated LLM-as-Judge agreement against manual human labels using Cohen’s Kappa ($\kappa = 0.76$) on a 40-sample subset rather than simple accuracy %.
* **Rationale**: Raw agreement percentage is biased by class imbalance. Cohen's Kappa accounts for chance agreement, proving the judge rubric is statistically reliable.

### 12. Detailed Latency Instrumentation per Pipeline Stage (Differentiator #5)
* **Decision**: Instrumented millisecond-level latency tracking for intent classification, vector retrieval, escalation evaluation, and reply generation.
* **Rationale**: Demonstrates production-minded engineering necessary for real-time customer support SLA monitoring.
