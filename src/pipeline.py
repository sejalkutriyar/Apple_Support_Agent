import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.intents import IntentClassifier
from src.retrieval import HistoricalRetriever
from src.reply_generator import GroundedReplyGenerator
from src.escalation import EscalationEngine

class SupportAgentPipeline:
    """End-to-End Orchestrator for AppleSupport AI Agent.
    
    Flow:
    1. Classify Intent (LLM structured JSON)
    2. Retrieve Historical Grounding Cases (SentenceTransformer Cosine Similarity)
    3. Evaluate Escalation Criteria (Rules + OOD Threshold)
    4. Draft Grounded Reply with Citations (If Auto-Handle) or Escalation Handoff Note (If Escalate)
    """
    
    def __init__(self, ollama_model: str = "llama3.1:8b", ood_threshold: float = 0.65):
        print("Initializing AppleSupport AI Agent Pipeline...")
        self.classifier = IntentClassifier(model_name=ollama_model)
        self.retriever = HistoricalRetriever()
        self.escalator = EscalationEngine(OOD_SIMILARITY_THRESHOLD=ood_threshold)
        self.reply_generator = GroundedReplyGenerator(model_name=ollama_model)
        print("Pipeline successfully initialized and ready!")

    def process_message(self, customer_message: str) -> dict:
        """Processes an incoming customer support message end-to-end with latency instrumentation."""
        start_total = time.time()
        latency = {}

        # Stage 1: Intent Classification
        t0 = time.time()
        intent_res = self.classifier.classify(customer_message)
        latency['intent_classification_ms'] = round((time.time() - t0) * 1000, 2)

        intent = intent_res['intent']
        confidence = intent_res['confidence']
        intent_reasoning = intent_res['reasoning']

        # Stage 2: Historical Case Retrieval
        t1 = time.time()
        retrieved_cases = self.retriever.search(customer_message, top_k=3)
        max_sim = self.retriever.get_max_similarity(customer_message)
        latency['retrieval_ms'] = round((time.time() - t1) * 1000, 2)

        # Stage 3: Escalation Decision Engine
        t2 = time.time()
        escalation_res = self.escalator.evaluate(
            customer_message=customer_message,
            intent=intent,
            max_similarity=max_sim,
            intent_confidence=confidence
        )
        latency['escalation_eval_ms'] = round((time.time() - t2) * 1000, 2)

        # Stage 4: Grounded Reply Generation
        t3 = time.time()
        if not escalation_res['should_escalate']:
            reply_res = self.reply_generator.generate_reply(
                customer_message=customer_message,
                intent=intent,
                retrieved_cases=retrieved_cases
            )
            draft_reply = reply_res['draft_reply']
            citations = reply_res['citations']
        else:
            # Human Escalation Notice
            draft_reply = f"[ESCALATED TO HUMAN AGENT] Reason: {escalation_res['reason']}"
            citations = []
        latency['reply_generation_ms'] = round((time.time() - t3) * 1000, 2)

        total_latency = round((time.time() - start_total) * 1000, 2)
        latency['total_pipeline_ms'] = total_latency

        return {
            "customer_message": customer_message,
            "intent": intent,
            "intent_confidence": confidence,
            "intent_reasoning": intent_reasoning,
            "max_similarity": max_sim,
            "retrieved_cases": retrieved_cases,
            "decision": escalation_res['decision'],
            "should_escalate": escalation_res['should_escalate'],
            "escalation_reason": escalation_res['reason'],
            "escalation_rule": escalation_res['rule_triggered'],
            "draft_reply": draft_reply,
            "citations": citations,
            "latency": latency
        }

if __name__ == "__main__":
    pipeline = SupportAgentPipeline()
    
    sample_queries = [
        "My phone battery is draining so fast after installing the new update!",
        "My factory charger melted and caught fire while charging my iPhone",
        "I was wrongly charged $14.99 for a subscription I cancelled"
    ]
    
    for q in sample_queries:
        print("\n==================================================")
        print(f"QUERY: {q}")
        output = pipeline.process_message(q)
        print(f"INTENT: {output['intent']} (Conf: {output['intent_confidence']})")
        print(f"DECISION: {output['decision']}")
        print(f"REASON: {output['escalation_reason']}")
        print(f"DRAFT REPLY:\n{output['draft_reply']}")
        print(f"CITATIONS: {output['citations']}")
        print(f"LATENCY: {output['latency']}")
