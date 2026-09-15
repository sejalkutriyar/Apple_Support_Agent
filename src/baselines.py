import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval import HistoricalRetriever

KEYWORD_INTENT_MAP = {
    "battery": "software_bug_report",
    "drain": "software_bug_report",
    "update": "software_bug_report",
    "ios": "software_bug_report",
    "freeze": "software_bug_report",
    "lag": "software_bug_report",
    "bug": "software_bug_report",
    "screen": "device_hardware_malfunction",
    "boot": "device_hardware_malfunction",
    "power": "device_hardware_malfunction",
    "fire": "device_hardware_malfunction",
    "broken": "device_hardware_malfunction",
    "charged": "billing_payment_issue",
    "bill": "billing_payment_issue",
    "refund": "billing_payment_issue",
    "money": "billing_payment_issue",
    "imessage": "service_status_issue",
    "icloud": "service_status_issue",
    "down": "service_status_issue",
    "repair": "repair_warranty_request",
    "warranty": "repair_warranty_request",
    "applecare": "repair_warranty_request",
    "store": "repair_warranty_request",
    "how do i": "account_howto_question",
    "how to": "account_howto_question",
    "transfer": "account_howto_question"
}

class TrivialBaseline:
    """Trivial Baseline: Predicts majority class, always auto-handles, uses canned response."""
    
    def process_message(self, customer_message: str) -> dict:
        return {
            "customer_message": customer_message,
            "intent": "software_bug_report",
            "intent_confidence": 1.0,
            "decision": "AUTO_HANDLE",
            "should_escalate": False,
            "escalation_reason": "Trivial baseline always auto-handles.",
            "draft_reply": "Hi! Thanks for reaching out to Apple Support. Please try restarting your device or updating to the latest iOS version.",
            "citations": []
        }

class SimpleBaseline:
    """Simple Baseline: Keyword-matching intent classifier + verbatim nearest-neighbor historical reply."""
    
    def __init__(self):
        self.retriever = HistoricalRetriever()

    def process_message(self, customer_message: str) -> dict:
        msg_lower = customer_message.lower()
        
        # 1. Keyword-based Intent Classification
        predicted_intent = "general_complaint_feedback"
        for kw, intent in KEYWORD_INTENT_MAP.items():
            if kw in msg_lower:
                predicted_intent = intent
                break

        # 2. Nearest Neighbor Retrieval
        retrieved = self.retriever.search(customer_message, top_k=1)
        
        # 3. Simple Escalation Logic
        should_escalate = False
        reason = "Simple baseline auto-handle."
        if any(w in msg_lower for w in ["fire", "lawsuit", "refund", "stolen", "charged"]):
            should_escalate = True
            reason = "Keyword match for high risk/billing issue."

        if not should_escalate and retrieved:
            reply = retrieved[0]['resolution']
            citations = [retrieved[0]['case_id']]
        else:
            reply = "[ESCALATED TO HUMAN AGENT]"
            citations = []

        return {
            "customer_message": customer_message,
            "intent": predicted_intent,
            "intent_confidence": 0.70,
            "decision": "ESCALATE" if should_escalate else "AUTO_HANDLE",
            "should_escalate": should_escalate,
            "escalation_reason": reason,
            "draft_reply": reply,
            "citations": citations
        }

if __name__ == "__main__":
    t_base = TrivialBaseline()
    s_base = SimpleBaseline()
    
    msg = "My iPhone battery is draining super fast after iOS 11 update"
    print("--- TRIVIAL BASELINE OUTPUT ---")
    print(t_base.process_message(msg))
    print("\n--- SIMPLE BASELINE OUTPUT ---")
    print(s_base.process_message(msg))
