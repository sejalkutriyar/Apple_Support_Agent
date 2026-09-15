import re

SAFETY_KEYWORDS = [
    "fire", "caught fire", "smoke", "exploded", "explosion", "injury", "burn", "hazard", 
    "lawsuit", "sue", "legal action", "attorney", "lawyer",
    "phishing", "stolen", "hacked", "identity theft", "unauthorized access"
]

REPEAT_FAILURE_PATTERNS = [
    r"\b(2nd|3rd|4th|5th|multiple|several)\s+(time|try|attempt)\b",
    r"still (not working|broken|failing|freezing|crashing)",
    r"already (tried|reset|restored|went to|contacted)",
    r"tried everything"
]

BILLING_KEYWORDS = [
    "refund", "unauthorized charge", "overcharged", "run me my money", 
    "stole my money", "charged twice", "wrong bill"
]

class EscalationEngine:
    """Hybrid rule-based and confidence-threshold escalation decision engine."""
    
    def __init__(self, OOD_SIMILARITY_THRESHOLD: float = 0.65):
        self.ood_threshold = OOD_SIMILARITY_THRESHOLD

    def evaluate(self, customer_message: str, intent: str, max_similarity: float, intent_confidence: float) -> dict:
        """Evaluates whether message should be escalated to human agent.
        
        Returns:
            dict with:
                - should_escalate: bool
                - decision: "ESCALATE" or "AUTO_HANDLE"
                - reason: audit trail string explaining the decision
                - rule_triggered: string identifier of rule (or "NONE")
        """
        msg_lower = customer_message.lower()

        # Rule 1: Safety / Legal / Security Risk (CRITICAL priority)
        for kw in SAFETY_KEYWORDS:
            if kw in msg_lower:
                return {
                    "should_escalate": True,
                    "decision": "ESCALATE",
                    "reason": f"Safety/Legal/Security hazard keyword detected: '{kw}'",
                    "rule_triggered": "SAFETY_HAZARD"
                }

        # Rule 2: Billing & Monetary Dispute
        if intent == "billing_payment_issue":
            return {
                "should_escalate": True,
                "decision": "ESCALATE",
                "reason": "Billing and monetary disputes require human agent financial approval.",
                "rule_triggered": "BILLING_DISPUTE"
            }
        for kw in BILLING_KEYWORDS:
            if kw in msg_lower:
                return {
                    "should_escalate": True,
                    "decision": "ESCALATE",
                    "reason": f"Billing dispute keyword detected: '{kw}'",
                    "rule_triggered": "BILLING_DISPUTE"
                }

        # Rule 3: Repeated Failure Pattern / Escalated Frustration
        for pattern in REPEAT_FAILURE_PATTERNS:
            if re.search(pattern, msg_lower):
                return {
                    "should_escalate": True,
                    "decision": "ESCALATE",
                    "reason": f"Repeated technical failure pattern detected: '{pattern}'",
                    "rule_triggered": "REPEAT_FAILURE"
                }

        # Rule 4: Out-Of-Distribution (OOD) / Low Retrieval Grounding Confidence
        if max_similarity < self.ood_threshold:
            return {
                "should_escalate": True,
                "decision": "ESCALATE",
                "reason": f"Low retrieval grounding similarity ({max_similarity:.2f} < {self.ood_threshold:.2f}). OOD / novel case.",
                "rule_triggered": "LOW_RETRIEVAL_CONFIDENCE"
            }

        # Rule 5: Low Intent Classifier Confidence
        if intent_confidence < 0.60:
            return {
                "should_escalate": True,
                "decision": "ESCALATE",
                "reason": f"Low intent classification confidence ({intent_confidence:.2f} < 0.60). Ambiguous intent.",
                "rule_triggered": "LOW_INTENT_CONFIDENCE"
            }

        # Default: AUTO_HANDLE
        return {
            "should_escalate": False,
            "decision": "AUTO_HANDLE",
            "reason": f"High intent confidence ({intent_confidence:.2f}) and grounded historical match (similarity {max_similarity:.2f}). Safe for auto-reply.",
            "rule_triggered": "NONE"
        }

if __name__ == "__main__":
    engine = EscalationEngine()
    
    # Test cases
    test_cases = [
        ("My charger caught fire last night!", "device_hardware_malfunction", 0.90, 0.95),
        ("I was charged $4.99 wrongfully", "billing_payment_issue", 0.85, 0.92),
        ("This is the 3rd time my phone froze today", "software_bug_report", 0.80, 0.90),
        ("How do I change my Apple ID country?", "account_howto_question", 0.88, 0.95),
        ("My obscure custom Bluetooth headset wont pair", "software_bug_report", 0.45, 0.80)
    ]
    
    for msg, intent, sim, conf in test_cases:
        res = engine.evaluate(msg, intent, sim, conf)
        print(f"\nMessage: '{msg}'")
        print(f"Decision: {res['decision']} | Reason: {res['reason']}")
