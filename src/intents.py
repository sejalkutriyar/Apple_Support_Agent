import json
import re
import ollama

VALID_INTENTS = [
    "software_bug_report",
    "device_hardware_malfunction",
    "billing_payment_issue",
    "service_status_issue",
    "repair_warranty_request",
    "account_howto_question",
    "general_complaint_feedback"
]

TAXONOMY_DESCRIPTIONS = {
    "software_bug_report": "Issue caused by OS or app update/software — freezing, battery drain, restart loops, UI bugs, lag, keyboard glitche, app crashing.",
    "device_hardware_malfunction": "Physical device failure — broken/shattered screen, battery overheating/swelling, charger fire, speaker/microphone physical failure, device won't turn on/boot at all.",
    "billing_payment_issue": "Wrongly charged, payment method declined, refund request, subscription auto-renewal issue, unexpected App Store invoice.",
    "service_status_issue": "A specific online Apple service (iMessage, iCloud, App Store, Apple Music, FaceTime) being down, slow, or inaccessible.",
    "repair_warranty_request": "Sending device for repair, warranty coverage, AppleCare questions, store appointment/stock availability, repair status tracking.",
    "account_howto_question": "Informational 'how do I do X' query — how to change Apple ID region, transfer purchases, backup data, reset password (no bug/frustration signal).",
    "general_complaint_feedback": "General venting, brand dissatisfaction, or feature requests without a specific fixable technical issue."
}

SYSTEM_PROMPT = """You are an expert customer support intent classifier for Apple Support.
Classify the given customer message into EXACTLY ONE of the following 7 intents:

1. software_bug_report: Software/OS glitches, battery drain after update, freezing, app crashes, lag, keyboard bugs.
2. device_hardware_malfunction: Physical hardware failure, charger melting/fire, screen damaged, won't turn on at all, speaker broken.
3. billing_payment_issue: Unexpected charges, refund requests, payment declined, free trial billing disputes.
4. service_status_issue: Apple services down (iMessage, iCloud, App Store, FaceTime, Apple Music).
5. repair_warranty_request: Sending for repair, AppleCare, warranty queries, store appointments, stock availability.
6. account_howto_question: How-to questions (transfer purchases, change Apple ID region, reset password, export photos).
7. general_complaint_feedback: General dissatisfaction, venting, feature requests, hating phone with no specific fixable bug.

EXAMPLES:
Message: "iOS 11 is killing my battery after the update. Fix it please!"
Output: {"intent": "software_bug_report", "confidence": 0.95, "reasoning": "Battery drain after OS update is a classic software bug report."}

Message: "My charger caught fire while plugged in last night."
Output: {"intent": "device_hardware_malfunction", "confidence": 0.98, "reasoning": "Charger fire is a serious physical hardware safety malfunction."}

Message: "I was charged $9.99 for an app after I already cancelled the trial."
Output: {"intent": "billing_payment_issue", "confidence": 0.95, "reasoning": "Customer disputing an unexpected subscription charge."}

Message: "Is anyone else's iMessage not sending messages right now?"
Output: {"intent": "service_status_issue", "confidence": 0.90, "reasoning": "Querying whether iMessage online service is currently down."}

Message: "How do I transfer my purchased apps to my new Apple ID?"
Output: {"intent": "account_howto_question", "confidence": 0.92, "reasoning": "Informational query about transferring account purchases."}

Message: "y'all need to add a dark mode option to iOS"
Output: {"intent": "general_complaint_feedback", "confidence": 0.88, "reasoning": "Feature request / general product feedback."}

Message: "Sent my iPhone off for repair 2 weeks ago, how do I check status?"
Output: {"intent": "repair_warranty_request", "confidence": 0.93, "reasoning": "Asking for update on an existing repair service request."}

Respond ONLY with a valid JSON object with keys: "intent", "confidence" (float 0.0 to 1.0), and "reasoning". Do not include extra text or markdown formatting.
"""

class IntentClassifier:
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.model_name = model_name

    def classify(self, message: str) -> dict:
        """Classifies a customer message into an intent using Ollama LLM."""
        user_prompt = f"Customer Message: \"{message}\"\nJSON Output:"
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                options={"temperature": 0.1}
            )
            raw_text = response['message']['content'].strip()
            
            # Attempt to extract JSON from response
            parsed = self._parse_json(raw_text)
            if parsed and parsed.get("intent") in VALID_INTENTS:
                parsed["confidence"] = float(parsed.get("confidence", 0.8))
                parsed["raw_response"] = raw_text
                return parsed
            
            # Fallback if intent is slightly misformatted or missing
            intent = self._fallback_intent_extract(raw_text)
            return {
                "intent": intent,
                "confidence": 0.5,
                "reasoning": f"Parsed via fallback heuristic from response: {raw_text[:100]}",
                "raw_response": raw_text
            }

        except Exception as e:
            # Fallback keyword intent classifier when Ollama server is offline (e.g. Streamlit Cloud)
            msg_lower = message.lower()
            intent = "general_complaint_feedback"
            
            if any(w in msg_lower for w in ["battery", "ios", "update", "freeze", "lag", "bug", "crash", "keyboard"]):
                intent = "software_bug_report"
            elif any(w in msg_lower for w in ["screen", "charger", "fire", "boot", "power", "smoke", "melt", "speaker"]):
                intent = "device_hardware_malfunction"
            elif any(w in msg_lower for w in ["charge", "refund", "card", "bill", "subscription", "pay", "money", "trial"]):
                intent = "billing_payment_issue"
            elif any(w in msg_lower for w in ["imessage", "icloud", "app store", "facetime", "music", "down", "server"]):
                intent = "service_status_issue"
            elif any(w in msg_lower for w in ["repair", "warranty", "applecare", "store", "appointment", "fix", "send"]):
                intent = "repair_warranty_request"
            elif any(w in msg_lower for w in ["how do i", "how to", "transfer", "password", "apple id", "reset", "backup"]):
                intent = "account_howto_question"

            return {
                "intent": intent,
                "confidence": 0.85,
                "reasoning": f"Classified via keyword-matching fallback engine (Ollama offline/cloud mode).",
                "raw_response": ""
            }

    def _parse_json(self, text: str) -> dict:
        """Extracts JSON block from raw LLM text."""
        # Try direct json loads
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try regex search for json code block or raw object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None

    def _fallback_intent_extract(self, text: str) -> str:
        """Finds valid intent string in raw text if JSON parsing fails."""
        for intent in VALID_INTENTS:
            if intent in text:
                return intent
        return "general_complaint_feedback"

if __name__ == "__main__":
    import pandas as pd
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("Testing IntentClassifier with Ollama (llama3.1:8b)...")
    classifier = IntentClassifier()
    
    # Load 5 sample messages from data
    df = pd.read_csv("data/apple_support_sample.csv")
    samples = df['customer_message'].head(5).tolist()
    
    for i, msg in enumerate(samples, 1):
        print(f"\n--- Test Sample {i} ---")
        print(f"Message: {msg}")
        result = classifier.classify(msg)
        print(f"Result: Intent='{result['intent']}', Confidence={result['confidence']}")
        print(f"Reasoning: {result['reasoning']}")
