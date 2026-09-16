import os
import pandas as pd
import numpy as np

def build_golden_evaluation_set(sample_path: str = "data/apple_support_sample.csv", 
                                output_path: str = "data/golden/golden_set.csv",
                                target_size: int = 200):
    """Generates a stratified, high-quality golden evaluation dataset of 200 examples."""
    print(f"Reading sample dataset from {sample_path}...")
    df = pd.read_csv(sample_path)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define heuristic keywords to stratify sample across all 7 intents & edge cases
    intent_rules = [
        ("software_bug_report", ["battery", "ios", "update", "freeze", "lag", "bug", "crash", "keyboard"]),
        ("device_hardware_malfunction", ["screen", "charger", "fire", "boot", "power", "smoke", "melt", "speaker"]),
        ("billing_payment_issue", ["charge", "refund", "card", "bill", "subscription", "pay", "money", "trial"]),
        ("service_status_issue", ["imessage", "icloud", "app store", "facetime", "music", "down", "server"]),
        ("repair_warranty_request", ["repair", "warranty", "applecare", "store", "appointment", "fix", "send"]),
        ("account_howto_question", ["how do i", "how to", "transfer", "password", "apple id", "reset", "backup"]),
        ("general_complaint_feedback", ["hate", "sucks", "dark mode", "feature", "why", "wtf", "terrible"])
    ]
    
    selected_rows = []
    seen_indices = set()
    
    # 1. Stratified sampling across intents (approx 25-30 per intent)
    for intent, keywords in intent_rules:
        count = 0
        for idx, row in df.iterrows():
            if idx in seen_indices:
                continue
            msg = str(row['customer_message']).lower()
            if any(kw in msg for kw in keywords):
                seen_indices.add(idx)
                
                # Determine ground truth escalation label
                should_escalate = False
                escalation_reason = "NONE"
                
                if any(w in msg for w in ["fire", "smoke", "exploded", "lawsuit", "sue", "stolen", "hacked"]):
                    should_escalate = True
                    escalation_reason = "SAFETY_HAZARD"
                elif intent == "billing_payment_issue" or any(w in msg for w in ["charge", "refund", "bill", "money"]):
                    should_escalate = True
                    escalation_reason = "BILLING_DISPUTE"
                elif any(w in msg for w in ["3rd time", "2nd time", "still not working", "tried everything"]):
                    should_escalate = True
                    escalation_reason = "REPEAT_FAILURE"
                
                # Build reference reply
                if not should_escalate and pd.notna(row['final_resolution']) and str(row['final_resolution']).strip() != "":
                    ref_reply = str(row['final_resolution'])
                elif should_escalate:
                    ref_reply = f"[HUMAN ESCALATION REQUIRED] Flagged for {escalation_reason}."
                else:
                    ref_reply = "Please send us a Direct Message with your device details so we can assist further."

                selected_rows.append({
                    "id": len(selected_rows) + 1,
                    "root_id": row['root_id'],
                    "customer_message": row['customer_message'],
                    "ground_truth_intent": intent,
                    "ground_truth_escalate": should_escalate,
                    "escalation_reason_category": escalation_reason,
                    "is_dm_redirect": row['is_dm_redirect'],
                    "reference_reply": ref_reply
                })
                count += 1
                if count >= 28:
                    break

    # Fill remaining to reach target_size if needed
    for idx, row in df.iterrows():
        if len(selected_rows) >= target_size:
            break
        if idx not in seen_indices:
            seen_indices.add(idx)
            selected_rows.append({
                "id": len(selected_rows) + 1,
                "root_id": row['root_id'],
                "customer_message": row['customer_message'],
                "ground_truth_intent": "general_complaint_feedback",
                "ground_truth_escalate": False,
                "escalation_reason_category": "NONE",
                "is_dm_redirect": row['is_dm_redirect'],
                "reference_reply": str(row['final_resolution']) if pd.notna(row['final_resolution']) else "Please send us a DM."
            })

    golden_df = pd.DataFrame(selected_rows)
    golden_df.to_csv(output_path, index=False)
    print(f"Golden dataset created with {len(golden_df)} rows saved to {output_path}")
    print("\nIntent distribution in Golden Set:")
    print(golden_df['ground_truth_intent'].value_counts())
    print("\nEscalation distribution in Golden Set:")
    print(golden_df['ground_truth_escalate'].value_counts())

if __name__ == "__main__":
    build_golden_evaluation_set()
