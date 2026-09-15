import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def evaluate_predictions(golden_path: str, predictions: list[dict], model_name: str = "Pipeline"):
    """Computes automated classification & escalation performance metrics against Golden Set."""
    df_golden = pd.read_csv(golden_path)
    
    y_intent_true = df_golden['ground_truth_intent'].tolist()
    y_escalate_true = df_golden['ground_truth_escalate'].tolist()
    
    y_intent_pred = [p['intent'] for p in predictions]
    y_escalate_pred = [p['should_escalate'] for p in predictions]
    
    # 1. Intent Metrics
    intent_acc = accuracy_score(y_intent_true, y_intent_pred)
    p_intent, r_intent, f1_intent, _ = precision_recall_fscore_support(
        y_intent_true, y_intent_pred, average='macro', zero_division=0
    )
    
    # 2. Escalation Metrics
    esc_acc = accuracy_score(y_escalate_true, y_escalate_pred)
    p_esc, r_esc, f1_esc, _ = precision_recall_fscore_support(
        y_escalate_true, y_escalate_pred, average='binary', zero_division=0
    )

    print(f"\n==================================================")
    print(f"EVALUATION METRICS FOR: {model_name}")
    print(f"==================================================")
    print(f"Intent Classification Accuracy : {intent_acc * 100:.2f}%")
    print(f"Intent Macro Precision         : {p_intent * 100:.2f}%")
    print(f"Intent Macro Recall            : {r_intent * 100:.2f}%")
    print(f"Intent Macro F1 Score          : {f1_intent * 100:.2f}%")
    print(f"--------------------------------------------------")
    print(f"Escalation Decision Accuracy   : {esc_acc * 100:.2f}%")
    print(f"Escalation Precision           : {p_esc * 100:.2f}%")
    print(f"Escalation Recall              : {r_esc * 100:.2f}%")
    print(f"Escalation F1 Score            : {f1_esc * 100:.2f}%")
    print(f"==================================================")

    return {
        "model_name": model_name,
        "intent_accuracy": round(intent_acc, 4),
        "intent_f1_macro": round(f1_intent, 4),
        "escalation_accuracy": round(esc_acc, 4),
        "escalation_f1": round(f1_esc, 4)
    }

if __name__ == "__main__":
    from src.baselines import TrivialBaseline, SimpleBaseline
    
    df_golden = pd.read_csv("data/golden/golden_set.csv")
    messages = df_golden['customer_message'].tolist()
    
    print("Evaluating Trivial Baseline on Golden Set...")
    tb = TrivialBaseline()
    tb_preds = [tb.process_message(m) for m in messages]
    evaluate_predictions("data/golden/golden_set.csv", tb_preds, model_name="Trivial Baseline")
    
    print("\nEvaluating Simple Baseline on Golden Set...")
    sb = SimpleBaseline()
    sb_preds = [sb.process_message(m) for m in messages]
    evaluate_predictions("data/golden/golden_set.csv", sb_preds, model_name="Simple Baseline")
