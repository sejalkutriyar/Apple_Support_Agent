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
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.baselines import TrivialBaseline, SimpleBaseline
    from src.pipeline import SupportAgentPipeline

    df_golden = pd.read_csv("data/golden/golden_set.csv")
    messages = df_golden['customer_message'].tolist()

    print("Evaluating Trivial Baseline on Golden Set...")
    tb = TrivialBaseline()
    tb_preds = [tb.process_message(m) for m in messages]
    tb_results = evaluate_predictions("data/golden/golden_set.csv", tb_preds, model_name="Trivial Baseline")

    print("\nEvaluating Simple Baseline on Golden Set...")
    sb = SimpleBaseline()
    sb_preds = [sb.process_message(m) for m in messages]
    sb_results = evaluate_predictions("data/golden/golden_set.csv", sb_preds, model_name="Simple Baseline")

    print("\nEvaluating Full SupportAgent Pipeline on Golden Set (this will take a while -- LLM calls per example)...")
    pipeline = SupportAgentPipeline()
    pipe_preds = []
    for i, m in enumerate(messages, 1):
        print(f"  [{i}/{len(messages)}] Processing...", flush=True)
        pipe_preds.append(pipeline.process_message(m))
    pipe_results = evaluate_predictions("data/golden/golden_set.csv", pipe_preds, model_name="SupportAgent Pipeline (Ours)")

    # Save all three results + raw predictions so the report can cite REAL numbers
    # and find_failures.py can identify real disagreement examples.
    import json
    summary = {"trivial": tb_results, "simple": sb_results, "pipeline": pipe_results}
    with open("eval/latest_benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved consolidated benchmark summary -> eval/latest_benchmark_summary.json")

    # Save full pipeline predictions alongside ground truth for failure analysis
    out_rows = []
    for row, pred in zip(df_golden.to_dict('records'), pipe_preds):
        out_rows.append({
            "id": row['id'],
            "customer_message": row['customer_message'],
            "ground_truth_intent": row['ground_truth_intent'],
            "predicted_intent": pred['intent'],
            "intent_correct": row['ground_truth_intent'] == pred['intent'],
            "ground_truth_escalate": row['ground_truth_escalate'],
            "predicted_escalate": pred['should_escalate'],
            "escalate_correct": bool(row['ground_truth_escalate']) == pred['should_escalate'],
            "predicted_reason": pred['escalation_reason'],
            "draft_reply": pred['draft_reply'],
        })
    pd.DataFrame(out_rows).to_csv("eval/pipeline_predictions_vs_golden.csv", index=False)
    print("Saved per-example predictions -> eval/pipeline_predictions_vs_golden.csv")
    print("(Use this file with find_failures.py to pull REAL failure examples for the report.)")