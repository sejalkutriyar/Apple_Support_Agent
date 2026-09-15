import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from eval.judge import LLMJudge

def evaluate_judge_human_agreement(golden_path: str = "data/golden/golden_set.csv", sample_size: int = 15):
    """Computes Cohen's Kappa score between LLM-as-Judge and Human Labels on a sample subset."""
    print(f"Loading Golden Set to evaluate Judge-Human Agreement (n={sample_size})...")
    df = pd.read_csv(golden_path).head(sample_size)
    
    judge = LLMJudge()
    
    human_labels = []
    judge_labels = []
    disagreements = []

    for idx, row in enumerate(df.iloc):
        msg = str(row['customer_message'])
        ref_reply = str(row['reference_reply'])
        
        print(f"[{idx+1}/{sample_size}] Evaluating sample ID #{row['id']} with Ollama LLM-Judge...", flush=True)
        
        # Human quality label (1 = High Quality / Grounded, 0 = Needs Escalation or Poor)
        human_pass = 0 if row['ground_truth_escalate'] else 1
        human_labels.append(human_pass)
        
        # Run LLM-as-Judge
        eval_res = judge.evaluate_reply(
            customer_message=msg,
            grounding_context=ref_reply,
            draft_reply=ref_reply
        )
        
        # Binned Judge decision (Overall score >= 3.5 means Pass=1, else Fail=0)
        judge_pass = 1 if eval_res['overall_score'] >= 3.5 else 0
        judge_labels.append(judge_pass)
        
        if human_pass != judge_pass:
            disagreements.append({
                "id": row['id'],
                "customer_message": msg,
                "human_label": "Pass" if human_pass == 1 else "Escalate/Fail",
                "judge_label": "Pass" if judge_pass == 1 else "Escalate/Fail",
                "judge_justification": eval_res['justification']
            })

    kappa = cohen_kappa_score(human_labels, judge_labels)
    agreement_pct = sum(1 for h, j in zip(human_labels, judge_labels) if h == j) / len(human_labels) * 100

    print(f"\n==================================================")
    print(f"JUDGE vs HUMAN AGREEMENT RESULTS (n={sample_size})")
    print(f"==================================================")
    print(f"Raw Agreement Rate     : {agreement_pct:.2f}%")
    print(f"Cohen's Kappa (κ)      : {kappa:.4f}")
    if kappa > 0.8:
        print("Interpretation         : Almost Perfect Agreement")
    elif kappa > 0.6:
        print("Interpretation         : Substantial Agreement")
    elif kappa > 0.4:
        print("Interpretation         : Moderate Agreement")
    else:
        print("Interpretation         : Fair/Slight Agreement")
    print(f"Total Disagreements    : {len(disagreements)} / {sample_size}")
    print(f"==================================================")

    if disagreements:
        print("\nSample Disagreement Case:")
        d = disagreements[0]
        print(f"Case ID: {d['id']} | Human: {d['human_label']} | Judge: {d['judge_label']}")
        print(f"Message: {d['customer_message']}")
        print(f"Judge Rationale: {d['judge_justification']}")

    return {
        "cohen_kappa": round(kappa, 4),
        "raw_agreement_pct": round(agreement_pct, 2),
        "disagreement_count": len(disagreements)
    }

if __name__ == "__main__":
    evaluate_judge_human_agreement()
