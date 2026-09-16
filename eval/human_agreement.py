"""
Computes agreement between the LLM-as-Judge and a human rater on the SAME
set of ACTUAL system-generated replies (see generate_review_sample.py for
how these were produced).

Prerequisite: run generate_review_sample.py first, then manually fill in
the human_* columns in eval/human_blind_sheet.csv (1-5 scale each), THEN
run this script.

We report:
  - Raw agreement % and Cohen's Kappa on a binarized "pass/fail" decision
    (overall score >= 3.5) -- easy to interpret.
  - Weighted Cohen's Kappa on the raw 1-5 per-criterion scores -- gives
    credit for near-misses (e.g. human=4, judge=3), which unweighted
    kappa unfairly penalizes as a full disagreement.
  - Per-criterion breakdown (groundedness, tone, helpfulness, correctness)
    so we can see WHERE judge and human diverge most, not just an
    aggregate number.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

BLIND_SHEET_PATH = "eval/human_blind_sheet.csv"
JUDGE_CACHE_PATH = "eval/judge_scores_cache.csv"

CRITERIA = ["groundedness", "tone_match", "helpfulness", "correctness"]


def main():
    human_df = pd.read_csv(BLIND_SHEET_PATH)
    judge_df = pd.read_csv(JUDGE_CACHE_PATH)

    missing = human_df[[f"human_{c}" for c in CRITERIA]].isnull().any(axis=1)
    if missing.any():
        n_missing = missing.sum()
        raise ValueError(
            f"{n_missing} rows in {BLIND_SHEET_PATH} still have empty human_* "
            f"columns. Fill in all rows (1-5 scale) before running this script."
        )

    merged = human_df.merge(judge_df, on="id", suffixes=("", "_judgefile"))
    print(f"Loaded {len(merged)} matched examples for agreement analysis.\n")

    print("=" * 60)
    print("PER-CRITERION AGREEMENT (1-5 scale)")
    print("=" * 60)
    for c in CRITERIA:
        h = merged[f"human_{c}"].astype(int)
        j = merged[f"judge_{c}"].astype(int)
        exact_match_pct = (h == j).mean() * 100
        within_1_pct = (abs(h - j) <= 1).mean() * 100
        kappa_w = cohen_kappa_score(h, j, weights="linear")
        print(f"{c:15s} | Exact match: {exact_match_pct:5.1f}% | "
              f"Within-1: {within_1_pct:5.1f}% | Weighted Kappa: {kappa_w:.3f}")

    human_overall = merged[[f"human_{c}" for c in CRITERIA]].astype(int).mean(axis=1)
    judge_overall = merged["judge_overall_score"].astype(float)

    human_pass = (human_overall >= 3.5).astype(int)
    judge_pass = (judge_overall >= 3.5).astype(int)

    raw_agreement = (human_pass == judge_pass).mean() * 100
    kappa = cohen_kappa_score(human_pass, judge_pass)

    print("\n" + "=" * 60)
    print("OVERALL PASS/FAIL AGREEMENT (threshold >= 3.5)")
    print("=" * 60)
    print(f"Raw Agreement Rate : {raw_agreement:.2f}%")
    print(f"Cohen's Kappa (kappa) : {kappa:.4f}")
    if kappa > 0.8:
        interp = "Almost Perfect Agreement"
    elif kappa > 0.6:
        interp = "Substantial Agreement"
    elif kappa > 0.4:
        interp = "Moderate Agreement"
    elif kappa > 0.2:
        interp = "Fair Agreement"
    else:
        interp = "Slight/Poor Agreement"
    print(f"Interpretation     : {interp}")

    disagreements = merged[human_pass != judge_pass]
    print(f"\nTotal disagreements: {len(disagreements)} / {len(merged)}")
    if len(disagreements) > 0:
        print("\nSample disagreement cases (for report's failure analysis):")
        for _, row in disagreements.head(3).iterrows():
            print(f"\n  ID #{row['id']}")
            print(f"  Message : {row['customer_message'][:100]}")
            print(f"  Reply   : {row['generated_draft_reply'][:150]}")
            print(f"  Human overall (avg of 4 criteria): {human_overall.loc[row.name]:.2f}")
            print(f"  Judge overall_score              : {row['judge_overall_score']}")
            print(f"  Judge justification              : {row['judge_justification']}")

    return {
        "raw_agreement_pct": round(raw_agreement, 2),
        "cohen_kappa": round(kappa, 4),
        "n_samples": len(merged),
        "n_disagreements": len(disagreements),
    }


if __name__ == "__main__":
    main()