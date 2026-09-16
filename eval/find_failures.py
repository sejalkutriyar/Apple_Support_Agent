"""
Extracts REAL failure examples from the pipeline's predictions on the golden
set (produced by metrics.py) for the report's "Top 5 Failure Modes" section.

This replaces any hand-written/invented failure examples with ones that
were actually observed when the pipeline ran -- required because the
assignment explicitly asks for "real examples", not plausible-sounding
hypothetical ones.

Prerequisite: run `python eval/metrics.py` first (with the final,
hand-labeled golden_set.csv) to produce eval/pipeline_predictions_vs_golden.csv

Run: python eval/find_failures.py
"""
import pandas as pd

PREDICTIONS_PATH = "eval/pipeline_predictions_vs_golden.csv"


def main():
    df = pd.read_csv(PREDICTIONS_PATH)

    intent_failures = df[df['intent_correct'] == False]
    escalation_failures = df[df['escalate_correct'] == False]

    print("=" * 70)
    print(f"INTENT MISCLASSIFICATIONS: {len(intent_failures)} / {len(df)} "
          f"({len(intent_failures)/len(df)*100:.1f}%)")
    print("=" * 70)
    for _, row in intent_failures.head(10).iterrows():
        print(f"\nID #{row['id']}")
        print(f"  Message         : {row['customer_message']}")
        print(f"  Ground truth    : {row['ground_truth_intent']}")
        print(f"  Predicted       : {row['predicted_intent']}")

    print("\n\n" + "=" * 70)
    print(f"ESCALATION DECISION ERRORS: {len(escalation_failures)} / {len(df)} "
          f"({len(escalation_failures)/len(df)*100:.1f}%)")
    print("=" * 70)
    for _, row in escalation_failures.head(10).iterrows():
        print(f"\nID #{row['id']}")
        print(f"  Message              : {row['customer_message']}")
        print(f"  Ground truth escalate: {row['ground_truth_escalate']}")
        print(f"  Predicted escalate   : {row['predicted_escalate']}")
        print(f"  System's stated reason: {row['predicted_reason']}")

    print("\n\nCopy 5 of the most illustrative examples above (mix of intent")
    print("and escalation failures) directly into report/REPORT.md Section 3,")
    print("with your hypothesis for WHY each one happened.")


if __name__ == "__main__":
    main()