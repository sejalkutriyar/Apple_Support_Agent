"""
Sanity-checks the completed golden_set.csv for common labeling inconsistencies,
so you can fix them in one focused pass instead of re-reading all 180 rows.

Checks performed:
1. Near-duplicate customer messages (same recurring bug/topic) that got
   DIFFERENT intent labels -- these should almost always match.
2. Rows where is_dm_redirect=True but ground_truth_escalate=False -- per our
   escalation design, a historical DM-redirect should usually be escalated.
3. reference_reply cells that look broken (just a number, or empty) --
   likely accidental menu-choice digits typed into the wrong field during
   interactive labeling.

Run: python eval/validate_golden_set.py
"""
import pandas as pd
from difflib import SequenceMatcher

GOLDEN_PATH = "data/golden/golden_set.csv"
SIMILARITY_THRESHOLD = 0.55  # rough text-similarity cutoff for "likely same topic"


def text_similarity(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def main():
    df = pd.read_csv(GOLDEN_PATH)
    print(f"Loaded {len(df)} labeled examples.\n")

    # ---- Check 1: near-duplicate messages with different intents ----
    print("=" * 70)
    print("CHECK 1: Possible near-duplicate messages with DIFFERENT intents")
    print("=" * 70)
    flagged_pairs = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            row_i, row_j = df.iloc[i], df.iloc[j]
            if row_i['ground_truth_intent'] == row_j['ground_truth_intent']:
                continue
            sim = text_similarity(row_i['customer_message'], row_j['customer_message'])
            if sim >= SIMILARITY_THRESHOLD:
                flagged_pairs.append((row_i, row_j, sim))

    if not flagged_pairs:
        print("None found.")
    for row_i, row_j, sim in flagged_pairs:
        print(f"\n  Similarity: {sim:.2f}")
        print(f"  ID {row_i['id']} [{row_i['ground_truth_intent']}]: {row_i['customer_message'][:90]}")
        print(f"  ID {row_j['id']} [{row_j['ground_truth_intent']}]: {row_j['customer_message'][:90]}")

    # ---- Check 2: DM-redirect but not marked for escalation ----
    print("\n\n" + "=" * 70)
    print("CHECK 2: is_dm_redirect=True but ground_truth_escalate=False")
    print("=" * 70)
    mismatch = df[(df['is_dm_redirect'] == True) & (df['ground_truth_escalate'] == False)]
    if len(mismatch) == 0:
        print("None found.")
    for _, row in mismatch.iterrows():
        print(f"\n  ID {row['id']}: {row['customer_message'][:90]}")
        print(f"  -> Consider changing ground_truth_escalate to True (historical case was DM-redirected)")

    # ---- Check 3: broken reference_reply cells ----
    print("\n\n" + "=" * 70)
    print("CHECK 3: Suspicious/broken reference_reply values")
    print("=" * 70)
    def looks_broken(val):
        s = str(val).strip()
        return s == "" or s == "nan" or (s.isdigit() and len(s) <= 2) or len(s) < 15
    broken = df[df['reference_reply'].apply(looks_broken)]
    if len(broken) == 0:
        print("None found.")
    for _, row in broken.iterrows():
        print(f"\n  ID {row['id']}: reference_reply = '{row['reference_reply']}'")
        print(f"  Message: {row['customer_message'][:90]}")

    # ---- Summary ----
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Near-duplicate/inconsistent-intent pairs : {len(flagged_pairs)}")
    print(f"DM-redirect/escalate mismatches           : {len(mismatch)}")
    print(f"Broken reference_reply cells              : {len(broken)}")
    print("\nOpen data/golden/golden_set.csv in Excel, fix the flagged rows above")
    print("(search by 'id' column), save, and re-run this script until it's clean.")


if __name__ == "__main__":
    main()