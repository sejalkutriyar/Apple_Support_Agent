"""
Step 1 of the golden set pipeline: SAMPLING (not labeling).

Selects a diverse spread of ~180 candidate messages from the full dataset
using keyword-based buckets. IMPORTANT: these keywords are used ONLY to
ensure the human labeler sees a good mix of topics -- they do NOT assign
the ground-truth label. The actual intent/escalation labels are decided
by a human (you) in the next step (label_interactively.py).

Output: data/golden/golden_candidates.csv (unlabeled, ready for labeling)
"""
import os
import pandas as pd

SAMPLE_PATH = "data/apple_support_sample.csv"
OUTPUT_PATH = "data/golden/golden_candidates.csv"
TARGET_SIZE = 180

# Used only to spread sampling across rough topic buckets so the labeler
# doesn't see 180 examples that are all "battery drain" tweets.
SAMPLING_BUCKETS = [
    ["battery", "ios", "update", "freeze", "lag", "bug", "crash", "keyboard"],
    ["screen", "charger", "fire", "boot", "power", "smoke", "melt", "speaker"],
    ["charge", "refund", "card", "bill", "subscription", "pay", "money", "trial"],
    ["imessage", "icloud", "app store", "facetime", "music", "down", "server"],
    ["repair", "warranty", "applecare", "store", "appointment", "fix", "send"],
    ["how do i", "how to", "transfer", "password", "apple id", "reset", "backup"],
    ["hate", "sucks", "dark mode", "feature", "why", "wtf", "terrible"],
]


def main():
    df = pd.read_csv(SAMPLE_PATH)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    per_bucket = TARGET_SIZE // (len(SAMPLING_BUCKETS) + 1)  # +1 for a random leftover bucket
    selected_idx = set()
    rows = []

    for keywords in SAMPLING_BUCKETS:
        bucket_matches = df[df['customer_message'].str.lower().apply(
            lambda m: any(kw in str(m) for kw in keywords)
        )]
        bucket_matches = bucket_matches[~bucket_matches.index.isin(selected_idx)]
        picked = bucket_matches.sample(min(per_bucket, len(bucket_matches)), random_state=42)
        selected_idx.update(picked.index)
        rows.append(picked)

    # Fill remainder with pure random samples (covers messages that don't
    # match any keyword bucket -- important so the labeler also sees
    # "unexpected"/edge-case messages, not just keyword-matched ones)
    remaining = df[~df.index.isin(selected_idx)]
    n_remaining_needed = TARGET_SIZE - sum(len(r) for r in rows)
    if n_remaining_needed > 0:
        rows.append(remaining.sample(min(n_remaining_needed, len(remaining)), random_state=42))

    candidates = pd.concat(rows).drop_duplicates(subset=['root_id']).reset_index(drop=True)
    candidates = candidates.sample(frac=1, random_state=7).reset_index(drop=True)  # shuffle order

    out = pd.DataFrame({
        "id": range(1, len(candidates) + 1),
        "root_id": candidates['root_id'],
        "customer_message": candidates['customer_message'],
        "full_thread": candidates['full_thread'],
        "is_dm_redirect": candidates['is_dm_redirect'],
        "historical_resolution": candidates['final_resolution'],
        # Empty columns for YOU to fill in during labeling:
        "ground_truth_intent": "",
        "ground_truth_escalate": "",
        "escalation_reason_category": "",
        "reference_reply": "",
        "labeler_notes": "",
    })

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(out)} unlabeled candidates to {OUTPUT_PATH}")
    print("\nNext step: run 'python eval/label_interactively.py' to label these quickly via CLI,")
    print("or open the CSV directly in Excel and fill in the empty columns by hand.")


if __name__ == "__main__":
    main()