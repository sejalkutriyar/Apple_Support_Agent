"""
Applies the specific corrections identified by validate_golden_set.py's output
to golden_set.csv. Run this ONCE (with the CSV closed in Excel), then re-run
validate_golden_set.py to confirm everything is clean.

Run: python eval/apply_manual_corrections.py
"""
import pandas as pd

GOLDEN_PATH = "data/golden/golden_set.csv"

# --- Check 1 fixes: intent corrections (same recurring topic, was inconsistent) ---
INTENT_FIXES = {
    4: "software_bug_report",
    168: "software_bug_report",
    33: "software_bug_report",
    63: "software_bug_report",
    23: "account_howto_question",
    31: "account_howto_question",
    50: "general_complaint_feedback",
    91: "software_bug_report",
    162: "billing_payment_issue",
}

# --- Check 2 fixes: escalate to True to match historical DM-redirect pattern ---
ESCALATE_TRUE_IDS = [6, 8, 12, 17, 24, 27, 33, 37, 38, 39, 41, 51, 52, 53, 61, 65, 71, 73, 74, 76, 78, 79]

# --- Check 3 fixes: broken reference_reply cells -> realistic replacement text ---
REFERENCE_REPLY_FIXES = {
    12: "We're sorry about that. Please DM us your country and the email associated with your Apple ID so our specialists can help reactivate it.",
    18: "Thanks for reaching out. We have a workaround for this here, and recommend updating to the latest iOS to fully resolve it.",
    57: "We hear you. Please back up your device and update to iOS 11.1.2, which resolves this autocorrect issue.",
    143: "We recently released an update that addresses keyboard issues. Please back up your device and update to the latest iOS version.",
    148: "This is a known issue with the 'i' autocorrect bug. Updating to iOS 11.1.1 or later resolves it.",
}


def main():
    df = pd.read_csv(GOLDEN_PATH)

    n_intent = 0
    for id_, intent in INTENT_FIXES.items():
        mask = df['id'] == id_
        if mask.any():
            df.loc[mask, 'ground_truth_intent'] = intent
            n_intent += 1

    n_escalate = 0
    for id_ in ESCALATE_TRUE_IDS:
        mask = df['id'] == id_
        if mask.any():
            df.loc[mask, 'ground_truth_escalate'] = True
            # Only set a reason if one isn't meaningfully set already
            current_reason = df.loc[mask, 'escalation_reason_category'].values[0]
            if str(current_reason) in ("NONE", "nan", ""):
                df.loc[mask, 'escalation_reason_category'] = "OTHER"
            n_escalate += 1

    n_reply = 0
    for id_, reply in REFERENCE_REPLY_FIXES.items():
        mask = df['id'] == id_
        if mask.any():
            df.loc[mask, 'reference_reply'] = reply
            n_reply += 1

    df.to_csv(GOLDEN_PATH, index=False)

    print(f"Applied {n_intent} intent corrections.")
    print(f"Applied {n_escalate} escalate corrections.")
    print(f"Applied {n_reply} reference_reply corrections.")
    print(f"\nSaved -> {GOLDEN_PATH}")
    print("\nNow run: python eval/validate_golden_set.py to confirm everything is clean.")


if __name__ == "__main__":
    main()