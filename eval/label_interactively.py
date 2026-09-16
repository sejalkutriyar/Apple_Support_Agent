"""
Step 2 of the golden set pipeline: HUMAN LABELING (the real deliverable).

An interactive terminal tool so you can label ~150-180 examples quickly via
keyboard, instead of clicking around in Excel. Progress is saved after every
single label, so you can stop and resume anytime (close the terminal, come
back tomorrow, whatever -- nothing is lost).

For each message you will:
  1. Read the customer message (and optionally the full historical thread
     for context on how it was actually resolved).
  2. Pick an intent (1-7) using YOUR OWN judgment based on the taxonomy
     definitions shown.
  3. Decide: should this be escalated to a human agent? (y/n)
  4. If yes, pick a reason category.
  5. Optionally edit the reference reply (pre-filled from the historical
     resolution, but you can rewrite it if the historical reply was poor,
     e.g. a DM-redirect with no real content).

Run: python eval/label_interactively.py
"""
import pandas as pd
import os

CANDIDATES_PATH = "data/golden/golden_candidates.csv"
OUTPUT_PATH = "data/golden/golden_set.csv"
TARGET_LABELED = 150  # minimum required by the assignment; can label more

INTENTS = [
    ("software_bug_report", "Issue caused by an OS/app update: freezing, battery drain, restart loops, UI bugs, lag."),
    ("device_hardware_malfunction", "Physical device failure: broken/won't-boot, charger/battery safety issue, faulty part."),
    ("billing_payment_issue", "Wrongly charged, payment declined, refund request, subscription dispute."),
    ("service_status_issue", "A specific Apple service (iMessage, iCloud, App Store, FaceTime) down/not working."),
    ("repair_warranty_request", "Sending for repair, warranty/AppleCare questions, store stock/appointment queries."),
    ("account_howto_question", "Informational 'how do I do X' -- no bug/frustration signal."),
    ("general_complaint_feedback", "Venting/dissatisfaction or feature request with no specific fixable ask."),
]

ESCALATION_REASONS = ["SAFETY_HAZARD", "BILLING_DISPUTE", "REPEAT_FAILURE", "LOW_CONFIDENCE_AMBIGUOUS", "OTHER"]


def load_progress():
    """Load existing labeled output if it exists, so we can resume."""
    if os.path.exists(OUTPUT_PATH):
        return pd.read_csv(OUTPUT_PATH)
    return None


def main():
    candidates = pd.read_csv(CANDIDATES_PATH)
    progress = load_progress()
    labeled_ids = set(progress['id'].tolist()) if progress is not None else set()
    results = progress.to_dict('records') if progress is not None else []

    remaining = candidates[~candidates['id'].isin(labeled_ids)]
    print(f"Already labeled: {len(labeled_ids)} | Remaining: {len(remaining)} | Target: {TARGET_LABELED}\n")

    for _, row in remaining.iterrows():
        if len(results) >= TARGET_LABELED + 30:  # small buffer, stop early if you want extra
            break

        print("\n" + "=" * 70)
        print(f"[{len(results)+1}] CUSTOMER MESSAGE:")
        print(f"  {row['customer_message']}")
        show_ctx = input("\nShow full historical thread for context? (y/N): ").strip().lower()
        if show_ctx == 'y':
            print(f"\n  FULL THREAD:\n  {row['full_thread']}")

        print("\nIntent options:")
        for i, (name, desc) in enumerate(INTENTS, 1):
            print(f"  {i}. {name} -- {desc}")

        while True:
            choice = input("\nYour intent choice (1-7, or 's' to skip, 'q' to quit and save): ").strip().lower()
            if choice == 'q':
                save(results)
                print(f"\nSaved {len(results)} labels so far. Run this script again to continue.")
                return
            if choice == 's':
                break
            if choice.isdigit() and 1 <= int(choice) <= 7:
                intent = INTENTS[int(choice) - 1][0]
                break
            print("Invalid input, try again.")

        if choice == 's':
            continue

        while True:
            esc = input("Should this be escalated to a human? (y/n): ").strip().lower()
            if esc in ('y', 'n'):
                should_escalate = (esc == 'y')
                break
            print("Please type y or n.")

        reason = "NONE"
        if should_escalate:
            print("Escalation reason:")
            for i, r in enumerate(ESCALATION_REASONS, 1):
                print(f"  {i}. {r}")
            while True:
                rchoice = input("Reason (1-5): ").strip()
                if rchoice.isdigit() and 1 <= int(rchoice) <= 5:
                    reason = ESCALATION_REASONS[int(rchoice) - 1]
                    break
                print("Invalid input, try again.")

        print(f"\nHistorical resolution (pre-filled as reference_reply, press Enter to accept, or type a replacement):")
        print(f"  {row['historical_resolution']}")
        custom_reply = input("> ").strip()
        reference_reply = custom_reply if custom_reply else row['historical_resolution']

        notes = input("Any notes on this label (optional, press Enter to skip): ").strip()

        results.append({
            "id": row['id'],
            "root_id": row['root_id'],
            "customer_message": row['customer_message'],
            "ground_truth_intent": intent,
            "ground_truth_escalate": should_escalate,
            "escalation_reason_category": reason,
            "is_dm_redirect": row['is_dm_redirect'],
            "reference_reply": reference_reply,
            "labeler_notes": notes,
        })

        save(results)  # save after every single label -- never lose progress
        print(f"Progress: {len(results)} labeled.")

    save(results)
    print(f"\nDone! {len(results)} examples labeled and saved to {OUTPUT_PATH}")


def save(results):
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()