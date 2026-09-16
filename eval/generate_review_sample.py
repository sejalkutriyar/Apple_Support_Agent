"""
Generates a judge-vs-human evaluation sample the RIGHT way:

1. Sample N examples from the golden set (stratified across intents).
2. Run the ACTUAL pipeline (retrieval + reply_generator) to get REAL
   system-generated draft replies -- not the reference_reply.
3. Score each generated reply with the LLM judge.
4. Save TWO files:
   - eval/human_blind_sheet.csv   -> for the human (you) to fill in scores,
     WITHOUT seeing the judge's scores (avoids anchoring bias).
   - eval/judge_scores_cache.csv  -> judge's scores, kept separate until
     you're done labeling, then merged by human_agreement.py.

Run this once. Then open human_blind_sheet.csv, fill in the human_* columns
(1-5 each) for every row, save it, and run human_agreement.py.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.retrieval import HistoricalRetriever
from src.reply_generator import GroundedReplyGenerator
from eval.judge import LLMJudge

N_SAMPLES = 20
GOLDEN_PATH = "data/golden/golden_set.csv"
BLIND_SHEET_PATH = "eval/human_blind_sheet.csv"
JUDGE_CACHE_PATH = "eval/judge_scores_cache.csv"


def main():
    print(f"Loading golden set from {GOLDEN_PATH}...")
    df = pd.read_csv(GOLDEN_PATH)

    # Prefer non-escalated cases since those are the ones that actually get a
    # generated reply worth judging (escalated cases just get a handoff message).
    # NOTE: simple random sampling is used here (not stratified groupby-apply)
    # because pandas 3.x changed groupby().apply() to drop the grouping column
    # from the sub-frame passed to the function, which breaks column access
    # inside the lambda. Simple random sampling avoids that entirely and is
    # sufficient for a 36-40 example agreement-check sample.
    df_auto = df[df['ground_truth_escalate'] == False].copy()
    if len(df_auto) < N_SAMPLES:
        print(f"Warning: only {len(df_auto)} non-escalated examples available.")
    sample = df_auto.sample(n=min(N_SAMPLES, len(df_auto)), random_state=42).reset_index(drop=True)

    print(f"Sampled {len(sample)} examples for judge-human agreement evaluation.")

    retriever = HistoricalRetriever()
    generator = GroundedReplyGenerator()
    judge = LLMJudge()

    blind_rows = []
    judge_rows = []

    for i, row in sample.iterrows():
        msg = str(row['customer_message'])
        print(f"[{i+1}/{len(sample)}] Generating + judging reply for id #{row['id']}...", flush=True)

        # Step 1: REAL retrieval + REAL reply generation (this is the actual system output)
        retrieved_cases = retriever.search(msg, top_k=3)
        reply_res = generator.generate_reply(
            customer_message=msg,
            intent=row['ground_truth_intent'],
            retrieved_cases=retrieved_cases
        )
        draft_reply = reply_res['draft_reply']

        # Step 2: Judge scores the REAL generated reply against the REAL retrieved context
        grounding_text = "\n".join([c['resolution'] for c in retrieved_cases]) if retrieved_cases else ""
        judge_res = judge.evaluate_reply(
            customer_message=msg,
            grounding_context=grounding_text,
            draft_reply=draft_reply
        )

        blind_rows.append({
            "id": row['id'],
            "customer_message": msg,
            "generated_draft_reply": draft_reply,
            "human_groundedness": "",
            "human_tone_match": "",
            "human_helpfulness": "",
            "human_correctness": "",
        })

        judge_rows.append({
            "id": row['id'],
            "customer_message": msg,
            "generated_draft_reply": draft_reply,
            "judge_groundedness": judge_res.get("groundedness"),
            "judge_tone_match": judge_res.get("tone_match"),
            "judge_helpfulness": judge_res.get("helpfulness"),
            "judge_correctness": judge_res.get("correctness"),
            "judge_overall_score": judge_res.get("overall_score"),
            "judge_justification": judge_res.get("justification"),
        })

    pd.DataFrame(blind_rows).to_csv(BLIND_SHEET_PATH, index=False)
    pd.DataFrame(judge_rows).to_csv(JUDGE_CACHE_PATH, index=False)

    print(f"\nSaved blind labeling sheet -> {BLIND_SHEET_PATH}")
    print(f"Saved judge scores cache   -> {JUDGE_CACHE_PATH}")
    print("\nNEXT STEP: Open human_blind_sheet.csv, read each generated_draft_reply,")
    print("and fill in human_groundedness / human_tone_match / human_helpfulness /")
    print("human_correctness with your own 1-5 scores WITHOUT looking at judge_scores_cache.csv.")
    print("Then run: python eval/human_agreement.py")


if __name__ == "__main__":
    main()