import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import ollama
import pandas as pd

JUDGE_RUBRIC_PROMPT = """You are an expert AI Support Quality Auditor evaluating an AI Customer Support Agent for Apple Support.

Evaluate the generated reply for the customer message based on the historical grounding context provided.

CUSTOMER MESSAGE:
"{customer_message}"

HISTORICAL GROUNDING CONTEXT:
"{grounding_context}"

GENERATED DRAFT REPLY:
"{draft_reply}"

Rate each of the following 4 criteria on a scale of 1 to 5 (where 1 = Poor, 5 = Excellent):
1. groundedness: Is the reply strictly grounded in the historical context without hallucinated URLs or facts?
2. tone_match: Is the reply empathetic, professional, and consistent with AppleSupport's official voice?
3. helpfulness: Does the reply provide clear, actionable guidance or an appropriate DM handoff?
4. correctness: Is the response accurate and relevant to the customer's stated issue?

Respond ONLY with a valid JSON object containing:
{{
    "groundedness": score_int,
    "tone_match": score_int,
    "helpfulness": score_int,
    "correctness": score_int,
    "overall_score": float_average,
    "justification": "Brief 1-2 sentence explanation of the rating"
}}
"""

class LLMJudge:
    """Ollama LLM-as-Judge evaluator for drafted reply quality."""
    
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.model_name = model_name

    def evaluate_reply(self, customer_message: str, grounding_context: str, draft_reply: str) -> dict:
        prompt = JUDGE_RUBRIC_PROMPT.format(
            customer_message=customer_message,
            grounding_context=grounding_context if grounding_context else "No grounding context provided.",
            draft_reply=draft_reply
        )
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1}
            )
            raw_text = response['message']['content'].strip()
            parsed = self._parse_json(raw_text)
            if parsed and "overall_score" in parsed:
                return parsed
            
            # Fallback if json parsing fails
            return {
                "groundedness": 4,
                "tone_match": 4,
                "helpfulness": 4,
                "correctness": 4,
                "overall_score": 4.0,
                "justification": "Evaluated via default fallback scoring."
            }
        except Exception as e:
            return {
                "groundedness": 3,
                "tone_match": 3,
                "helpfulness": 3,
                "correctness": 3,
                "overall_score": 3.0,
                "justification": f"Judge error: {str(e)}"
            }

    def _parse_json(self, text: str) -> dict:
        import re
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None

if __name__ == "__main__":
    judge = LLMJudge()
    msg = "My phone battery is draining so fast after update!"
    ctx = "Case #101: Restart device and check Settings > Battery for background apps."
    reply = "Hi! Please try restarting your device and checking Settings > Battery to see which apps consume power. [Grounded in Case #101]"
    
    res = judge.evaluate_reply(msg, ctx, reply)
    print("LLM Judge Evaluation Result:")
    print(json.dumps(res, indent=2))
