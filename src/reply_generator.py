import json
import ollama

SYSTEM_PROMPT = """You are AppleSupport's official AI assistant.
Your task is to draft a helpful, professional, concise reply (Twitter support style) for an incoming customer tweet.

CRITICAL GROUNDING RULES:
1. Base your reply ONLY on the historical AppleSupport resolutions provided below.
2. DO NOT invent web links, contact numbers, or steps that are not present in the historical cases.
3. Keep the tone friendly, helpful, and empathetic, matching AppleSupport's voice.
4. At the end of your reply, explicitly state which Case ID(s) you cited using the format: "[Grounded in Case #<ID>]".

HISTORICAL GROUNDING CASES:
{grounding_context}

INCOMING CUSTOMER MESSAGE:
"{customer_message}"
INTENT: {intent}

Respond with a JSON object containing:
- "draft_reply": The exact reply text including case citation tag at the end.
- "citations": A list of Case IDs cited (e.g. ["123456"]).
- "reasoning": Brief explanation of how the historical precedent informed this reply.
"""

class GroundedReplyGenerator:
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.model_name = model_name

    def generate_reply(self, customer_message: str, intent: str, retrieved_cases: list[dict]) -> dict:
        """Drafts a grounded reply with explicit citations based on historical cases."""
        if not retrieved_cases:
            return {
                "draft_reply": "We'd love to look into this with you! Please send us a Direct Message with your device details so we can help.",
                "citations": [],
                "reasoning": "No grounded historical cases retrieved. Using safe fall-back DM invite.",
                "raw_response": ""
            }

        # Format historical cases into context block
        context_blocks = []
        for i, case in enumerate(retrieved_cases, 1):
            context_blocks.append(
                f"Case #{case['case_id']} (Similarity: {case['similarity_score']:.2f}):\n"
                f"  Customer asked: \"{case['customer_message']}\"\n"
                f"  AppleSupport resolved: \"{case['resolution']}\""
            )
        grounding_context = "\n\n".join(context_blocks)

        prompt = SYSTEM_PROMPT.format(
            grounding_context=grounding_context,
            customer_message=customer_message,
            intent=intent
        )

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2}
            )
            raw_text = response['message']['content'].strip()

            # Attempt JSON parse
            parsed = self._parse_json(raw_text)
            if parsed and "draft_reply" in parsed:
                return {
                    "draft_reply": parsed["draft_reply"],
                    "citations": parsed.get("citations", [c['case_id'] for c in retrieved_cases[:1]]),
                    "reasoning": parsed.get("reasoning", "Grounded in top retrieved historical case."),
                    "raw_response": raw_text
                }

            # Fallback parsing
            return {
                "draft_reply": raw_text,
                "citations": [c['case_id'] for c in retrieved_cases[:1]],
                "reasoning": "Generated reply from raw LLM output.",
                "raw_response": raw_text
            }

        except Exception as e:
            # Fallback when Ollama server is offline (e.g. Streamlit Cloud)
            top_case = retrieved_cases[0]
            return {
                "draft_reply": f"Hi! Regarding your query, here is how AppleSupport historically resolved similar issues: {top_case['resolution']} [Grounded in Case #{top_case['case_id']}]",
                "citations": [top_case['case_id']],
                "reasoning": f"Grounded in top retrieved historical case (Ollama offline/cloud mode).",
                "raw_response": ""
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
    generator = GroundedReplyGenerator()
    dummy_cases = [{
        "case_id": "1001",
        "customer_message": "My battery is draining fast after update",
        "resolution": "We recommend restarting your device and checking Battery Usage in Settings > Battery. Let us know if issues persist in DM.",
        "similarity_score": 0.88,
        "num_turns": 3
    }]
    res = generator.generate_reply(
        customer_message="iOS update killed my battery!",
        intent="software_bug_report",
        retrieved_cases=dummy_cases
    )
    print("Drafted Reply:")
    print(res["draft_reply"])
    print("Citations:", res["citations"])
