"""
agent.py
---------
The negotiation agent. This is the LLM layer, and it is DELIBERATELY
constrained: it can only select from the plans that scoring.py has
already ranked, it can never invent a discount/amount/term of its own,
and it must escalate to a human in a defined set of situations.

This boundary is the core "mature system design" answer to the
interview question: "Should an LLM be allowed to directly control
financial decisions?" -> No. The LLM chooses which pre-approved option
fits the conversation and writes natural language; it does not decide
what the options are or invent new ones.

Uses Groq API (free tier, no billing required for light use).
Requires the GROQ_API_KEY environment variable.
"""

import json
import os

from groq import Groq

from scoring import score_all_plans

# Lazily instantiated so the server can start (and /score-plans can be used)
# even if GEMINI_API_KEY isn't set yet - the key is only required at the
# moment a negotiation turn is actually run.
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return _client


MIN_VIABLE_PROBABILITY = 0.30  # if every plan is below this, escalate instead of offering

SYSTEM_PROMPT = """You are a payment recovery negotiation assistant for a fintech company.

You are talking to a customer who owes money (failed EMI, lapsed subscription,
or overdue B2B invoice). Your job is to have a calm, respectful, human-sounding
conversation and guide them toward ONE of the pre-approved payment plans below.

STRICT RULES (never break these):
1. You may ONLY offer plans from the "approved_plans" list given to you. Never
   invent a discount percentage, amount, or schedule that isn't in that list.
2. Never be pushy, threatening, or guilt-inducing. This should feel like a
   helpful conversation, not a collections call from a bad movie.
3. If the customer disputes the charge entirely, becomes hostile/abusive, asks
   for something outside all approved plans, or repeatedly rejects every
   approved plan, you must escalate to a human agent instead of continuing
   to negotiate.
4. Keep replies short (2-4 sentences), natural, and specific to what the
   customer just said.
5. Always respond with ONLY a single JSON object, no other text, matching
   this exact schema:
{
  "reply": "<what you say to the customer>",
  "offered_plan": "<one of the approved plan codes, or null if escalating/just responding>",
  "escalate": <true or false>,
  "escalation_reason": "<short reason, or null>"
}
"""


def _build_user_prompt(customer: dict, approved_plans: list[dict], conversation: list[dict], customer_message: str) -> str:
    history_lines = []
    for turn in conversation:
        speaker = "Customer" if turn["role"] == "customer" else "Agent"
        history_lines.append(f"{speaker}: {turn['content']}")
    history_text = "\n".join(history_lines) if history_lines else "(no prior messages)"

    plans_text = "\n".join(
        f"- {p['plan']}: {p['plan_description']} "
        f"(model-predicted recovery probability: {p['recovery_probability']:.0%})"
        for p in approved_plans
    )

    return f"""Customer profile:
- Amount owed: Rs.{customer['amount_owed']:.2f}
- Days overdue: {customer['days_overdue']}
- Past defaults: {customer['past_default_count']}
- Historical on-time payment rate: {customer['past_ontime_pct']:.0%}
- Reason for default: {customer['reason_code']}

Approved plans you may offer (ranked best to worst by predicted recovery):
{plans_text}

Conversation so far:
{history_text}

New customer message:
"{customer_message}"

Respond with the JSON object described in your instructions."""


def run_negotiation_turn(customer: dict, conversation: list[dict], customer_message: str) -> dict:
    ranked_plans = score_all_plans(customer)

    # Filter out plans already explicitly rejected earlier in the conversation
    already_offered = {
        turn.get("offered_plan")
        for turn in conversation
        if turn.get("role") == "agent" and turn.get("offered_plan")
    }
    # naive rejection tracking: if customer said something like "no" right after
    # a plan was offered, drop that plan from consideration
    rejected_plans = set()
    for i, turn in enumerate(conversation):
        if turn["role"] == "agent" and turn.get("offered_plan"):
            if i + 1 < len(conversation) and conversation[i + 1]["role"] == "customer":
                reply = conversation[i + 1]["content"].lower()
                if any(w in reply for w in ["no", "can't", "cannot", "won't", "too much", "still tough"]):
                    rejected_plans.add(turn["offered_plan"])

    approved_plans = [p for p in ranked_plans if p["plan"] not in rejected_plans]

    # Escalation guardrail 1: no viable plans left
    if not approved_plans or all(p["recovery_probability"] < MIN_VIABLE_PROBABILITY for p in approved_plans):
        return {
            "reply": (
                "I understand none of our standard options work for your situation. "
                "Let me connect you with a specialist who can look at more flexible terms."
            ),
            "offered_plan": None,
            "escalate": True,
            "escalation_reason": "No approved plan meets the minimum viable recovery probability.",
            "plans_considered": ranked_plans,
        }

    user_prompt = _build_user_prompt(customer, approved_plans, conversation, customer_message)

    client = _get_client()
    response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ],
    temperature=0.7,
    max_tokens=500,
)

    raw_text = response.choices[0].message.content.strip()

    try:
        # strip potential code fences just in case
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError):
        parsed = {
            "reply": "I'm sorry, I had trouble processing that. Could you rephrase?",
            "offered_plan": None,
            "escalate": False,
            "escalation_reason": None,
        }

    # Guardrail 2: never trust an offered_plan the model hallucinates outside the approved set
    approved_codes = {p["plan"] for p in approved_plans}
    if parsed.get("offered_plan") and parsed["offered_plan"] not in approved_codes:
        parsed["offered_plan"] = approved_plans[0]["plan"]  # fall back to best-ranked approved plan

    parsed["plans_considered"] = ranked_plans
    return parsed
