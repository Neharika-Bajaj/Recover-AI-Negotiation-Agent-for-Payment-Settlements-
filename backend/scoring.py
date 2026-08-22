"""
scoring.py
-----------
Loads the trained recovery model and exposes a single function:
    score_all_plans(customer) -> ranked list of (plan, recovery_probability)

This is the "calculator" layer - pure ML, no LLM involved. The agent
layer (agent.py) is only allowed to choose FROM this ranked list, never
invent numbers of its own. This boundary is the core safety property of
the whole system and is the answer to "should an LLM control financial
decisions?" -> No, it operates only within ML-approved bounds.
"""

import os

import numpy as np
import joblib

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "recovery_model.pkl")
_bundle = joblib.load(_MODEL_PATH)

_MODEL = _bundle["model"]
_SCALER = _bundle["scaler"]
_ENCODER = _bundle["encoder"]
_NUMERIC = _bundle["numeric_features"]
_CATEGORICAL = _bundle["categorical_features"]
_PLANS = _bundle["plans"]

PLAN_DESCRIPTIONS = {
    "FULL_NOW": "Pay the full amount immediately",
    "INSTALLMENT_3": "Split into 3 monthly installments",
    "INSTALLMENT_6": "Split into 6 monthly installments",
    "DISCOUNT_FULL": "Pay in full now with a discount incentive",
}


def score_all_plans(customer: dict) -> list[dict]:
    """
    customer: dict with keys amount_owed, days_overdue, past_default_count,
              past_ontime_pct, reminder_open_rate, reason_code
    Returns a list of dicts: [{plan, recovery_probability, plan_description}, ...]
    sorted by recovery_probability descending.
    """
    rows = []
    for plan in _PLANS:
        row = {**customer, "plan_offered": plan}
        rows.append(row)

    import pandas as pd

    numeric_df = pd.DataFrame([[row[f] for f in _NUMERIC] for row in rows], columns=_NUMERIC)
    numeric_scaled = _SCALER.transform(numeric_df)

    cat_df = pd.DataFrame(
        [[row["reason_code"], row["plan_offered"]] for row in rows],
        columns=_CATEGORICAL,
    )
    cat_encoded = _ENCODER.transform(cat_df)

    X = np.hstack([numeric_scaled, cat_encoded])
    probs = _MODEL.predict_proba(X)[:, 1]

    scored = [
        {
            "plan": plan,
            "recovery_probability": round(float(p), 4),
            "plan_description": PLAN_DESCRIPTIONS[plan],
        }
        for plan, p in zip(_PLANS, probs)
    ]
    scored.sort(key=lambda r: r["recovery_probability"], reverse=True)
    return scored
