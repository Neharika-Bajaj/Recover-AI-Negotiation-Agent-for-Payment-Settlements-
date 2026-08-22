"""
generate_data.py
-----------------
Generates a realistic SYNTHETIC dataset of overdue customers for the
AI Negotiation Agent for Payment Settlements project.

Why synthetic: real collections/recovery data is never publicly available
(it's sensitive financial data). This script builds a dataset with
realistic feature relationships so the trained model behaves sensibly,
and is clearly documented as synthetic for the README / interview.

Output: data/customers.csv
Columns:
    customer_id
    amount_owed            -> INR amount overdue
    days_overdue            -> how many days past due
    past_default_count      -> number of prior defaults
    past_ontime_pct         -> historical on-time payment rate (0-1)
    reminder_open_rate      -> fraction of reminder emails/sms opened (0-1)
    reason_code             -> categorical: EMI_FAIL, SUB_LAPSE, B2B_INVOICE, CARD_EXPIRED, DISPUTE
    plan_offered            -> categorical: FULL_NOW, INSTALLMENT_3, INSTALLMENT_6, DISCOUNT_FULL
    recovered               -> target label (1 = paid under this plan, 0 = did not)
"""

import os

import numpy as np
import pandas as pd

_OUT_DIR = os.path.join(os.path.dirname(__file__))

RNG = np.random.default_rng(42)
N_CUSTOMERS = 4000
PLANS = ["FULL_NOW", "INSTALLMENT_3", "INSTALLMENT_6", "DISCOUNT_FULL"]
REASONS = ["EMI_FAIL", "SUB_LAPSE", "B2B_INVOICE", "CARD_EXPIRED", "DISPUTE"]

# Each customer gets a "true underlying willingness/ability to pay" latent score.
# We then simulate how that customer would respond to EACH of the 4 plans,
# so the final dataset has ~N_CUSTOMERS * 4 rows: one row per (customer, plan).
# This is what lets the model later predict recovery probability PER PLAN
# for a given customer (the "what-if" comparison table used by the agent).

rows = []
for i in range(N_CUSTOMERS):
    amount_owed = float(np.round(RNG.gamma(shape=2.2, scale=3500) + 500, 2))
    days_overdue = int(np.clip(RNG.gamma(shape=2.0, scale=12), 1, 180))
    past_default_count = int(RNG.poisson(0.6))
    past_ontime_pct = float(np.clip(RNG.beta(5, 2), 0, 1))
    reminder_open_rate = float(np.clip(RNG.beta(3, 2), 0, 1))
    reason_code = RNG.choice(REASONS, p=[0.30, 0.25, 0.15, 0.20, 0.10])

    # Latent "ability & willingness to pay" score (not observed directly by the model,
    # only used to *simulate* realistic outcomes)
    latent = (
        2.2 * past_ontime_pct
        + 1.1 * reminder_open_rate
        - 0.55 * past_default_count
        - 0.010 * days_overdue
        - 0.00006 * amount_owed
        + RNG.normal(0, 0.35)
    )
    if reason_code == "DISPUTE":
        latent -= 0.9  # disputes are much harder to recover
    if reason_code == "CARD_EXPIRED":
        latent += 0.4  # easy fix, high recovery

    for plan in PLANS:
        # Each plan shifts the effective "ease of paying" differently.
        if plan == "FULL_NOW":
            ease = latent - 1.4  # hardest: full amount immediately
        elif plan == "INSTALLMENT_3":
            ease = latent + 0.35
        elif plan == "INSTALLMENT_6":
            ease = latent + 0.75  # easiest cash-flow wise
        elif plan == "DISCOUNT_FULL":
            ease = latent - 0.25 + 0.5  # discount offsets some of the "full now" pain

        prob = 1 / (1 + np.exp(-ease))  # sigmoid -> probability of recovery
        recovered = int(RNG.random() < prob)

        rows.append(
            {
                "customer_id": f"C{i:05d}",
                "amount_owed": amount_owed,
                "days_overdue": days_overdue,
                "past_default_count": past_default_count,
                "past_ontime_pct": round(past_ontime_pct, 3),
                "reminder_open_rate": round(reminder_open_rate, 3),
                "reason_code": reason_code,
                "plan_offered": plan,
                "recovered": recovered,
            }
        )

df = pd.DataFrame(rows)
df.to_csv(os.path.join(_OUT_DIR, "customers.csv"), index=False)

print(f"Generated {len(df)} rows for {N_CUSTOMERS} synthetic customers x {len(PLANS)} plans")
print(df.groupby("plan_offered")["recovered"].mean().rename("recovery_rate"))
print(df.head())
