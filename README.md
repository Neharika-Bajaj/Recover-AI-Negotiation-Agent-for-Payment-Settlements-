# Recover — AI Negotiation Agent for Payment Settlements

**Track:** AI Revenue Recovery (Razorpay AI Internship, Sept 2026)

An agentic system that recovers overdue payments (failed EMIs, lapsed
subscriptions, overdue B2B invoices) by *negotiating* with the customer,
instead of just predicting whether they'll default.

Most "AI collections" projects stop at: `predict_default_probability(customer) -> 0.73`.
This project goes further and asks a more useful question: **given several
possible payment plans, which one has the best chance of actually getting
the money back — and how do we get there through a real conversation?**

---

## 1. The problem

When a customer misses a payment, a company has a menu of options it could
offer them (pay in full now, split into installments, a discount for paying
immediately, etc). Picking the wrong option — or just demanding full payment
— often means the company recovers nothing when a more flexible plan would
have worked. Manually negotiating every overdue account doesn't scale.

## 2. The architecture

```
                    ┌─────────────────────────┐
                    │   Customer Profile        │
                    │ (amount, days overdue,    │
                    │  payment history, reason) │
                    └────────────┬──────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   ML Recovery Model       │   <- "the calculator"
                    │ (Random Forest, selected  │      pure ML, no LLM
                    │  after comparing 3 models)│
                    └────────────┬──────────────┘
                                 │
                 scores EVERY candidate plan:
        FULL_NOW · INSTALLMENT_3 · INSTALLMENT_6 · DISCOUNT_FULL
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Negotiation Agent (LLM) │   <- "the conversationalist"
                    │  - reads customer message │      constrained: can ONLY
                    │  - picks best-fit APPROVED│      pick from ML-approved
                    │    plan                   │      plans, never invents
                    │  - writes natural reply    │      terms of its own
                    │  - escalates when needed   │
                    └────────────┬──────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Human agent (escalation) │
                    └─────────────────────────┘
```

**The core design decision, and the answer to "should an LLM control
financial decisions":** No. The ML model is the only thing that determines
which plans exist and their recovery probability. The LLM's job is strictly
limited to (a) reading the conversation, (b) picking the best-fit plan from
the pre-approved, pre-scored list, and (c) writing the natural-language
reply. It is structurally incapable of inventing a discount or term that
wasn't already validated by the model — enforced in code (`agent.py`), not
just by prompting.

## 3. The ML layer — model selection, not model guessing

Trained and compared three models on a synthetic but realistically-structured
dataset (`data/generate_data.py`) of 4,000 customers × 4 plans = 16,000
customer–plan observations:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.779 | 0.797 | 0.927 | 0.857 | **0.783** |
| **Random Forest (selected)** | 0.780 | 0.794 | **0.935** | 0.859 | 0.779 |
| Gradient Boosting | 0.773 | 0.795 | 0.920 | 0.853 | 0.775 |

**Why we optimize for recall, not accuracy:** in a revenue-recovery context,
a false negative (predicting a customer won't pay when they actually would
have, under the right plan) is a customer we give up on and lose money from.
A false positive (offering a generous plan to someone who was always going
to pay in full anyway) is a much smaller cost. Random Forest was selected
for its highest recall.

Top features by importance: **which plan is offered** dominates (as
expected — this is the whole point of the project), followed by the
customer's historical on-time payment rate, amount owed, and past default
count.

## 4. The agent layer — guardrails, not just prompting

`backend/agent.py` implements three hard guardrails, enforced in code:

1. **Plan whitelist enforcement** — if the LLM's JSON output names a plan
   outside the ML-approved set (hallucination), the code silently falls
   back to the best-ranked approved plan rather than trusting the model.
2. **Automatic escalation** — if every approved plan's recovery probability
   is below a minimum threshold (30%), the agent escalates to a human
   instead of continuing to negotiate a lost cause.
3. **Rejection tracking** — if the customer declines an offered plan, it's
   removed from consideration in the next turn, so the agent doesn't loop
   offering the same rejected plan.

The agent also escalates on dispute, hostility, or requests outside all
approved plans (enforced via the system prompt + the JSON schema it must
return).

## 5. Project structure

```
negotiation-agent/
├── data/
│   ├── generate_data.py      # synthetic dataset generator
│   └── customers.csv         # generated output (4000 customers x 4 plans)
├── ml/
│   ├── train_models.py       # trains & compares LR / RF / GB, saves best
│   ├── recovery_model.pkl    # saved model + encoders (generated)
│   └── model_comparison.json # metrics for all 3 models (generated)
├── backend/
│   ├── main.py                # FastAPI app (endpoints)
│   ├── scoring.py             # ML scoring layer ("the calculator")
│   ├── agent.py                # LLM negotiation layer + guardrails
│   └── schemas.py              # Pydantic request/response models
├── frontend/
│   └── index.html              # live chat demo UI (no build step needed)
├── requirements.txt
└── README.md
```

## 6. How to run locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the synthetic dataset
python data/generate_data.py

# 3. Train and compare the models (saves the best one)
python ml/train_models.py

# 4. Set your Gemini API key (needed for the live negotiation endpoint)
#    Get a free key at https://aistudio.google.com/apikey - no billing required
export GEMINI_API_KEY=your-key-here

# 5. Start the backend
cd backend
uvicorn main:app --reload --port 8000

# 6. Open the frontend
# Just open frontend/index.html directly in your browser
# (it calls http://localhost:8000 — make sure the backend is running)
```

## 7. API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | health check |
| `/model-info` | GET | returns the 3-model comparison metrics |
| `/score-plans` | POST | pure ML: ranks all 4 plans by recovery probability for a given customer |
| `/negotiate` | POST | full pipeline: ML scoring + one LLM negotiation turn |

Example `/score-plans` request:
```json
{
  "amount_owed": 8500,
  "days_overdue": 25,
  "past_default_count": 1,
  "past_ontime_pct": 0.6,
  "reminder_open_rate": 0.4,
  "reason_code": "EMI_FAIL"
}
```

## 8. Demo flow

1. Open `frontend/index.html`, fill in a customer profile, click **Load
   Customer & Score Plans** — the left panel shows all 4 plans ranked by
   ML-predicted recovery probability.
2. Type as the customer in the chat (e.g. *"I can't pay it all right
   now"*) — the agent reads it, picks the best-fit approved plan, and
   replies naturally, with the offered plan tagged on its message.
3. Try rejecting an offer (*"still too much"*) — watch the plan get
   struck through in the left panel and the agent pivot to the next-best
   approved option.
4. Try a hostile or dispute-style message — watch the agent escalate
   instead of continuing to negotiate.

## 9. Challenges faced

- **Avoiding the "fake AI" trap**: an early version let the LLM freely
  decide payment terms. This was a bad design — indefensible under
  questioning ("why should an LLM decide someone's financial terms?").
  Rebuilt so the LLM can only *select from* ML-approved options.
- **Preventing the agent from re-offering rejected plans**: added explicit
  rejection tracking by scanning conversation history for negative
  responses immediately after an offer, rather than relying on the LLM to
  remember on its own.
- **JSON reliability from the LLM**: the model occasionally wraps JSON in
  markdown code fences or adds stray text; the backend strips fences and
  falls back to a safe default reply if parsing fails, rather than
  crashing the conversation.

## 10. Honest limitations & future work

- **Synthetic data**: no real collections data is publicly available for
  a project like this; the dataset is generated with realistic feature
  relationships (documented in `data/generate_data.py`) but is not real
  customer behavior.
- **Single-turn escalation logic**: the escalation rules are deliberately
  simple (probability threshold + keyword-based rejection detection) for
  transparency and defensibility. A production system would use a more
  robust sentiment/intent classifier.
- **Reinforcement learning extension (not implemented)**: a natural next
  step would be framing offer *sequencing* itself as an RL problem — which
  plan to lead with, given the conversation state, to maximize expected
  recovery. This wasn't implemented here because it would require either
  real interaction data or a carefully validated simulator, and a poorly
  validated one would just be learning the project author's own
  assumptions back — not a genuine improvement. Noted here as a
  deliberate scope decision, not an oversight.

## 11. Tech stack

- **ML**: scikit-learn (Logistic Regression, Random Forest, Gradient Boosting)
- **Backend**: FastAPI, Pydantic
- **Agent**: Google Gemini API (free tier), JSON-constrained output with code-level guardrails
- **Frontend**: vanilla HTML/CSS/JS (no build step, single file)
- **Data**: pandas, numpy
