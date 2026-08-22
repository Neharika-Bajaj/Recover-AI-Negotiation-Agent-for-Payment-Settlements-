"""
main.py
--------
FastAPI backend for the AI Negotiation Agent for Payment Settlements.

Endpoints:
    GET  /health              -> health check
    POST /score-plans         -> pure ML: rank recovery probability per plan
    POST /negotiate           -> full pipeline: ML scoring + LLM negotiation turn
    GET  /model-info          -> which model was selected and comparison metrics

Run locally:
    export GROQ_API_KEY=gsk-...
    cd backend
    uvicorn main:app --reload --port 8000
"""

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agent import run_negotiation_turn
from schemas import (
    ChatMessage,
    CustomerProfile,
    NegotiationRequest,
    NegotiationResponse,
    PlanScore,
    ScoreResponse,
)
from scoring import score_all_plans

app = FastAPI(
    title="AI Negotiation Agent for Payment Settlements",
    description=(
        "ML-scored, LLM-conducted negotiation agent for recovering overdue "
        "payments. Built for the Razorpay AI Internship (Revenue Recovery track)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    path = os.path.join(os.path.dirname(__file__), "..", "ml", "model_comparison.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="model_comparison.json not found. Run ml/train_models.py first.")
    with open(path) as f:
        comparison = json.load(f)
    return {"comparison": comparison}


@app.post("/score-plans", response_model=ScoreResponse)
def score_plans(customer: CustomerProfile):
    """Pure ML endpoint - no LLM call. Returns ranked recovery probability per plan."""
    scored = score_all_plans(customer.model_dump())
    plan_scores = [PlanScore(**s) for s in scored]
    return ScoreResponse(
        customer_id=customer.customer_id,
        scores=plan_scores,
        recommended_plan=plan_scores[0].plan,
    )


@app.post("/negotiate", response_model=NegotiationResponse)
def negotiate(req: NegotiationRequest):
    """Full pipeline: ML scoring + one turn of LLM-conducted negotiation."""
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not set on the server. Export it before starting uvicorn.",
        )

    conversation = [turn.model_dump() for turn in req.conversation]
    result = run_negotiation_turn(
        customer=req.customer.model_dump(),
        conversation=conversation,
        customer_message=req.customer_message,
    )

    return NegotiationResponse(
        agent_reply=result["reply"],
        offered_plan=result.get("offered_plan"),
        escalated=result.get("escalate", False),
        escalation_reason=result.get("escalation_reason"),
        plans_considered=[PlanScore(**p) for p in result.get("plans_considered", [])],
    )
