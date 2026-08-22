"""
schemas.py - Pydantic models for request/response validation.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

ReasonCode = Literal["EMI_FAIL", "SUB_LAPSE", "B2B_INVOICE", "CARD_EXPIRED", "DISPUTE"]
PlanCode = Literal["FULL_NOW", "INSTALLMENT_3", "INSTALLMENT_6", "DISCOUNT_FULL"]


class CustomerProfile(BaseModel):
    customer_id: str = "DEMO_CUSTOMER"
    amount_owed: float = Field(..., gt=0, description="Amount overdue in INR")
    days_overdue: int = Field(..., ge=0)
    past_default_count: int = Field(0, ge=0)
    past_ontime_pct: float = Field(0.7, ge=0, le=1)
    reminder_open_rate: float = Field(0.5, ge=0, le=1)
    reason_code: ReasonCode = "EMI_FAIL"


class PlanScore(BaseModel):
    plan: PlanCode
    recovery_probability: float
    plan_description: str


class ScoreResponse(BaseModel):
    customer_id: str
    scores: list[PlanScore]
    recommended_plan: PlanCode


class ChatMessage(BaseModel):
    role: Literal["customer", "agent", "system"]
    content: str


class NegotiationRequest(BaseModel):
    customer: CustomerProfile
    conversation: list[ChatMessage] = []
    customer_message: str


class NegotiationResponse(BaseModel):
    agent_reply: str
    offered_plan: Optional[PlanCode] = None
    escalated: bool = False
    escalation_reason: Optional[str] = None
    plans_considered: list[PlanScore] = []
