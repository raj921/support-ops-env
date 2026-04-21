from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field

Priority = Literal["low", "normal", "high", "urgent"]
SupportTeam = Literal[
    "general",
    "billing",
    "access",
    "product",
    "security",
    "compliance",
    "success",
]
CaseStatus = Literal["open", "pending_customer", "escalated", "resolved", "closed"]
CollectionId = Literal["C1", "C2", "C4", "C8"]

ToolName = Literal[
    "inbox.list_cases",
    "inbox.open_case",
    "inbox.merge_case",
    "inbox.add_note",
    "crm.get_account",
    "crm.get_contacts",
    "crm.get_contract",
    "billing.get_invoice",
    "billing.get_subscription",
    "billing.issue_credit",
    "access.get_org_state",
    "access.get_auth_events",
    "access.revoke_sessions",
    "policy.search",
    "workflow.set_priority",
    "workflow.assign_team",
    "workflow.set_status",
    "workflow.add_tags",
    "comms.draft_reply",
    "submit_resolution",
]


class CaseRecord(BaseModel):
    case_id: str
    account_id: str
    company: str
    requester: str
    subject: str
    body: str
    priority: Priority
    assigned_team: SupportTeam
    status: CaseStatus
    tags: List[str] = Field(default_factory=list)
    merged_into: Optional[str] = None
    note_log: List[str] = Field(default_factory=list)
    reply_draft: str = ""
    facts: Dict[str, str] = Field(default_factory=dict)


class ContactRecord(BaseModel):
    contact_id: str
    account_id: str
    name: str
    role: str
    email: str
    facts: Dict[str, str] = Field(default_factory=dict)


class AccountRecord(BaseModel):
    account_id: str
    company: str
    tier: str
    renewal_risk: str
    lifecycle_stage: str
    admin_summary: str
    facts: Dict[str, str] = Field(default_factory=dict)


class ContractRecord(BaseModel):
    account_id: str
    sla: str
    renewal_date: str
    csm: str
    special_terms: str
    facts: Dict[str, str] = Field(default_factory=dict)


class InvoiceRecord(BaseModel):
    invoice_id: str
    account_id: str
    status: str
    amount: float
    summary: str
    facts: Dict[str, str] = Field(default_factory=dict)


class SubscriptionRecord(BaseModel):
    account_id: str
    plan_name: str
    seat_summary: str
    billing_summary: str
    facts: Dict[str, str] = Field(default_factory=dict)


class AccessOrgRecord(BaseModel):
    account_id: str
    sso_state: str
    session_state: str
    admin_fallback: str
    facts: Dict[str, str] = Field(default_factory=dict)


class AccessEventRecord(BaseModel):
    event_id: str
    account_id: str
    occurred_at: str
    summary: str
    facts: Dict[str, str] = Field(default_factory=dict)


class PolicyRecord(BaseModel):
    policy_id: str
    title: str
    body: str
    facts: Dict[str, str] = Field(default_factory=dict)


class ToolCall(BaseModel):
    name: ToolName
    args: Dict[str, Any] = Field(default_factory=dict)


class ToolResultRecord(BaseModel):
    name: str
    ok: bool
    result: Dict[str, Any] = Field(default_factory=dict)
    surfaced_fact_ids: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class ResolutionAnswer(BaseModel):
    primary_case_id: str
    resolved_case_ids: List[str] = Field(default_factory=list)
    final_team: SupportTeam
    final_priority: Priority
    final_status: CaseStatus
    final_tags: List[str] = Field(default_factory=list)
    reply_text: str = ""
    done: bool = False


class SupportOpsAction(Action):
    assistant_message: str = Field(min_length=1)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    answer: Optional[ResolutionAnswer] = None


class SupportOpsObservation(Observation):
    task_id: str
    collection: CollectionId
    task_family: str
    task_title: str
    objective: str
    conversation: List[Dict[str, str]] = Field(default_factory=list)
    tool_results: List[ToolResultRecord] = Field(default_factory=list)
    app_summaries: Dict[str, str] = Field(default_factory=dict)
    progress_score: float = Field(default=0.0, ge=0.0, le=1.0)
    remaining_steps: int = Field(default=0, ge=0)
    visible_case_ids: List[str] = Field(default_factory=list)
    recent_actions: List[str] = Field(default_factory=list)
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)
    penalty_breakdown: Dict[str, float] = Field(default_factory=dict)
    guidance: str = (
        "Investigate with tools before acting. Surface evidence, update the correct cases, "
        "draft a grounded reply, then submit a final resolution."
    )


class SupportOpsState(State):
    task_id: str
    collection: CollectionId
    task_family: str
    task_title: str
    difficulty: str
    step_count: int = Field(default=0, ge=0)
    cases: List[CaseRecord] = Field(default_factory=list)
    accounts: Dict[str, AccountRecord] = Field(default_factory=dict)
    contacts: Dict[str, List[ContactRecord]] = Field(default_factory=dict)
    contracts: Dict[str, ContractRecord] = Field(default_factory=dict)
    invoices: Dict[str, InvoiceRecord] = Field(default_factory=dict)
    subscriptions: Dict[str, SubscriptionRecord] = Field(default_factory=dict)
    access_orgs: Dict[str, AccessOrgRecord] = Field(default_factory=dict)
    access_events: Dict[str, List[AccessEventRecord]] = Field(default_factory=dict)
    policies: List[PolicyRecord] = Field(default_factory=list)
    app_views: Dict[str, str] = Field(default_factory=dict)
    seen_entities: List[str] = Field(default_factory=list)
    seen_facts: List[str] = Field(default_factory=list)
    tool_history: List[str] = Field(default_factory=list)
    conversation: List[Dict[str, str]] = Field(default_factory=list)
    tool_results_history: List[ToolResultRecord] = Field(default_factory=list)
    current_score: float = Field(default=0.0, ge=0.0, le=1.0)
    cumulative_reward: float = 0.0
    invalid_action_count: int = Field(default=0, ge=0)
    no_progress_count: int = Field(default=0, ge=0)
    submitted: bool = False
    submission_requested: bool = False
    reply_case_id: Optional[str] = None
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)
    penalty_breakdown: Dict[str, float] = Field(default_factory=dict)
    unmet_requirements: List[str] = Field(default_factory=list)
    termination_reason: str = ""
    curriculum_stats: Dict[str, Any] = Field(default_factory=dict)
