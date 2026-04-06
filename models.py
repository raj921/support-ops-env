from __future__ import annotations

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field

ActionType = Literal[
    "view_ticket", "set_priority", "assign_team", "add_tags",
    "set_status", "draft_reply", "merge_ticket", "submit",
]
Priority = Literal["low", "normal", "high", "urgent"]
SupportTeam = Literal["general", "billing", "access", "product", "security", "compliance"]
TicketStatus = Literal["open", "pending_customer", "escalated", "resolved", "closed"]


class TicketSnapshot(BaseModel):
    ticket_id: str
    customer_name: str
    company: str
    tier: str
    subject: str
    body: str
    status: TicketStatus
    priority: Priority
    assigned_team: SupportTeam
    tags: List[str] = Field(default_factory=list)
    reply_draft: str = ""
    merged_into: Optional[str] = None


class SupportOpsAction(Action):
    action_type: ActionType
    ticket_id: Optional[str] = None
    target_ticket_id: Optional[str] = None
    priority: Optional[Priority] = None
    team: Optional[SupportTeam] = None
    status: Optional[TicketStatus] = None
    tags: List[str] = Field(default_factory=list)
    reply: Optional[str] = None
    reasoning: Optional[str] = None


class SupportOpsObservation(Observation):
    task_id: str
    task_title: str
    objective: str
    inbox_summary: str
    current_ticket_id: Optional[str] = None
    current_ticket_view: str = ""
    last_event: str = ""
    progress_score: float = Field(default=0.0, ge=0.0, le=1.0)
    remaining_steps: int = Field(default=0, ge=0)
    recent_actions: List[str] = Field(default_factory=list)
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    penalty_breakdown: Dict[str, float] = Field(default_factory=dict)
    guidance: str = (
        "Inspect relevant tickets, update routing fields, draft one clear reply, "
        "and call submit when you are done."
    )


class SupportOpsState(State):
    task_id: str
    task_title: str
    selected_ticket_id: Optional[str] = None
    tickets: List[TicketSnapshot] = Field(default_factory=list)
    viewed_ticket_ids: List[str] = Field(default_factory=list)
    action_history: List[str] = Field(default_factory=list)
    current_score: float = Field(default=0.0, ge=0.0, le=1.0)
    cumulative_reward: float = Field(default=0.0, ge=0.0)
    invalid_action_count: int = Field(default=0, ge=0)
    no_progress_count: int = Field(default=0, ge=0)
    submitted: bool = False
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    penalty_breakdown: Dict[str, float] = Field(default_factory=dict)
    unmet_requirements: List[str] = Field(default_factory=list)
    task_metadata: Dict[str, str] = Field(default_factory=dict)
