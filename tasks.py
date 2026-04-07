from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

try:
    from .models import Priority, SupportTeam, TicketSnapshot, TicketStatus
except ImportError:
    from models import Priority, SupportTeam, TicketSnapshot, TicketStatus


@dataclass(frozen=True)
class ReplyRequirement:
    name: str
    alternatives: Sequence[str]
    weight: float


@dataclass(frozen=True)
class TicketExpectation:
    priority: Priority
    team: SupportTeam
    status: TicketStatus
    required_tags: Sequence[str]
    reply_requirements: Sequence[ReplyRequirement]
    forbidden_reply_phrases: Sequence[str]


@dataclass(frozen=True)
class MergeExpectation:
    source_ticket_id: str
    target_ticket_id: str


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    difficulty: str
    title: str
    objective: str
    intro: str
    max_steps: int
    ideal_steps: int
    tickets: Sequence[TicketSnapshot]
    required_views: Sequence[str]
    expectations: Dict[str, TicketExpectation]
    merge_expectations: Sequence[MergeExpectation]
    irrelevant_ticket_ids: Sequence[str]
    policy_hints: Sequence[str]


def _tk(tid, name, co, tier, subj, body):
    return TicketSnapshot(
        ticket_id=tid, customer_name=name, company=co, tier=tier,
        subject=subj, body=body, status="open", priority="normal",
        assigned_team="general", tags=[], reply_draft="", merged_into=None,
    )


TASKS: Dict[str, TaskSpec] = {
    "easy_vip_sso": TaskSpec(
        task_id="easy_vip_sso",
        difficulty="easy",
        title="VIP lockout during an SSO migration",
        objective=(
            "Triage the urgent enterprise access issue. Inspect the correct ticket, "
            "route it to the right team, add the important tags, send a concise "
            "customer reply, and submit the case."
        ),
        intro=(
            "You are the first-line operator in a SaaS support queue. One enterprise "
            "customer appears blocked during a time-sensitive event."
        ),
        max_steps=8,
        ideal_steps=6,
        tickets=[
            _tk(
                "E-1001", "Marta Silva", "Northstar Capital", "Enterprise",
                "Board meeting in 2 hours and our admin is locked out",
                "We completed your SSO migration this morning. I can no longer "
                "reach the admin dashboard and keep seeing 'SAML assertion expired'. "
                "I need access before our board deck review today.",
            ),
            _tk(
                "E-1002", "Leo Carter", "Foundry Labs", "Pro",
                "Dark mode request",
                "Could you add a dark mode toggle to the analytics page?",
            ),
            _tk(
                "E-1003", "Sanjay Mehta", "Northstar Capital", "Enterprise",
                "Need a copy of last month usage export",
                "When you have time, can someone send us March workspace usage "
                "numbers for finance reconciliation?",
            ),
        ],
        required_views=["E-1001"],
        expectations={
            "E-1001": TicketExpectation(
                priority="urgent",
                team="access",
                status="escalated",
                required_tags=("vip_customer", "sso"),
                reply_requirements=(
                    ReplyRequirement("empathy", ("sorry", "apolog", "understand"), 0.25),
                    ReplyRequirement(
                        "escalation",
                        ("escalated", "specialist", "identity team", "access team"),
                        0.30,
                    ),
                    ReplyRequirement(
                        "time_commitment",
                        ("15 minutes", "15-minute", "shortly", "right away"),
                        0.20,
                    ),
                    ReplyRequirement(
                        "workaround",
                        ("temporary access", "backup admin", "workaround", "alternate admin"),
                        0.25,
                    ),
                ),
                forbidden_reply_phrases=("send us your password", "share your password"),
            )
        },
        merge_expectations=(),
        irrelevant_ticket_ids=("E-1002", "E-1003"),
        policy_hints=(
            "Enterprise access incidents blocking executive meetings should be routed urgently.",
            "Identity or SSO failures belong with the access team, not general support.",
        ),
    ),
    "medium_refund_duplicate": TaskSpec(
        task_id="medium_refund_duplicate",
        difficulty="medium",
        title="Duplicate billing case with a duplicate ticket",
        objective=(
            "Handle the duplicate-charge case. Inspect the relevant tickets, merge the "
            "duplicate, assign the correct team, set the right urgency, send one clear "
            "reply, and submit the queue update."
        ),
        intro=(
            "Several tickets are in the inbox. Only one customer is dealing with an "
            "active billing failure and duplicate outreach from their finance team."
        ),
        max_steps=10,
        ideal_steps=7,
        tickets=[
            _tk(
                "M-2001", "Ava Bennett", "Cinder Health", "Pro",
                "Feature request: recurring exports",
                "We would love recurring CSV exports from the dashboard.",
            ),
            _tk(
                "M-2002", "Nadia Rahman", "Harbor Retail", "Business",
                "Charged twice after our seat downgrade",
                "We downgraded from 22 to 14 seats yesterday and the card was "
                "charged twice. Invoice 18391 is attached. Please fix this today.",
            ),
            _tk(
                "M-2003", "Tom Liu", "Branchline AI", "Starter",
                "CSV import skipped one row",
                "The import worked except for one row with a malformed phone number.",
            ),
            _tk(
                "M-2004", "Priya Rahman", "Harbor Retail", "Business",
                "Finance follow-up on duplicate payment",
                "I am following up from finance on Nadia's note. The duplicate "
                "charge is still pending in our ledger. Same invoice: 18391.",
            ),
            _tk(
                "M-2005", "Noah Adams", "Blue Harbor Cafe", "Business",
                "Can you upgrade us to annual billing next month?",
                "We want to switch from monthly to annual billing at the next renewal date.",
            ),
        ],
        required_views=["M-2002", "M-2004"],
        expectations={
            "M-2002": TicketExpectation(
                priority="high",
                team="billing",
                status="escalated",
                required_tags=("refund", "duplicate"),
                reply_requirements=(
                    ReplyRequirement(
                        "acknowledgement",
                        ("sorry", "apolog", "thanks for flagging"),
                        0.20,
                    ),
                    ReplyRequirement(
                        "refund_review",
                        ("refund review", "billing review", "3 business days", "3-day"),
                        0.35,
                    ),
                    ReplyRequirement(
                        "duplicate_consolidation",
                        ("merged", "consolidated", "single ticket", "one thread"),
                        0.25,
                    ),
                    ReplyRequirement(
                        "ownership",
                        ("billing team", "billing specialist", "finance operations"),
                        0.20,
                    ),
                ),
                forbidden_reply_phrases=("close your bank dispute", "ignore the second charge"),
            )
        },
        merge_expectations=(MergeExpectation("M-2004", "M-2002"),),
        irrelevant_ticket_ids=("M-2001", "M-2003", "M-2005"),
        policy_hints=(
            "Duplicate billing reports from the same company should be "
            "consolidated into one owning ticket.",
            "Billing disputes should be escalated without promising an immediate refund.",
        ),
    ),
    "hard_security_phishing": TaskSpec(
        task_id="hard_security_phishing",
        difficulty="hard",
        title="Potential phishing incident and credential exposure",
        objective=(
            "Triage the highest-risk case without overstating certainty. Inspect the "
            "relevant tickets, merge the related follow-up, route to security, tag it "
            "appropriately, draft a safe incident response, and submit."
        ),
        intro=(
            "A customer may have interacted with a spoofed support email. The response "
            "must be urgent, security-aware, and precise."
        ),
        max_steps=10,
        ideal_steps=7,
        tickets=[
            _tk(
                "H-3001", "Olivia Chen", "SummitGrid", "Enterprise",
                "Did your support team ask us for our admin password?",
                "One of our admins received an email claiming to be OpenEnv support "
                "asking for an admin password and MFA code to 'revalidate our tenant'. "
                "Is this legitimate?",
            ),
            _tk(
                "H-3002", "Olivia Chen", "SummitGrid", "Enterprise",
                "We clicked the link before noticing the sender domain was wrong",
                "Separate follow-up: one teammate opened the link before we noticed "
                "the sender domain was suspicious. No credentials were intentionally "
                "shared, but should we rotate tokens and active sessions?",
            ),
            _tk(
                "H-3003", "Ethan Park", "Mapletree", "Pro",
                "Dashboard loading slowly today",
                "The analytics dashboard is a little slower than normal after lunch.",
            ),
            _tk(
                "H-3004", "Rina Khatri", "SummitGrid", "Enterprise",
                "Need your latest SOC 2 report",
                "Our procurement team needs your latest SOC 2 report before renewal.",
            ),
        ],
        required_views=["H-3001", "H-3002"],
        expectations={
            "H-3001": TicketExpectation(
                priority="urgent",
                team="security",
                status="escalated",
                required_tags=("security", "phishing"),
                reply_requirements=(
                    ReplyRequirement(
                        "not_legitimate",
                        ("not legitimate", "not from us", "not sent by our team", "spoofed"),
                        0.20,
                    ),
                    ReplyRequirement(
                        "password_safety",
                        ("do not share", "never ask for your password", "do not send passwords"),
                        0.20,
                    ),
                    ReplyRequirement(
                        "containment",
                        ("reset password", "rotate tokens", "revoke sessions", "rotate credentials"),
                        0.25,
                    ),
                    ReplyRequirement(
                        "security_escalation",
                        ("security team", "incident team", "security specialist"),
                        0.20,
                    ),
                    ReplyRequirement(
                        "evidence_preservation",
                        ("headers", "screenshot", "forward the email", "preserve"),
                        0.15,
                    ),
                    ReplyRequirement(
                        "log_review",
                        ("audit logs", "sign-in logs", "review logs", "activity logs"),
                        0.10,
                    ),
                ),
                forbidden_reply_phrases=(
                    "send us your password",
                    "share your mfa code",
                    "we confirmed a breach",
                    "malware infection confirmed",
                ),
            )
        },
        merge_expectations=(MergeExpectation("H-3002", "H-3001"),),
        irrelevant_ticket_ids=("H-3003", "H-3004"),
        policy_hints=(
            "Possible phishing and credential exposure should be routed to security immediately.",
            "Do not claim a confirmed breach unless the evidence already establishes one.",
            "Ask the customer to preserve evidence and rotate credentials or sessions.",
        ),
    ),
    "expert_compliance_trap": TaskSpec(
        task_id="expert_compliance_trap",
        difficulty="expert",
        title="GDPR deletion disguised as a billing cancellation",
        objective=(
            "Identify the real compliance obligation hidden inside what looks like "
            "a routine churn case. Two tickets from the same company must be handled "
            "together: one is a straightforward cancellation, the other is a legally "
            "binding data-deletion request under GDPR Article 17. Route each to the "
            "correct team, merge the related follow-up, draft a reply that addresses "
            "both the cancellation and the legal obligation without overpromising "
            "timelines, and submit."
        ),
        intro=(
            "Your inbox has several tickets. At first glance most look like billing "
            "or product issues. Read carefully — one customer's request carries legal "
            "weight that changes how you must handle the entire case."
        ),
        max_steps=16,
        ideal_steps=12,
        tickets=[
            _tk(
                "X-4001", "Freya Lindqvist", "Velora GmbH", "Enterprise",
                "We are cancelling — please confirm final invoice",
                "Hi, we have decided not to renew. Can you confirm the final "
                "invoice amount and when billing stops? Thanks.",
            ),
            _tk(
                "X-4002", "Freya Lindqvist", "Velora GmbH", "Enterprise",
                "GDPR Art. 17 — formal request to erase all our data",
                "Per GDPR Article 17, we formally request erasure of all personal "
                "data associated with our organisation within the statutory 30-day "
                "window. This includes backups and third-party sub-processors. "
                "Please confirm receipt and provide a written timeline.",
            ),
            _tk(
                "X-4003", "Marco Bianchi", "Velora GmbH", "Enterprise",
                "Follow-up from legal on the data deletion",
                "I am outside counsel for Velora. Freya asked me to ensure the "
                "data-erasure request is logged as a formal legal demand. Please "
                "route to your DPO or compliance officer.",
            ),
            _tk(
                "X-4004", "Aisha Okafor", "Pinegrove Analytics", "Business",
                "We think we were overcharged last quarter",
                "Our usage report says 14 seats but the invoice shows 18. Could "
                "someone look at invoice 29104?",
            ),
            _tk(
                "X-4005", "Jordan Kim", "Rivercrest Labs", "Pro",
                "API rate limit seems too low",
                "We keep hitting 429s on the /export endpoint. Can the limit be raised?",
            ),
            _tk(
                "X-4006", "Lukas Petrov", "Velora GmbH", "Enterprise",
                "Can we still download our dashboards before the account closes?",
                "Before everything is wiped, we need to export our saved dashboards "
                "and report templates. Is there a self-serve option or do we need help?",
            ),
        ],
        required_views=["X-4001", "X-4002", "X-4003"],
        expectations={
            "X-4002": TicketExpectation(
                priority="urgent",
                team="compliance",
                status="escalated",
                required_tags=("gdpr", "data_deletion", "legal"),
                reply_requirements=(
                    ReplyRequirement(
                        "ack_legal",
                        ("formal request", "data erasure", "article 17", "gdpr"),
                        0.20,
                    ),
                    ReplyRequirement(
                        "timeline",
                        ("30 day", "30-day", "statutory", "calendar days"),
                        0.15,
                    ),
                    ReplyRequirement(
                        "dpo_route",
                        ("data protection", "compliance team", "dpo", "compliance officer"),
                        0.20,
                    ),
                    ReplyRequirement(
                        "sub_processors",
                        ("sub-processor", "third-party", "third party", "downstream"),
                        0.15,
                    ),
                    ReplyRequirement(
                        "written_confirm",
                        ("written confirmation", "confirm in writing", "formal acknowledgement",
                         "confirmation email"),
                        0.15,
                    ),
                    ReplyRequirement(
                        "no_overpromise",
                        ("review", "assess", "scope", "verify"),
                        0.15,
                    ),
                ),
                forbidden_reply_phrases=(
                    "all data has been deleted",
                    "deletion is complete",
                    "we have erased everything",
                    "your data is already gone",
                    "we don't store personal data",
                ),
            ),
            "X-4001": TicketExpectation(
                priority="high",
                team="billing",
                status="pending_customer",
                required_tags=("churn",),
                reply_requirements=(
                    ReplyRequirement(
                        "cancellation_ack",
                        ("cancellation", "non-renewal", "final invoice", "last invoice"),
                        0.40,
                    ),
                    ReplyRequirement(
                        "export_reminder",
                        ("export", "download", "retrieve your data", "before account closes"),
                        0.30,
                    ),
                    ReplyRequirement(
                        "billing_stop",
                        ("billing will stop", "no further charges", "end of billing",
                         "billing cycle"),
                        0.30,
                    ),
                ),
                forbidden_reply_phrases=(
                    "we have deleted your data",
                    "sorry to see you go but your data is erased",
                ),
            ),
        },
        merge_expectations=(MergeExpectation("X-4003", "X-4002"),),
        irrelevant_ticket_ids=("X-4004", "X-4005"),
        policy_hints=(
            "GDPR Article 17 requests are legal obligations, not feature requests.",
            "A cancellation ticket and a data-deletion ticket from the same company "
            "are related but require different teams and different handling.",
            "Do not confirm data has been deleted before the compliance team reviews.",
            "The legal counsel follow-up should be merged into the GDPR ticket, not "
            "the billing ticket.",
            "X-4006 is from the same company but is not irrelevant — it relates to the "
            "cancellation workflow. Do not penalize or modify it unnecessarily.",
        ),
    ),
}

TASK_IDS: List[str] = list(TASKS.keys())


def list_task_specs() -> Iterable[TaskSpec]:
    for tid in TASK_IDS:
        yield TASKS[tid]


def get_task_spec(task_id: Optional[str]) -> TaskSpec:
    if task_id is None:
        return TASKS[TASK_IDS[0]]
    try:
        return TASKS[task_id]
    except KeyError as exc:
        raise ValueError(f"Unknown task_id: {task_id}") from exc
