from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..graders import GradeResult, grade_state
    from ..models import SupportOpsAction, SupportOpsObservation, SupportOpsState, TicketSnapshot
    from ..tasks import TaskSpec, get_task_spec
except ImportError:
    from graders import GradeResult, grade_state
    from models import SupportOpsAction, SupportOpsObservation, SupportOpsState, TicketSnapshot
    from tasks import TaskSpec, get_task_spec


class SupportOpsEnvironment(Environment[SupportOpsAction, SupportOpsObservation, SupportOpsState]):

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._task: TaskSpec = get_task_spec(None)
        self._st: SupportOpsState = self._init_state(self._task, eid=str(uuid4()))
        self._done = False

    def _init_state(self, task: TaskSpec, eid: str) -> SupportOpsState:
        tix = [deepcopy(t) for t in task.tickets]
        return SupportOpsState(
            episode_id=eid, step_count=0, task_id=task.task_id,
            task_title=task.title, selected_ticket_id=None, tickets=tix,
            viewed_ticket_ids=[], action_history=[], current_score=0.0,
            cumulative_reward=0.0, invalid_action_count=0, no_progress_count=0,
            submitted=False, score_breakdown={}, penalty_breakdown={},
            unmet_requirements=[],
            task_metadata={
                "difficulty": task.difficulty,
                "objective": task.objective,
                "policy_hints": " | ".join(task.policy_hints),
            },
        )

    def reset(
        self, seed: Optional[int] = None, episode_id: Optional[str] = None,
        task_id: Optional[str] = None, **_: object,
    ) -> SupportOpsObservation:
        del seed
        self._task = get_task_spec(task_id)
        self._st = self._init_state(self._task, eid=episode_id or str(uuid4()))
        self._done = False
        return self._obs(last_event=self._task.intro, reward=0.0, done=False)

    def _tmap(self) -> Dict[str, TicketSnapshot]:
        return {t.ticket_id: t for t in self._st.tickets}

    def _sel(self) -> Optional[TicketSnapshot]:
        if self._st.selected_ticket_id is None:
            return None
        return self._tmap().get(self._st.selected_ticket_id)

    def _log(self, entry: str) -> None:
        self._st.action_history.append(entry)
        self._st.action_history = self._st.action_history[-12:]

    def _grade(self) -> GradeResult:
        g = grade_state(self._st, self._task)
        self._st.current_score = g.score
        self._st.score_breakdown = g.components
        self._st.penalty_breakdown = g.penalties
        self._st.unmet_requirements = g.unmet_requirements
        return g

    def _inbox(self) -> str:
        lines = []
        for t in self._st.tickets:
            mg = f" -> merged into {t.merged_into}" if t.merged_into else ""
            lines.append(
                f"{t.ticket_id} | {t.company} ({t.tier}) | "
                f"priority={t.priority} | team={t.assigned_team} | "
                f"status={t.status}{mg} | subject={t.subject}"
            )
        return "\n".join(lines)

    def _tv(self, t: Optional[TicketSnapshot]) -> str:
        if t is None:
            return "No ticket is currently selected. Use view_ticket first."
        return (
            f"Ticket: {t.ticket_id}\n"
            f"Customer: {t.customer_name} at {t.company} ({t.tier})\n"
            f"Subject: {t.subject}\n"
            f"Body: {t.body}\n"
            f"Current priority: {t.priority}\n"
            f"Assigned team: {t.assigned_team}\n"
            f"Status: {t.status}\n"
            f"Tags: {', '.join(t.tags) if t.tags else 'none'}\n"
            f"Reply draft: {t.reply_draft or 'none'}"
        )

    def _obs(self, last_event: str, reward: float, done: bool) -> SupportOpsObservation:
        g = self._grade()
        sel = self._sel()
        return SupportOpsObservation(
            task_id=self._task.task_id,
            task_title=self._task.title,
            objective=self._task.objective,
            inbox_summary=self._inbox(),
            current_ticket_id=sel.ticket_id if sel else None,
            current_ticket_view=self._tv(sel),
            last_event=last_event,
            progress_score=g.score,
            remaining_steps=max(0, self._task.max_steps - self._st.step_count),
            recent_actions=self._st.action_history[-5:],
            score_breakdown=g.components,
            penalty_breakdown=g.penalties,
            reward=round(reward, 4),
            done=done,
            metadata={
                "difficulty": self._task.difficulty,
                "submitted": self._st.submitted,
                "unmet_requirements": g.unmet_requirements,
            },
        )

    def _req(self, act: SupportOpsAction) -> TicketSnapshot:
        if not act.ticket_id:
            raise ValueError(f"{act.action_type} requires ticket_id")
        try:
            return self._tmap()[act.ticket_id]
        except KeyError as exc:
            raise ValueError(f"Unknown ticket_id: {act.ticket_id}") from exc

    def _do_view(self, act: SupportOpsAction) -> str:
        t = self._req(act)
        self._st.selected_ticket_id = t.ticket_id
        if t.ticket_id not in self._st.viewed_ticket_ids:
            self._st.viewed_ticket_ids.append(t.ticket_id)
        return f"Opened ticket {t.ticket_id}."

    def _do_priority(self, act: SupportOpsAction) -> str:
        t = self._req(act)
        if act.priority is None:
            raise ValueError("set_priority requires priority")
        if t.priority == act.priority:
            raise ValueError(f"{t.ticket_id} already has priority {act.priority}")
        t.priority = act.priority
        return f"Set priority for {t.ticket_id} to {act.priority}."

    def _do_team(self, act: SupportOpsAction) -> str:
        t = self._req(act)
        if act.team is None:
            raise ValueError("assign_team requires team")
        if t.assigned_team == act.team:
            raise ValueError(f"{t.ticket_id} is already assigned to {act.team}")
        t.assigned_team = act.team
        return f"Assigned {t.ticket_id} to {act.team}."

    def _do_tags(self, act: SupportOpsAction) -> str:
        t = self._req(act)
        if not act.tags:
            raise ValueError("add_tags requires at least one tag")
        added: List[str] = []
        for tag in act.tags:
            norm = tag.strip().lower().replace(" ", "_")
            if norm and norm not in t.tags:
                t.tags.append(norm)
                added.append(norm)
        if not added:
            raise ValueError(f"{t.ticket_id} already has all requested tags")
        t.tags.sort()
        return f"Added tags to {t.ticket_id}: {', '.join(act.tags)}."

    def _do_status(self, act: SupportOpsAction) -> str:
        t = self._req(act)
        if act.status is None:
            raise ValueError("set_status requires status")
        if t.status == act.status:
            raise ValueError(f"{t.ticket_id} is already {act.status}")
        t.status = act.status
        return f"Set status for {t.ticket_id} to {act.status}."

    def _do_reply(self, act: SupportOpsAction) -> str:
        t = self._req(act)
        if not act.reply or not act.reply.strip():
            raise ValueError("draft_reply requires reply text")
        txt = act.reply.strip()
        if t.reply_draft == txt:
            raise ValueError(f"{t.ticket_id} already has the same reply draft")
        t.reply_draft = txt
        return f"Drafted customer reply for {t.ticket_id}."

    def _do_merge(self, act: SupportOpsAction) -> str:
        src = self._req(act)
        if not act.target_ticket_id:
            raise ValueError("merge_ticket requires target_ticket_id")
        tgt = self._tmap().get(act.target_ticket_id)
        if tgt is None:
            raise ValueError(f"Unknown target_ticket_id: {act.target_ticket_id}")
        if src.ticket_id == tgt.ticket_id:
            raise ValueError("Cannot merge a ticket into itself")
        if src.merged_into is not None:
            raise ValueError(f"{src.ticket_id} has already been merged")
        src.merged_into = tgt.ticket_id
        src.status = "closed"
        self._st.selected_ticket_id = tgt.ticket_id
        return f"Merged {src.ticket_id} into {tgt.ticket_id}."

    def _do_submit(self, act: SupportOpsAction) -> str:
        del act
        self._st.submitted = True
        self._done = True
        return "Submitted the queue update for grading."

    _HANDLERS = {
        "view_ticket": _do_view,
        "set_priority": _do_priority,
        "assign_team": _do_team,
        "add_tags": _do_tags,
        "set_status": _do_status,
        "draft_reply": _do_reply,
        "merge_ticket": _do_merge,
        "submit": _do_submit,
    }

    def step(
        self, action: SupportOpsAction,
        timeout_s: Optional[float] = None, **_: object,
    ) -> SupportOpsObservation:
        del timeout_s
        if self._done:
            return self._obs(
                last_event="Episode already finished. Call reset() to start a new task.",
                reward=0.0, done=True,
            )

        before = self._grade()
        evt = ""
        pen = 0.0
        invalid = False
        try:
            handler = self._HANDLERS.get(action.action_type)
            if handler is None:
                raise ValueError(f"Unsupported action_type: {action.action_type}")
            evt = handler(self, action)
        except ValueError as exc:
            pen = 0.05
            invalid = True
            self._st.invalid_action_count += 1
            evt = f"Invalid action: {exc}"

        hv = (
            f"{action.action_type}:{action.ticket_id or '-'}:{action.target_ticket_id or '-'}:"
            f"{action.priority or action.team or action.status or ','.join(action.tags) or 'n/a'}"
        )
        self._log(hv)
        self._st.step_count += 1

        if self._st.step_count >= self._task.max_steps:
            self._done = True
            if not self._st.submitted:
                evt += " Maximum step budget reached."

        after = self._grade()
        sg = max(0.0, after.score - before.score)
        cg = sum(max(0.0, v - before.components.get(k, 0.0)) for k, v in after.components.items())
        rw = min(1.0, sg + 0.15 * cg)
        if invalid:
            rw = max(0.0, rw - pen)
        if after.score <= before.score and not invalid:
            self._st.no_progress_count += 1
            rw = max(0.0, rw - 0.01)
        rw = round(rw, 4)
        self._st.cumulative_reward += rw

        return self._obs(last_event=evt, reward=rw, done=self._done)

    @property
    def state(self) -> SupportOpsState:
        self._grade()
        return self._st

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="SupportOpsEnvironment",
            description="A support inbox triage benchmark with deterministic graders.",
            version="1.0.0",
            author="Codex",
        )
