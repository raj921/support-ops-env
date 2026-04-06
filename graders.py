from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

try:
    from .models import SupportOpsState, TicketSnapshot
    from .tasks import MergeExpectation, TaskSpec, TicketExpectation
except ImportError:
    from models import SupportOpsState, TicketSnapshot
    from tasks import MergeExpectation, TaskSpec, TicketExpectation


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _has(text: str, needles: Sequence[str]) -> bool:
    h = _norm(text)
    return any(n.lower() in h for n in needles)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


@dataclass(frozen=True)
class GradeResult:
    score: float
    components: Dict[str, float]
    penalties: Dict[str, float]
    unmet_requirements: List[str]


def _reply_score(reply: str, exp: TicketExpectation) -> tuple[float, List[str], float]:
    if not reply.strip():
        return 0.0, ["missing customer reply"], 0.0

    tw = sum(r.weight for r in exp.reply_requirements)
    mw = 0.0
    miss: List[str] = []
    for r in exp.reply_requirements:
        if _has(reply, r.alternatives):
            mw += r.weight
        else:
            miss.append(f"reply missing {r.name}")

    hits = sum(1 for p in exp.forbidden_reply_phrases if p.lower() in _norm(reply))
    pen = min(0.5, hits * 0.25)
    return _clamp(mw / tw if tw else 1.0), miss, pen


def _tkt_map(st: SupportOpsState) -> Dict[str, TicketSnapshot]:
    return {t.ticket_id: t for t in st.tickets}


def _field_score(t: TicketSnapshot, exp: TicketExpectation) -> tuple[float, List[str]]:
    s = 0.0
    um: List[str] = []

    if t.priority == exp.priority:
        s += 0.30
    else:
        um.append(f"{t.ticket_id} priority should be {exp.priority}")

    if t.assigned_team == exp.team:
        s += 0.30
    else:
        um.append(f"{t.ticket_id} team should be {exp.team}")

    if t.status == exp.status:
        s += 0.20
    else:
        um.append(f"{t.ticket_id} status should be {exp.status}")

    matched = sum(1 for tag in exp.required_tags if tag in t.tags)
    s += 0.20 * (matched / len(exp.required_tags))
    for tag in exp.required_tags:
        if tag not in t.tags:
            um.append(f"{t.ticket_id} missing tag {tag}")

    return s, um


def _merge_score(
    tm: Dict[str, TicketSnapshot], merges: Sequence[MergeExpectation]
) -> tuple[float, List[str]]:
    if not merges:
        return 1.0, []
    ok = 0
    um: List[str] = []
    for m in merges:
        src = tm.get(m.source_ticket_id)
        if src and src.merged_into == m.target_ticket_id:
            ok += 1
        else:
            um.append(f"{m.source_ticket_id} should be merged into {m.target_ticket_id}")
    return ok / len(merges), um


def _wrong_merge_pen(
    st: SupportOpsState, expected: Sequence[MergeExpectation]
) -> tuple[float, List[str]]:
    exp_pairs = {(m.source_ticket_id, m.target_ticket_id) for m in expected}
    exp_srcs = {m.source_ticket_id for m in expected}
    bad = 0
    um: List[str] = []
    for t in st.tickets:
        if t.merged_into is None:
            continue
        pair = (t.ticket_id, t.merged_into)
        if pair not in exp_pairs:
            bad += 1
            um.append(f"unexpected merge {t.ticket_id} -> {t.merged_into}")
        elif t.ticket_id not in exp_srcs:
            bad += 1
            um.append(f"unexpected merge source {t.ticket_id}")
    return min(0.2, bad * 0.1), um


def grade_state(st: SupportOpsState, task: TaskSpec) -> GradeResult:
    tm = _tkt_map(st)
    um: List[str] = []

    vs = sum(tid in st.viewed_ticket_ids for tid in task.required_views) / max(1, len(task.required_views))
    for tid in task.required_views:
        if tid not in st.viewed_ticket_ids:
            um.append(f"{tid} was never viewed")

    fs: List[float] = []
    rs: List[float] = []
    rp = 0.0

    for tid, exp in task.expectations.items():
        t = tm[tid]
        fv, fu = _field_score(t, exp)
        fs.append(fv)
        um.extend(fu)

        rv, ru, rfp = _reply_score(t.reply_draft, exp)
        rs.append(rv)
        um.extend(f"{tid}: {msg}" for msg in ru)
        rp += rfp

    ms, mu = _merge_score(tm, task.merge_expectations)
    um.extend(mu)
    wmp, wmu = _wrong_merge_pen(st, task.merge_expectations)
    um.extend(wmu)

    irr = 0
    bad_res = 0
    for t in st.tickets:
        if t.ticket_id in task.irrelevant_ticket_ids:
            if (t.priority != "normal" or t.assigned_team != "general"
                    or t.tags or t.reply_draft or t.merged_into is not None):
                irr += 1
        if t.ticket_id not in task.expectations and t.status in {"resolved", "closed"}:
            bad_res += 1

    eff_pen = 0.0
    if st.step_count > task.ideal_steps:
        overflow = st.step_count - task.ideal_steps
        budget = max(1, task.max_steps - task.ideal_steps)
        eff_pen = min(0.15, 0.15 * (overflow / budget))

    rep_pen = min(0.10, max(0, len(st.action_history) - len(set(st.action_history))) * 0.02)
    irr_pen = min(0.10, irr * 0.05)
    wr_pen = min(0.10, bad_res * 0.05)
    inv_pen = min(0.10, st.invalid_action_count * 0.03)
    np_pen = min(0.10, st.no_progress_count * 0.015)
    tot_pen = min(0.45, rp + eff_pen + rep_pen + irr_pen + wr_pen + wmp + inv_pen + np_pen)

    comp = {
        "views": vs,
        "field_alignment": sum(fs) / max(1, len(fs)),
        "reply_quality": sum(rs) / max(1, len(rs)),
        "merge_quality": ms,
        "submitted": 1.0 if st.submitted else 0.0,
    }

    if task.difficulty == "easy":
        w = 0.10 * comp["views"] + 0.55 * comp["field_alignment"] + 0.25 * comp["reply_quality"] + 0.10 * comp["submitted"]
    elif task.difficulty == "medium":
        w = 0.10 * comp["views"] + 0.35 * comp["field_alignment"] + 0.20 * comp["reply_quality"] + 0.25 * comp["merge_quality"] + 0.10 * comp["submitted"]
    else:
        w = 0.10 * comp["views"] + 0.25 * comp["field_alignment"] + 0.30 * comp["reply_quality"] + 0.20 * comp["merge_quality"] + 0.15 * comp["submitted"]

    score = _clamp(w - tot_pen)
    pens = {
        "reply_safety_penalty": round(rp, 4),
        "efficiency_penalty": round(eff_pen, 4),
        "repeat_penalty": round(rep_pen, 4),
        "irrelevant_penalty": round(irr_pen, 4),
        "wrong_resolution_penalty": round(wr_pen, 4),
        "wrong_merge_penalty": round(wmp, 4),
        "invalid_action_penalty": round(inv_pen, 4),
        "no_progress_penalty": round(np_pen, 4),
    }
    return GradeResult(
        score=round(score, 4),
        components={k: round(v, 4) for k, v in comp.items()},
        penalties=pens,
        unmet_requirements=sorted(set(um)),
    )
