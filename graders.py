from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

try:
    from .models import CaseRecord, SupportOpsState
    from .tasks import ExpectedCaseState, GroundingRule, TaskSpec, TaskExpectation
except ImportError:
    from models import CaseRecord, SupportOpsState
    from tasks import ExpectedCaseState, GroundingRule, TaskSpec, TaskExpectation


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _has(text: str, needles: Sequence[str]) -> bool:
    haystack = _norm(text)
    return any(_norm(needle) in haystack for needle in needles)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


_OPEN_LO = 1e-3
_OPEN_HI = 1.0 - 1e-3


def _open01(score: float) -> float:
    c = _clamp(score)
    if c <= 0.0:
        return _OPEN_LO
    if c >= 1.0:
        return _OPEN_HI
    return max(_OPEN_LO, min(_OPEN_HI, c))


@dataclass(frozen=True)
class GradeResult:
    score: float
    components: Dict[str, float]
    penalties: Dict[str, float]
    unmet_requirements: List[str]


def _case_map(state: SupportOpsState) -> Dict[str, CaseRecord]:
    return {case.case_id: case for case in state.cases}


def _tool_name_history(state: SupportOpsState) -> List[str]:
    return [entry.split("(", 1)[0] for entry in state.tool_history]


def _reply_score(reply_text: str, expectation: TaskExpectation) -> tuple[float, List[str]]:
    if not reply_text.strip():
        return 0.0, ["missing reply draft"]

    total_weight = sum(item.weight for item in expectation.reply_requirements)
    matched_weight = 0.0
    unmet: List[str] = []
    for item in expectation.reply_requirements:
        if _has(reply_text, item.alternatives):
            matched_weight += item.weight
        else:
            unmet.append(f"reply missing {item.name}")

    return (_clamp(matched_weight / total_weight if total_weight else 1.0), unmet)


def _routing_score(case: CaseRecord, expected: ExpectedCaseState) -> tuple[float, List[str]]:
    score = 0.0
    unmet: List[str] = []

    if case.priority == expected.priority:
        score += 0.25
    else:
        unmet.append(f"{case.case_id} priority should be {expected.priority}")

    if case.assigned_team == expected.team:
        score += 0.25
    else:
        unmet.append(f"{case.case_id} team should be {expected.team}")

    if case.status == expected.status:
        score += 0.20
    else:
        unmet.append(f"{case.case_id} status should be {expected.status}")

    matched_tags = sum(1 for tag in expected.required_tags if tag in case.tags)
    score += 0.20 * (matched_tags / max(1, len(expected.required_tags)))
    for tag in expected.required_tags:
        if tag not in case.tags:
            unmet.append(f"{case.case_id} missing tag {tag}")

    if expected.merged_into is None:
        score += 0.10 if case.merged_into is None else 0.0
        if case.merged_into is not None:
            unmet.append(f"{case.case_id} should not be merged")
    else:
        if case.merged_into == expected.merged_into:
            score += 0.10
        else:
            unmet.append(f"{case.case_id} should merge into {expected.merged_into}")

    return score, unmet


def _grounding_score(reply_text: str, seen_facts: Sequence[str], rules: Sequence[GroundingRule]) -> tuple[float, List[str]]:
    if not reply_text.strip():
        return 0.0, ["reply missing grounded claims"]

    missing: List[str] = []
    total = 0
    matched = 0
    reply = _norm(reply_text)
    for rule in rules:
        if _norm(rule.phrase) in reply:
            total += 1
            if rule.fact_id in seen_facts:
                matched += 1
            else:
                missing.append(f"reply claim '{rule.phrase}' was not surfaced")
    if total == 0:
        return 1.0, []
    return matched / total, missing


def _investigation_score(task: TaskSpec, state: SupportOpsState) -> tuple[float, List[str]]:
    seen_facts = set(state.seen_facts)
    tool_history = set(_tool_name_history(state))

    tool_fraction = sum(tool in tool_history for tool in task.expectation.required_tools) / max(
        1, len(task.expectation.required_tools)
    )
    fact_fraction = sum(fid in seen_facts for fid in task.expectation.required_fact_ids) / max(
        1, len(task.expectation.required_fact_ids)
    )

    unmet = []
    for tool in task.expectation.required_tools:
        if tool not in tool_history:
            unmet.append(f"tool {tool} was never used")
    for fact_id in task.expectation.required_fact_ids:
        if fact_id not in seen_facts:
            unmet.append(f"fact {fact_id} was never surfaced")

    return (0.45 * tool_fraction + 0.55 * fact_fraction, unmet)


def _irrelevant_penalty(task: TaskSpec, state: SupportOpsState, case_map: Dict[str, CaseRecord]) -> tuple[float, List[str]]:
    touched = 0
    unmet: List[str] = []
    for case_id in task.expectation.irrelevant_case_ids:
        case = case_map.get(case_id)
        if case is None:
            continue
        if (
            case.priority != "normal"
            or case.assigned_team != "general"
            or case.status != "open"
            or case.tags
            or case.merged_into is not None
            or case.reply_draft
            or case.note_log
        ):
            touched += 1
            unmet.append(f"irrelevant case {case_id} was modified")
    return min(0.15, touched * 0.08), unmet


def _repeat_penalty(state: SupportOpsState) -> float:
    repeats = max(0, len(state.tool_history) - len(set(state.tool_history)))
    return min(0.12, repeats * 0.03)


def grade_state(state: SupportOpsState, task: TaskSpec) -> GradeResult:
    case_map = _case_map(state)
    unmet: List[str] = []

    investigation, inv_unmet = _investigation_score(task, state)
    unmet.extend(inv_unmet)

    routing_scores = []
    for case_id, expected in task.expectation.expected_cases.items():
        case = case_map[case_id]
        route_score, route_unmet = _routing_score(case, expected)
        routing_scores.append(route_score)
        unmet.extend(route_unmet)

    primary_case = case_map[task.expectation.primary_case_id]
    reply_quality, reply_unmet = _reply_score(primary_case.reply_draft, task.expectation)
    unmet.extend(reply_unmet)

    groundedness, grounded_unmet = _grounding_score(
        primary_case.reply_draft,
        state.seen_facts,
        task.expectation.grounding_rules,
    )
    unmet.extend(grounded_unmet)

    submission = 1.0 if state.submitted else 0.0

    forbidden_hits = sum(
        phrase.lower() in _norm(primary_case.reply_draft)
        for phrase in task.expectation.forbidden_reply_phrases
    )
    unsupported_claim_penalty = min(0.35, len(grounded_unmet) * 0.12)
    unsafe_reply_penalty = min(0.35, forbidden_hits * 0.2)
    repeat_penalty = _repeat_penalty(state)
    irrelevant_penalty, irr_unmet = _irrelevant_penalty(task, state, case_map)
    unmet.extend(irr_unmet)
    invalid_action_penalty = min(0.20, state.invalid_action_count * 0.08)
    no_progress_penalty = min(0.10, state.no_progress_count * 0.03)

    disallowed_tool_uses = sum(
        tool in _tool_name_history(state) for tool in task.expectation.disallowed_tools
    )
    disallowed_tool_penalty = min(0.25, disallowed_tool_uses * 0.15)
    for tool in task.expectation.disallowed_tools:
        if tool in _tool_name_history(state):
            unmet.append(f"disallowed tool {tool} was used")

    if state.step_count > task.ideal_steps:
        overflow = state.step_count - task.ideal_steps
        budget = max(1, task.max_steps - task.ideal_steps)
        efficiency_penalty = min(0.12, 0.12 * (overflow / budget))
    else:
        efficiency_penalty = 0.0

    components = {
        "investigation": round(investigation, 4),
        "routing": round(sum(routing_scores) / max(1, len(routing_scores)), 4),
        "reply_quality": round(reply_quality, 4),
        "groundedness": round(groundedness, 4),
        "submission": round(submission, 4),
    }

    weighted = (
        0.28 * components["investigation"]
        + 0.32 * components["routing"]
        + 0.20 * components["reply_quality"]
        + 0.12 * components["groundedness"]
        + 0.08 * components["submission"]
    )
    penalties = {
        "unsupported_claim_penalty": round(unsupported_claim_penalty, 4),
        "unsafe_reply_penalty": round(unsafe_reply_penalty, 4),
        "repeat_penalty": round(repeat_penalty, 4),
        "irrelevant_penalty": round(irrelevant_penalty, 4),
        "invalid_action_penalty": round(invalid_action_penalty, 4),
        "no_progress_penalty": round(no_progress_penalty, 4),
        "efficiency_penalty": round(efficiency_penalty, 4),
        "disallowed_tool_penalty": round(disallowed_tool_penalty, 4),
    }
    total_penalty = sum(penalties.values())
    score = _open01(weighted - total_penalty)

    return GradeResult(
        score=round(score, 4),
        components=components,
        penalties=penalties,
        unmet_requirements=sorted(set(unmet)),
    )

