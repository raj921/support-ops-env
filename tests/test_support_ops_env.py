from support_ops_env.inference import fallback_action
from support_ops_env.models import SupportOpsAction
from support_ops_env.server.support_ops_environment import SupportOpsEnvironment


def run_fallback(task_id: str) -> float:
    env = SupportOpsEnvironment()
    env.reset(task_id=task_id)
    history = []
    done = False
    while not done:
        raw = fallback_action(task_id, history)
        action = SupportOpsAction(**raw)
        obs = env.step(action)
        history.append(raw)
        done = obs.done
    return env.state.current_score


def test_c1_access_happy_path_reaches_high_score():
    assert run_fallback("c1_access_lockout") >= 0.78


def test_c4_gdpr_happy_path_reaches_high_score():
    assert run_fallback("c4_gdpr_churn") >= 0.72


def test_score_is_strictly_inside_unit_interval():
    env = SupportOpsEnvironment()
    env.reset(task_id="c1_access_lockout")
    score = env.state.current_score
    assert 0.0 < score < 1.0


def test_invalid_tool_call_hard_fails():
    env = SupportOpsEnvironment()
    env.reset(task_id="c1_access_lockout")
    action = SupportOpsAction(
        assistant_message="I am trying an unsupported tool.",
        tool_calls=[{"name": "policy.search", "args": {"query": "access"}}, {"name": "workflow.set_priority", "args": {"case_id": "NOPE", "priority": "urgent"}}],
    )
    res = env.step(action)
    assert res.done is True
    assert res.reward == -1.0
    assert env.state.termination_reason == "invalid_tool"


def test_irrelevant_case_modification_reduces_score():
    env = SupportOpsEnvironment()
    env.reset(task_id="c1_access_lockout")
    env.step(
        SupportOpsAction(
            assistant_message="I am incorrectly routing the distractor.",
            tool_calls=[
                {"name": "workflow.assign_team", "args": {"case_id": "A-1002", "team": "billing"}},
            ],
        )
    )
    assert env.state.penalty_breakdown["irrelevant_penalty"] > 0.0


def test_unsupported_reply_claim_hard_fails():
    env = SupportOpsEnvironment()
    env.reset(task_id="c1_access_lockout")
    env.step(
        SupportOpsAction(
            assistant_message="I am opening the urgent case.",
            tool_calls=[{"name": "inbox.open_case", "args": {"case_id": "A-1001"}}],
        )
    )
    res = env.step(
        SupportOpsAction(
            assistant_message="I am drafting an unsafe claim.",
            tool_calls=[
                {
                    "name": "comms.draft_reply",
                    "args": {
                        "case_id": "A-1001",
                        "reply_text": "We will update you within 15 minutes and your backup admin can help immediately.",
                    },
                }
            ],
        )
    )
    assert res.done is True
    assert res.reward == -1.0
    assert env.state.termination_reason == "unsafe_reply"


def test_curriculum_promotes_after_repeated_success():
    env = SupportOpsEnvironment()
    for _ in range(3):
        env.reset(task_id="c1_access_lockout")
        history = []
        done = False
        while not done:
            raw = fallback_action("c1_access_lockout", history)
            obs = env.step(SupportOpsAction(**raw))
            history.append(raw)
            done = obs.done
    stats = env.state.curriculum_stats
    assert stats["difficulty_level"] >= 2
