from support_ops_env.inference import fallback_action
from support_ops_env.models import SupportOpsAction
from support_ops_env.server.support_ops_environment import SupportOpsEnvironment


def run_fallback(task_id: str) -> float:
    env = SupportOpsEnvironment()
    env.reset(task_id=task_id)
    hist = []
    done = False
    while not done:
        raw = fallback_action(task_id, hist)
        act = SupportOpsAction(**raw)
        obs = env.step(act)
        hist.append(raw)
        done = obs.done
    return env.state.current_score


def test_easy_task_can_reach_high_score():
    assert run_fallback("easy_vip_sso") >= 0.9


def test_medium_task_can_reach_high_score():
    assert run_fallback("medium_refund_duplicate") >= 0.8


def test_hard_task_can_reach_high_score():
    assert run_fallback("hard_security_phishing") >= 0.8


def test_expert_task_can_reach_high_score():
    assert run_fallback("expert_compliance_trap") >= 0.7


def test_invalid_duplicate_action_increases_invalid_count():
    env = SupportOpsEnvironment()
    env.reset(task_id="easy_vip_sso")
    env.step(SupportOpsAction(action_type="set_priority", ticket_id="E-1001", priority="urgent"))
    res = env.step(SupportOpsAction(action_type="set_priority", ticket_id="E-1001", priority="urgent"))
    assert res.reward == 0.0
    assert "Invalid action" in res.last_event
    assert env.state.invalid_action_count == 1


def test_max_steps_end_episode_without_submit():
    env = SupportOpsEnvironment()
    env.reset(task_id="easy_vip_sso")
    last = None
    for _ in range(8):
        last = env.step(SupportOpsAction(action_type="view_ticket", ticket_id="E-1001"))
    assert last is not None
    assert last.done is True
    assert env.state.submitted is False


def test_unsafe_reply_is_penalized():
    env = SupportOpsEnvironment()
    env.reset(task_id="hard_security_phishing")
    env.step(SupportOpsAction(action_type="view_ticket", ticket_id="H-3001"))
    env.step(SupportOpsAction(action_type="view_ticket", ticket_id="H-3002"))
    env.step(SupportOpsAction(action_type="merge_ticket", ticket_id="H-3002", target_ticket_id="H-3001"))
    env.step(SupportOpsAction(action_type="set_priority", ticket_id="H-3001", priority="urgent"))
    env.step(SupportOpsAction(action_type="assign_team", ticket_id="H-3001", team="security"))
    env.step(SupportOpsAction(action_type="add_tags", ticket_id="H-3001", tags=["security", "phishing"]))
    env.step(SupportOpsAction(action_type="set_status", ticket_id="H-3001", status="escalated"))
    env.step(SupportOpsAction(
        action_type="draft_reply", ticket_id="H-3001",
        reply="This was definitely a confirmed breach. Please send us your password and MFA code so we can fix it.",
    ))
    assert env.state.penalty_breakdown["reply_safety_penalty"] > 0.0
