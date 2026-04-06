from support_ops_env.inference import fallback_action
from support_ops_env.models import SupportOpsAction
from support_ops_env.server.support_ops_environment import SupportOpsEnvironment
from support_ops_env.tasks import TASK_IDS


def run(task_id: str) -> float:
    env = SupportOpsEnvironment()
    env.reset(task_id=task_id, seed=7)
    hist = []
    done = False
    while not done:
        act = SupportOpsAction(**fallback_action(task_id, hist))
        obs = env.step(act)
        hist.append(act.model_dump())
        done = obs.done
    return env.state.current_score


def test_fallback_baseline_is_reproducible():
    s1 = {tid: run(tid) for tid in TASK_IDS}
    s2 = {tid: run(tid) for tid in TASK_IDS}
    assert s1 == s2
