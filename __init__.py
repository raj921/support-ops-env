"""Support Ops Env — an OpenEnv benchmark for SaaS support operations."""

from .client import SupportOpsEnv
from .models import SupportOpsAction, SupportOpsObservation, SupportOpsState
from .tasks import TASK_IDS, get_task_spec, list_task_specs

__all__ = [
    "SupportOpsAction",
    "SupportOpsEnv",
    "SupportOpsObservation",
    "SupportOpsState",
    "TASK_IDS",
    "get_task_spec",
    "list_task_specs",
]


def get_training_utils():
    """Lazy-import training utilities from :mod:`support_ops_env.train`.

    Returns a dict with: ``SYSTEM_PROMPT``, ``rollout_once``,
    ``format_observation``, ``format_history``, ``parse_tool_calls``,
    ``apply_chat_template``, ``reward_total``, ``reward_fields``,
    ``reward_reply``, ``reward_grounding``, ``plot_rewards``,
    ``patch_trl_vllm_compat``.

    Example (Colab)::

        from support_ops_env import get_training_utils

        tu = get_training_utils()
        SYSTEM_PROMPT = tu["SYSTEM_PROMPT"]
        rollout_once = tu["rollout_once"]
    """
    from . import train as _train

    return {
        "SYSTEM_PROMPT": _train.SYSTEM_PROMPT,
        "rollout_once": _train.rollout_once,
        "format_observation": _train.format_observation,
        "format_history": _train.format_history,
        "parse_tool_calls": _train.parse_tool_calls,
        "apply_chat_template": _train.apply_chat_template,
        "reward_total": _train.reward_total,
        "reward_fields": _train.reward_fields,
        "reward_reply": _train.reward_reply,
        "reward_grounding": _train.reward_grounding,
        "plot_rewards": _train.plot_rewards,
        "patch_trl_vllm_compat": _train.patch_trl_vllm_compat,
    }


def get_gemma4_training_utils():
    """Lazy-import the Gemma 4 training utilities from :mod:`support_ops_env.train_gemma4`.

    This path uses TRL's ``environment_factory`` API (the newer pattern shipped
    with Gemma 4's CARLA reference). Install with::

        pip install -e ".[gemma]"

    Returns a dict with: ``SYSTEM_PROMPT``, ``SupportOpsToolEnv``, ``reward_total``,
    ``reward_fields``, ``reward_reply``, ``reward_merge``.

    Example (Colab)::

        from support_ops_env import get_gemma4_training_utils

        g4 = get_gemma4_training_utils()
        ToolEnv = g4["SupportOpsToolEnv"]
        ToolEnv._env_url = "https://<you>-support-ops-env.hf.space"
    """
    from . import train_gemma4 as _g4

    return {
        "SYSTEM_PROMPT": _g4.SYSTEM_PROMPT,
        "SupportOpsToolEnv": _g4.SupportOpsToolEnv,
        "reward_total": _g4.reward_total,
        "reward_investigation": _g4.reward_investigation,
        "reward_routing": _g4.reward_routing,
        "reward_reply": _g4.reward_reply,
        "reward_groundedness": _g4.reward_groundedness,
        # Back-compat aliases (older cells may still import these).
        "reward_fields": _g4.reward_routing,
        "reward_merge": _g4.reward_investigation,
    }
