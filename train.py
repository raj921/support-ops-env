"""
GRPO Training — Support Ops Env (Qwen path).

Two loaders:

* **Stable (default)** — HF ``transformers`` + ``bitsandbytes`` + ``peft``. Default
  model ``Qwen/Qwen3-4B-Instruct-2507``. Runs without Unsloth; vLLM is opt-in.
* **Unsloth fast path** (``--use-unsloth``) — ``unsloth.FastLanguageModel`` in
  bf16 + LoRA. Default model ``unsloth/Qwen3.5-4B``. Uses Unsloth inference
  (``fast_inference=False`` is required for Qwen3.5 GRPO). ~1.5× faster and
  ~50% less VRAM vs FA2 (Unsloth docs). Install with ``pip install -e '.[unsloth]'``.

Default model: ``Qwen/Qwen3-4B-Instruct-2507``.

Why this model (as of Apr 2026):
    * Dense 4B fits Colab T4 in 4-bit NF4 with bf16 compute, and fits comfortably
      on A100/L4 in bf16.
    * Non-thinking instruct — our JSON action parser does not have to strip
      ``<think>`` blocks every turn (unlike Qwen3.5-4B, which thinks by default).
    * Plays cleanly with vLLM for fast rollouts on A100+; on T4 we default to
      ``--no-vllm`` because vLLM colocate memory does not fit alongside a 4B
      model + KV cache.
    * Known stable with TRL GRPO (no weight-prefix mismatch bug seen on
      Qwen3.5-text-only checkpoints).

Upgrade path: swap to ``Qwen/Qwen3.5-4B`` on A100+ via ``--model-id`` if you want
the newer base. Expect to pass ``enable_thinking=False`` in
``chat_template_kwargs`` and set ``--no-vllm`` until vLLM registers a text-only
causal class for Qwen3.5.

Quickstart (T4 / A100)::

    pip install -e ".[train]"

    # Terminal 1 — serve the environment
    python -m server.app --port 8000

    # Terminal 2 — train the agent
    python train.py --env-url http://localhost:8000  # T4 defaults (no vLLM)
    python train.py --env-url http://localhost:8000 --use-vllm  # A100+

Design notes:
    * The training signal comes from :func:`grade_state` deltas, *not* from a
      pre-collected dataset. One episode = one Colab row in ``dataset``; rewards
      are produced at rollout time by the agent interacting with the live env.
    * ``reward_total`` wraps the grader's overall score; ``reward_fields``,
      ``reward_reply``, and ``reward_grounding`` expose the three main
      components so GRPO can see dense per-skill signal.
    * Tool-tier shaping (``TOOL_TIERS``) gives small positive reward for sensible
      investigation (CRM lookups, access checks, policy search) so the agent
      does not have to discover the full submission-first-then-reward curve
      from scratch.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

try:
    from support_ops_env import (
        SupportOpsAction,
        SupportOpsEnv,
        SupportOpsObservation,
    )
    from support_ops_env.tasks import TASK_IDS, get_curriculum_task_ids
except ImportError:
    from client import SupportOpsEnv
    from models import SupportOpsAction, SupportOpsObservation
    from tasks import TASK_IDS, get_curriculum_task_ids


logger = logging.getLogger(__name__)


# ============================================================
# TRL / vLLM compatibility
# ============================================================

_orig_vllm_gen = None


def _patch_vllm_generate(trainer: Any) -> None:
    """Wrap vLLM generate so TRL gets top-k-list logprobs even on vLLM 0.11.x."""
    global _orig_vllm_gen
    if _orig_vllm_gen is not None or not hasattr(trainer, "vllm_generation"):
        return
    _orig_vllm_gen = trainer.vllm_generation.generate

    def _wrapped_generate(**kwargs: Any) -> Any:
        result = _orig_vllm_gen(**kwargs)
        prompt_ids, completion_ids, logprobs, *rest = result
        if logprobs and logprobs[0] and isinstance(logprobs[0][0], float):
            logprobs = [[[lp] for lp in seq] for seq in logprobs]
        return (prompt_ids, completion_ids, logprobs, *rest)

    trainer.vllm_generation.generate = _wrapped_generate


def patch_trl_vllm_compat() -> None:
    """Apply TRL/vLLM compatibility patches. Call before ``trainer.train()``."""
    from trl import GRPOTrainer

    _orig_train = GRPOTrainer.train

    def _patched_train(self: Any, *args: Any, **kwargs: Any) -> Any:
        _patch_vllm_generate(self)
        return _orig_train(self, *args, **kwargs)

    GRPOTrainer.train = _patched_train


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT = """You are SupportOps Control Tower, a deterministic support-operations agent.

You operate six apps via tool calls: Inbox, CRM, Billing, Access, Policy, Comms.

You must respond with EXACTLY ONE JSON object per turn, no prose outside it:

{
  "assistant_message": "<one short sentence explaining this turn>",
  "tool_calls": [{"name": "<tool>", "args": {...}}],
  "answer": null | {
      "primary_case_id": "<id>",
      "resolved_case_ids": ["<id>", ...],
      "final_team": "<team>",
      "final_priority": "<priority>",
      "final_status": "<status>",
      "final_tags": ["..."],
      "reply_text": "<customer-facing reply, grounded in surfaced facts>",
      "done": true
  }
}

Available tools (use exact names):
  inbox.list_cases, inbox.open_case, inbox.merge_case, inbox.add_note
  crm.get_account, crm.get_contacts, crm.get_contract
  billing.get_invoice, billing.get_subscription, billing.issue_credit
  access.get_org_state, access.get_auth_events, access.revoke_sessions
  policy.search
  workflow.set_priority, workflow.assign_team, workflow.set_status, workflow.add_tags
  comms.draft_reply
  submit_resolution

Workflow:
  1. Open every relevant case with inbox.open_case.
  2. Investigate: pull CRM, billing, access, and policy context via the right tools.
  3. Route each case (priority + team + status + tags) using workflow.* tools.
  4. Merge duplicates with inbox.merge_case when two cases are the same incident.
  5. Draft ONE grounded reply with comms.draft_reply. Only claim facts you have surfaced.
  6. When all cases are handled, emit ``answer`` with ``done=true`` and call ``submit_resolution``.

Safety rules:
  * Never promise refunds, deletions, or SLAs that the policy does not support.
  * Never ask the customer for a password.
  * Do not merge unrelated cases.
  * Do not touch cases that belong to other customers' incidents (distractors).
"""


# ============================================================
# Tool-tier shaped reward
# ============================================================

TOOL_TIERS: Dict[str, float] = {
    # Tier 3 — high-value targeted investigation
    "inbox.open_case": 0.12,
    "crm.get_account": 0.12,
    "crm.get_contract": 0.12,
    "access.get_org_state": 0.12,
    "billing.get_invoice": 0.12,
    # Tier 2 — search / broader context
    "access.get_auth_events": 0.10,
    "billing.get_subscription": 0.10,
    "policy.search": 0.10,
    "crm.get_contacts": 0.08,
    # Tier 1 — orientation
    "inbox.list_cases": 0.04,
    # Tier 2.5 — productive routing / consolidation
    "workflow.set_priority": 0.08,
    "workflow.assign_team": 0.08,
    "workflow.set_status": 0.06,
    "workflow.add_tags": 0.06,
    "inbox.merge_case": 0.12,
    "inbox.add_note": 0.03,
    # Tier 3.5 — closing actions
    "comms.draft_reply": 0.15,
    "submit_resolution": 0.20,
    # Tier 1.5 — containment
    "access.revoke_sessions": 0.10,
    "billing.issue_credit": 0.05,
}


# ============================================================
# Prompt / observation formatting
# ============================================================

def format_observation(obs: SupportOpsObservation) -> str:
    """Render a :class:`SupportOpsObservation` as agent-readable text."""
    pieces: List[str] = [
        f"TASK: {obs.task_title}",
        f"COLLECTION: {obs.collection}  FAMILY: {obs.task_family}",
        f"OBJECTIVE: {obs.objective}",
        "",
        "APP SUMMARIES:",
    ]
    for app, summary in (obs.app_summaries or {}).items():
        pieces.append(f"  [{app}] {summary}")

    if obs.tool_results:
        pieces.append("")
        pieces.append("LATEST TOOL RESULTS:")
        for tr in obs.tool_results[-4:]:
            status = "ok" if tr.ok else "error"
            result_str = json.dumps(tr.result, ensure_ascii=True)[:280]
            pieces.append(f"  {tr.name} [{status}] surfaced={tr.surfaced_fact_ids} -> {result_str}")

    pieces.append("")
    pieces.append(
        f"PROGRESS: score={obs.progress_score:.2f}  "
        f"remaining_steps={obs.remaining_steps}  "
        f"visible_cases={obs.visible_case_ids}"
    )
    if obs.recent_actions:
        pieces.append(f"RECENT ACTIONS: {obs.recent_actions[-5:]}")
    pieces.append(f"GUIDANCE: {obs.guidance}")
    return "\n".join(pieces)


def format_history(history: List[Dict[str, Any]]) -> str:
    """Compact summary of past (action, reward) pairs for the agent."""
    if not history:
        return ""
    lines = ["PREVIOUS TURNS:"]
    for entry in history[-5:]:
        tc_names = [tc.get("name", "?") for tc in entry.get("tool_calls") or []]
        reward = entry.get("reward", 0.0)
        done = entry.get("done", False)
        msg = (entry.get("assistant_message") or "")[:140]
        lines.append(f"  msg={msg!r} tools={tc_names} reward={reward:+.2f} done={done}")
    return "\n".join(lines)


def apply_chat_template(
    tokenizer: Any,
    system_prompt: str,
    user_text: str,
    history_text: str = "",
) -> str:
    """Render a chat-style prompt using the tokenizer's chat template."""
    messages = [{"role": "system", "content": system_prompt}]
    if history_text:
        messages.append({"role": "user", "content": history_text})
    messages.append({"role": "user", "content": user_text})
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for tokenizers without a chat template
        joined = "\n\n".join(m["content"] for m in messages)
        return joined + "\n\nASSISTANT:"


# ============================================================
# Action parsing
# ============================================================

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.MULTILINE)


def _strip_fences(text: str) -> str:
    match = _JSON_FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def parse_tool_calls(text: str) -> Dict[str, Any]:
    """Parse a model completion into fields for :class:`SupportOpsAction`.

    Returns a dict with keys ``assistant_message``, ``tool_calls``, ``answer``.
    Falls back to a safe ``submit_resolution`` no-op if parsing fails so the
    episode still makes forward progress instead of crashing the trainer.
    """
    cleaned = _strip_fences(text)
    try:
        obj = json.loads(cleaned)
        if not isinstance(obj, dict):
            raise ValueError("expected a JSON object")
        return {
            "assistant_message": obj.get("assistant_message") or "(no message)",
            "tool_calls": obj.get("tool_calls") or [],
            "answer": obj.get("answer"),
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "assistant_message": "parser_error: falling back to safe no-op",
            "tool_calls": [],
            "answer": None,
        }


def to_action(parsed: Dict[str, Any]) -> SupportOpsAction:
    return SupportOpsAction(
        assistant_message=parsed["assistant_message"] or "(empty)",
        tool_calls=parsed.get("tool_calls") or [],
        answer=parsed.get("answer"),
    )


# ============================================================
# Reward wrappers
# ============================================================

def _score_components(info: Dict[str, Any]) -> Dict[str, float]:
    """Extract grader components from a step's ``info`` / ``observation`` dump."""
    rb = info.get("reward_breakdown") or {}
    return {
        "views": float(rb.get("views", 0.0)),
        "field_alignment": float(rb.get("field_alignment", 0.0)),
        "reply_quality": float(rb.get("reply_quality", 0.0)),
        "merge_quality": float(rb.get("merge_quality", 0.0)),
        "submitted": float(rb.get("submitted", 0.0)),
        "grounding": float(rb.get("grounding_quality", 0.0)),
    }


def _coerce_reward_list(values: Any) -> List[float]:
    if isinstance(values, list):
        return [float(v) for v in values]
    return [float(values)]


def reward_total(total_reward, **kwargs) -> List[float]:
    """Overall reward per sample from rollout_func extra fields."""
    del kwargs
    return _coerce_reward_list(total_reward)


def reward_fields(field_reward, **kwargs) -> List[float]:
    """Routing-field correctness from rollout_func extra fields."""
    del kwargs
    return _coerce_reward_list(field_reward)


def reward_reply(reply_reward, **kwargs) -> List[float]:
    """Reply quality from rollout_func extra fields."""
    del kwargs
    return _coerce_reward_list(reply_reward)


def reward_grounding(grounding_reward, **kwargs) -> List[float]:
    """Grounding quality from rollout_func extra fields."""
    del kwargs
    return _coerce_reward_list(grounding_reward)


# ============================================================
# Rollout
# ============================================================

def rollout_once(
    trainer: Any,
    env: SupportOpsEnv,
    tokenizer: Any,
    system_prompt: str = SYSTEM_PROMPT,
    max_turns: int = 15,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one episode and return a GRPO-ready dict.

    The returned dict matches TRL's ``rollout_func`` contract:

    * ``prompt_ids``, ``completion_ids``, ``logprobs`` — passed through vLLM.
    * ``total_reward``, ``field_reward``, ``reply_reward``, ``grounding_reward``
      — scalar per-episode rewards surfaced to GRPO's reward functions.
    """
    import torch

    device = trainer.accelerator.device if hasattr(trainer, "accelerator") else "cuda"
    model = trainer.model
    tokenizer.padding_side = "left"

    task_id = task_id or TASK_IDS[0]
    reset = env.reset(task_id=task_id, seed=None)
    obs: SupportOpsObservation = reset.observation
    history: List[Dict[str, Any]] = []

    prompt_ids_acc: List[int] = []
    completion_ids_acc: List[int] = []
    logprobs_acc: List[List[float]] = []

    total_reward = 0.0
    shaped_bonus = 0.0
    done = False

    for turn in range(max_turns):
        user_text = format_observation(obs)
        history_text = format_history(history)
        prompt = apply_chat_template(tokenizer, system_prompt, user_text, history_text)

        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        completion_ids = gen[0, input_ids.shape[1]:].tolist()
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

        prompt_ids_acc.extend(input_ids[0].tolist())
        completion_ids_acc.extend(completion_ids)
        logprobs_acc.append([0.0] * len(completion_ids))  # vLLM mode replaces this

        parsed = parse_tool_calls(completion_text)
        action = to_action(parsed)

        # Tool-tier shaping before calling env
        for tc in action.tool_calls or []:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name:
                shaped_bonus += TOOL_TIERS.get(name, 0.0)
        if action.answer and action.answer.get("done"):
            shaped_bonus += TOOL_TIERS.get("submit_resolution", 0.0)

        step = env.step(action)
        step_reward = float(step.reward or 0.0)
        total_reward += step_reward

        history.append(
            {
                "assistant_message": action.assistant_message,
                "tool_calls": action.tool_calls,
                "reward": step_reward,
                "done": bool(step.done),
            }
        )

        obs = step.observation
        done = bool(step.done)
        if done:
            break

    # Pull component breakdown from last observation's reward_breakdown.
    # Keys match graders.py: investigation, routing, reply_quality, groundedness, submission.
    breakdown = obs.reward_breakdown or {}
    penalty = obs.penalty_breakdown or {}
    investigation_reward = float(breakdown.get("investigation", 0.0))
    routing_reward = float(breakdown.get("routing", 0.0))
    reply_reward = float(breakdown.get("reply_quality", 0.0))
    grounding_reward = float(breakdown.get("groundedness", 0.0))
    submission_reward = float(breakdown.get("submission", 0.0))
    total_penalty = float(sum(penalty.values()))

    return {
        "prompt_ids": prompt_ids_acc,
        "completion_ids": completion_ids_acc,
        "logprobs": logprobs_acc[-1] if logprobs_acc else [],
        "total_reward": total_reward + shaped_bonus,
        # back-compat names expected by older notebooks + reward_funcs
        "field_reward": routing_reward,
        "reply_reward": reply_reward,
        "grounding_reward": grounding_reward,
        # full breakdown for new plot/eval tooling
        "investigation_reward": investigation_reward,
        "routing_reward": routing_reward,
        "submission_reward": submission_reward,
        "penalty_total": total_penalty,
        "penalty_breakdown": dict(penalty),
        "task_id": task_id,
        "shaped_bonus": shaped_bonus,
        "turns": len(history),
        "done": done,
    }


# ============================================================
# Plotting
# ============================================================

def plot_rewards(csv_path: os.PathLike, out_path: Optional[os.PathLike] = None) -> Optional[Path]:
    """Render the reward curve from a CSV log written during training."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.warning("reward csv not found at %s", csv_path)
        return None

    episodes: List[int] = []
    total: List[float] = []
    field: List[float] = []
    reply: List[float] = []
    ground: List[float] = []
    with open(csv_path) as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 5:
                continue
            episodes.append(int(row[0]))
            total.append(float(row[1]))
            field.append(float(row[2]))
            reply.append(float(row[3]))
            ground.append(float(row[4]))

    if not episodes:
        logger.warning("reward csv at %s has no rows", csv_path)
        return None

    def _rolling(values: List[float], window: int = 10) -> List[float]:
        window = min(window, len(values))
        return [
            sum(values[max(0, i - window):i + 1]) / min(i + 1, window)
            for i in range(len(values))
        ]

    out = Path(out_path) if out_path else csv_path.with_suffix(".png")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    window = min(10, len(episodes))

    ax1.plot(episodes, total, alpha=0.3, color="#1f77b4", label="per episode")
    ax1.plot(episodes, _rolling(total, window), color="#1f77b4", linewidth=2, label=f"rolling({window})")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Total reward")
    ax1.set_title(f"Support Ops Env — GRPO reward ({len(episodes)} episodes)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, _rolling(field, window), color="#ff7f0e", linewidth=2, label="field alignment")
    ax2.plot(episodes, _rolling(reply, window), color="#2ca02c", linewidth=2, label="reply quality")
    ax2.plot(episodes, _rolling(ground, window), color="#9467bd", linewidth=2, label="grounding")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Component reward")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("wrote reward plot to %s", out)
    return out


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for Support Ops Env")
    parser.add_argument("--model-id", default=None,
                        help="HF model id. Defaults to 'Qwen/Qwen3-4B-Instruct-2507' for the "
                             "stable path, or 'unsloth/Qwen3.5-4B' when --use-unsloth is set.")
    parser.add_argument("--use-unsloth", action="store_true",
                        help="Use the Unsloth loader (bf16 LoRA, Unsloth generation). "
                             "Requires the [unsloth] extra.")
    parser.add_argument("--env-url", default="http://localhost:8000")
    parser.add_argument("--dataset-size", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--use-vllm", action="store_true",
                        help="Enable vLLM rollouts (A100/L4+). Off by default — too big for T4.")
    parser.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load base model in NF4 (bitsandbytes). T4-friendly default via notebook.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--reward-log", default="reward_log.csv")
    parser.add_argument(
        "--difficulty",
        default="driftshield_easy",
        help=(
            "Curriculum: driftshield | driftshield_easy | easy | medium | hard | expert | all | "
            "<task_id>. Defaults to 'driftshield_easy' (D1 prompt-injection warmup) so the model "
            "sees non-zero reward early on the new DriftShield collection."
        ),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Build trainer but skip trainer.train() for smoke-testing wiring.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    # Resolve default model id based on loader choice.
    if args.model_id is None:
        args.model_id = (
            "unsloth/Qwen3.5-4B" if args.use_unsloth else "Qwen/Qwen3-4B-Instruct-2507"
        )

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    patch_trl_vllm_compat()

    # ---------- Model loader ----------
    loaded_model: Any = None        # object if preloaded, else None -> TRL loads by id
    if args.use_unsloth:
        try:
            from unsloth import FastLanguageModel  # type: ignore
        except ImportError as exc:                  # noqa: BLE001
            raise SystemExit(
                "--use-unsloth requires the [unsloth] extra: pip install -e '.[unsloth]'"
            ) from exc

        logger.info("[unsloth] loading %s (bf16 LoRA, fast_inference=False)", args.model_id)
        loaded_model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_id,
            max_seq_length=4096,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
            fast_inference=False,  # required for Qwen3.5 GRPO (Unsloth docs)
        )
        loaded_model = FastLanguageModel.get_peft_model(
            loaded_model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            max_seq_length=4096,
        )
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    env = SupportOpsEnv(base_url=args.env_url).sync()
    dataset = Dataset.from_dict(
        {"prompt": ["Triage and resolve this support operations case."] * args.dataset_size}
    )

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir or f"outputs/support-ops-grpo-{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    reward_log = output_dir / args.reward_log
    # Widened schema: all 5 components + penalty total + task_id. Backward compat
    # readers can ignore extra columns (see plot_rewards.py).
    with open(reward_log, "w", newline="") as fh:
        csv.writer(fh).writerow([
            "episode", "task_id",
            "total_reward",
            "investigation", "routing", "reply_quality", "groundedness", "submission",
            "penalty_total",
            "timestamp",
        ])

    curriculum = get_curriculum_task_ids(args.difficulty)
    logger.info("curriculum [%s] -> %s", args.difficulty, curriculum)
    task_cursor = [0]
    episode_counter = [0]

    def _next_task_id() -> str:
        tid = curriculum[task_cursor[0] % len(curriculum)]
        task_cursor[0] += 1
        return tid

    def _log_episode(ep: Dict[str, Any]) -> None:
        episode_counter[0] += 1
        with open(reward_log, "a", newline="") as fh:
            csv.writer(fh).writerow([
                episode_counter[0],
                ep.get("task_id", ""),
                round(ep["total_reward"], 4),
                round(ep["investigation_reward"], 4),
                round(ep["routing_reward"], 4),
                round(ep["reply_reward"], 4),
                round(ep["grounding_reward"], 4),
                round(ep["submission_reward"], 4),
                round(ep["penalty_total"], 4),
                datetime.now().isoformat(),
            ])

    def rollout_func(prompts: List[str], trainer: Any) -> Dict[str, List[Any]]:
        out: Dict[str, List[Any]] = {
            "prompt_ids": [], "completion_ids": [], "logprobs": [],
            "total_reward": [], "field_reward": [], "reply_reward": [], "grounding_reward": [],
        }
        for _ in prompts:
            ep = rollout_once(
                trainer, env, tokenizer, SYSTEM_PROMPT, args.max_turns,
                task_id=_next_task_id(),
            )
            _log_episode(ep)
            for key in out:
                out[key].append(ep[key])
        return out

    grpo_kwargs: Dict[str, Any] = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.num_generations,
        num_generations=args.num_generations,
        max_completion_length=512,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        temperature=args.temperature,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_total_limit=3,
    )
    if args.use_vllm and not args.use_unsloth:
        grpo_kwargs.update(
            use_vllm=True,
            vllm_mode=args.vllm_mode,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        )
    elif args.use_vllm and args.use_unsloth:
        logger.warning("--use-vllm is ignored in Unsloth mode (Unsloth owns generation).")

    if args.load_in_4bit and not args.use_unsloth:
        import torch
        from transformers import BitsAndBytesConfig

        grpo_kwargs["model_init_kwargs"] = {
            "torch_dtype": torch.bfloat16,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
        }
    elif args.load_in_4bit and args.use_unsloth:
        logger.warning(
            "--load-in-4bit is ignored in Unsloth mode (QLoRA not recommended for Qwen3.5; using bf16)."
        )

    grpo_config = GRPOConfig(**grpo_kwargs)

    # In Unsloth mode the LoRA adapters were already attached by
    # FastLanguageModel.get_peft_model, so we pass the loaded model object and
    # set peft_config=None. In stable mode we pass the HF model id (a string)
    # and let TRL+PEFT attach LoRA for us.
    if args.use_unsloth:
        model_arg: Any = loaded_model
        peft_config: Optional[Any] = None
    else:
        from peft import LoraConfig

        model_arg = args.model_id
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )

    trainer = GRPOTrainer(
        model=model_arg,
        processing_class=tokenizer,
        reward_funcs=[reward_total, reward_fields, reward_reply, reward_grounding],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
        peft_config=peft_config,
    )

    if args.dry_run:
        logger.info("--dry-run: trainer constructed, skipping trainer.train()")
        return

    try:
        trainer.train()
    finally:
        env.close() if hasattr(env, "close") else None

    trainer.save_model(str(output_dir))
    plot_rewards(reward_log, output_dir / "reward_curve.png")
    logger.info("training complete, artifacts in %s", output_dir)


if __name__ == "__main__":
    main()
