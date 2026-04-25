#!/usr/bin/env python3
"""
eval_compare.py — Baseline vs Trained comparison runner (guide §19 demo format).

Runs the **same** set of Support Ops tasks against:

  1. A base model (e.g. ``Qwen/Qwen3-4B-Instruct-2507``), and optionally
  2. The same base model with a trained LoRA adapter attached.

Emits a deterministic JSON + Markdown table comparing the two runs, broken
down by component (investigation / routing / reply_quality / groundedness /
submission) plus penalties and a pass/fail flag. This is the artifact judges
want to see: "before vs after, on identical tasks, with the numbers and the
safeguards clearly shown".

Usage
-----

Baseline only (quick smoke)::

    python eval_compare.py --env-url http://localhost:8000 --episodes 1

Baseline vs trained LoRA::

    python eval_compare.py \
        --env-url https://raj23211-support-ops-env.hf.space \
        --base-model Qwen/Qwen3-4B-Instruct-2507 \
        --adapter-path outputs/support-ops-grpo-2026-.../ \
        --difficulty easy \
        --episodes 2 \
        --output-dir eval_runs/run1

Design
------
* Uses the same ``apply_chat_template`` + ``parse_tool_calls`` + ``rollout_once``
  path as training (``support_ops_env.train``), so eval and training see the
  same formatting — no off-by-one surprises.
* Loads the model in 4-bit NF4 / bf16 compute by default so a T4 can run eval
  too. Pass ``--no-4bit`` to load in bf16 (A100+).
* The LoRA adapter is loaded via ``peft.PeftModel.from_pretrained`` — never
  merged, never upcast (guide §16).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Data shapes
# ----------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    run: str                 # "baseline" or "trained"
    task_id: str
    total_reward: float
    investigation: float
    routing: float
    reply_quality: float
    groundedness: float
    submission: float
    penalty_total: float
    penalty_breakdown: Dict[str, float]
    turns: int
    done: bool
    surfaced_facts: List[str]
    tool_call_names: List[str]
    final_answer: Optional[Dict[str, Any]]


# ----------------------------------------------------------------------
# Minimal rollout that captures component breakdown + tool sequence
# ----------------------------------------------------------------------

def _run_episode(trainer_model, tokenizer, env, task_id: str, max_turns: int, system_prompt: str) -> EpisodeRecord:
    """Run one episode using the Qwen chat-template rollout from train.py.

    We deliberately reuse ``rollout_once`` indirectly by re-implementing its
    hot loop here — that lets us *also* record tool-call names and the final
    answer (which ``rollout_once`` returns only as aggregated reward).
    """
    import torch

    from support_ops_env import SupportOpsAction
    from support_ops_env.train import (
        SYSTEM_PROMPT as _DEFAULT_SYS,
        apply_chat_template,
        format_history,
        format_observation,
        parse_tool_calls,
    )

    system_prompt = system_prompt or _DEFAULT_SYS

    device = trainer_model.device if hasattr(trainer_model, "device") else "cuda"
    tokenizer.padding_side = "left"

    reset = env.reset(task_id=task_id)
    obs = reset.observation
    history: List[Dict[str, Any]] = []
    tool_call_names: List[str] = []
    surfaced_facts: List[str] = []
    final_answer: Optional[Dict[str, Any]] = None
    done = False

    for _ in range(max_turns):
        user_text = format_observation(obs)
        history_text = format_history(history)
        prompt = apply_chat_template(tokenizer, system_prompt, user_text, history_text)

        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            gen = trainer_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=False,            # greedy for deterministic eval
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion_text = tokenizer.decode(
            gen[0, input_ids.shape[1]:], skip_special_tokens=True
        )
        parsed = parse_tool_calls(completion_text)

        for tc in parsed.get("tool_calls") or []:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name:
                tool_call_names.append(name)

        if parsed.get("answer") and parsed["answer"].get("done"):
            final_answer = parsed["answer"]

        action = SupportOpsAction(
            assistant_message=parsed["assistant_message"],
            tool_calls=parsed.get("tool_calls") or [],
            answer=parsed.get("answer"),
        )
        step = env.step(action)

        # Track surfaced facts from the latest tool result.
        for tr in (step.observation.tool_results or [])[-len(action.tool_calls or [1]):]:
            surfaced_facts.extend(tr.surfaced_fact_ids or [])

        history.append({
            "assistant_message": action.assistant_message,
            "tool_calls": action.tool_calls,
            "reward": float(step.reward or 0.0),
            "done": bool(step.done),
        })

        obs = step.observation
        done = bool(step.done)
        if done:
            break

    breakdown = obs.reward_breakdown or {}
    penalty = obs.penalty_breakdown or {}

    return EpisodeRecord(
        run="",  # filled in by caller
        task_id=task_id,
        total_reward=float(obs.progress_score or 0.0),
        investigation=float(breakdown.get("investigation", 0.0)),
        routing=float(breakdown.get("routing", 0.0)),
        reply_quality=float(breakdown.get("reply_quality", 0.0)),
        groundedness=float(breakdown.get("groundedness", 0.0)),
        submission=float(breakdown.get("submission", 0.0)),
        penalty_total=float(sum(penalty.values())),
        penalty_breakdown=dict(penalty),
        turns=len(history),
        done=done,
        surfaced_facts=sorted(set(surfaced_facts)),
        tool_call_names=tool_call_names,
        final_answer=final_answer,
    )


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------

def _load_model(base_model: str, adapter_path: Optional[str], load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: Dict[str, Any] = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    logger.info("loading base model %s (4bit=%s)", base_model, load_in_4bit)
    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    if adapter_path:
        from peft import PeftModel

        logger.info("attaching LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def _aggregate(records: List[EpisodeRecord]) -> Dict[str, float]:
    if not records:
        return {}
    n = len(records)
    keys = ("total_reward", "investigation", "routing", "reply_quality",
            "groundedness", "submission", "penalty_total", "turns")
    agg = {k: sum(getattr(r, k) for r in records) / n for k in keys}
    agg["done_rate"] = sum(1 for r in records if r.done) / n
    agg["pass_rate"] = sum(1 for r in records if r.total_reward >= 0.5) / n
    return agg


def _markdown_report(base_agg: Dict[str, float], trained_agg: Dict[str, float],
                     base_model: str, adapter_path: Optional[str],
                     difficulty: str, episodes: int) -> str:
    lines = [
        f"# Baseline vs Trained — Support Ops Env",
        f"",
        f"- Base model: `{base_model}`",
        f"- Adapter:    `{adapter_path or '(none — baseline only)'}`",
        f"- Curriculum: `{difficulty}`",
        f"- Episodes:   {episodes} per run per task",
        f"",
        f"## Component means (higher is better except penalty)",
        f"",
        f"| Metric | Baseline | Trained | Δ |",
        f"|--------|----------|---------|----|",
    ]
    keys = [
        ("total_reward", "Total (progress_score)"),
        ("investigation", "Investigation"),
        ("routing", "Routing"),
        ("reply_quality", "Reply quality"),
        ("groundedness", "Groundedness"),
        ("submission", "Submission"),
        ("pass_rate", "Pass rate (total≥0.5)"),
        ("done_rate", "Done rate"),
        ("penalty_total", "Penalty total (↓)"),
        ("turns", "Turns (mean)"),
    ]
    for key, label in keys:
        b = base_agg.get(key, 0.0)
        t = trained_agg.get(key) if trained_agg else None
        if t is None:
            lines.append(f"| {label} | {b:+.3f} | — | — |")
        else:
            delta = t - b
            arrow = "▲" if delta > 0.001 else ("▼" if delta < -0.001 else "·")
            lines.append(f"| {label} | {b:+.3f} | {t:+.3f} | {arrow} {delta:+.3f} |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline vs Trained evaluation for Support Ops Env")
    p.add_argument("--env-url", default="http://localhost:8000")
    p.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--adapter-path", default=None,
                   help="Path to the trained LoRA adapter dir (optional). If omitted, runs baseline only.")
    p.add_argument("--difficulty", default="driftshield",
                   help="Curriculum to evaluate on: easy | medium | hard | expert | all | <task_id>")
    p.add_argument("--episodes", type=int, default=1,
                   help="Episodes per task (greedy, so usually 1 is enough).")
    p.add_argument("--max-turns", type=int, default=15)
    p.add_argument("--no-4bit", action="store_true",
                   help="Load the base model in bf16 instead of NF4 4-bit (use on A100+).")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    from support_ops_env import SupportOpsEnv, get_curriculum_task_ids
    from support_ops_env.train import SYSTEM_PROMPT

    tasks = get_curriculum_task_ids(args.difficulty)
    logger.info("evaluating on curriculum [%s] -> %s", args.difficulty, tasks)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output_dir or f"eval_runs/eval-{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = SupportOpsEnv(base_url=args.env_url).sync()

    def _run(run_name: str, adapter: Optional[str]) -> List[EpisodeRecord]:
        model, tok = _load_model(args.base_model, adapter, load_in_4bit=not args.no_4bit)
        records: List[EpisodeRecord] = []
        try:
            for task_id in tasks:
                for ep in range(args.episodes):
                    logger.info("[%s] task=%s ep=%d", run_name, task_id, ep)
                    rec = _run_episode(model, tok, env, task_id, args.max_turns, SYSTEM_PROMPT)
                    rec.run = run_name
                    records.append(rec)
                    logger.info(
                        "[%s] task=%s total=%.3f routing=%.3f reply=%.3f ground=%.3f penalty=%.3f",
                        run_name, task_id, rec.total_reward, rec.routing,
                        rec.reply_quality, rec.groundedness, rec.penalty_total,
                    )
        finally:
            # Free VRAM between runs.
            import gc
            import torch
            del model, tok
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return records

    baseline = _run("baseline", adapter=None)
    trained = _run("trained", adapter=args.adapter_path) if args.adapter_path else []

    base_agg = _aggregate(baseline)
    trained_agg = _aggregate(trained) if trained else {}

    # Save JSON
    payload = {
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "difficulty": args.difficulty,
        "env_url": args.env_url,
        "episodes_per_task": args.episodes,
        "tasks": tasks,
        "baseline": [asdict(r) for r in baseline],
        "trained": [asdict(r) for r in trained],
        "aggregates": {"baseline": base_agg, "trained": trained_agg},
        "generated_at": datetime.now().isoformat(),
    }
    (out_dir / "eval_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # Save Markdown
    md = _markdown_report(base_agg, trained_agg, args.base_model, args.adapter_path,
                          args.difficulty, args.episodes)
    (out_dir / "eval_results.md").write_text(md)

    logger.info("wrote %s", out_dir / "eval_results.json")
    logger.info("wrote %s", out_dir / "eval_results.md")
    print("\n" + md)


if __name__ == "__main__":
    main()
