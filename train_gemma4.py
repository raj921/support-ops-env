"""
GRPO Training — Support Ops Env × Gemma 4 (TRL environment_factory pattern).

This module mirrors the official reference from Hugging Face:
    https://github.com/huggingface/huggingface-gemma-recipes/blob/main/scripts/carla_vlm_gemma.py

Key differences vs ``train.py``:

    * Uses TRL's new ``environment_factory`` API (tools as Python methods on an env
      class, auto-introspected by their docstrings + type hints) — not ``rollout_func``.
    * Loads Gemma 4 via ``AutoModelForCausalLM`` + ``AutoProcessor`` (text-only path).
      Gemma 4 is multimodal but we don't feed images, so the simpler CausalLM head is
      enough and avoids pulling in vision tower weights we'd never use.
    * Toggles reasoning via ``chat_template_kwargs={"enable_thinking": True}`` so the
      rollouts get the "Thinking" boost that shows up on the Tau2 benchmark.

Default model: ``google/gemma-4-E2B-it`` (5.1B total params, fits Colab T4 in 4-bit).

Quickstart::

    # 1. Install (TRL git HEAD required for environment_factory + Gemma 4)
    pip install "trl @ git+https://github.com/huggingface/trl.git"
    pip install "transformers>=4.57.0" accelerate peft bitsandbytes matplotlib

    # 2. Serve the environment (local dev) or point at a deployed HF Space
    python -m server.app --port 8000

    # 3. Train
    python train_gemma4.py --env-url http://localhost:8000

Design note:
    Tools on ``SupportOpsToolEnv`` use **snake_case method names** (e.g.
    ``inbox_open_case``) instead of the wire-format ``inbox.open_case``. This is a
    Python-syntax constraint from TRL's introspection. The method body translates
    the snake_case name back to the dotted tool name before calling the env.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

try:
    from support_ops_env import SupportOpsAction, SupportOpsEnv
    from support_ops_env.tasks import TASK_IDS
except ImportError:
    from client import SupportOpsEnv
    from models import SupportOpsAction
    from tasks import TASK_IDS


logger = logging.getLogger(__name__)


# ============================================================
# System prompt
# ============================================================

SYSTEM_PROMPT = """You are SupportOps Control Tower, a first-line SaaS support operator.

You must resolve customer cases by calling the tools you have available. Always:
  1. Open the relevant case(s) first with inbox_open_case.
  2. Investigate account / contract / billing / access context before deciding.
  3. Merge duplicates with inbox_merge_case when two cases are the same incident.
  4. Route each real case with workflow_set_priority / workflow_assign_team /
     workflow_set_status / workflow_add_tags.
  5. Draft ONE grounded customer reply with comms_draft_reply — only claim facts
     you have surfaced via earlier tool calls.
  6. When all real cases are handled, call submit_resolution with the final state.

Safety rules:
  * Never promise refunds, deletions, or SLAs not supported by policy.
  * Never ask the customer for a password.
  * Do not touch distractor cases from other customers.
"""


# ============================================================
# Tool-name mapping (snake_case method -> dotted wire name)
# ============================================================

_METHOD_TO_TOOL = {
    "inbox_list_cases":       "inbox.list_cases",
    "inbox_open_case":        "inbox.open_case",
    "inbox_merge_case":       "inbox.merge_case",
    "crm_get_account":        "crm.get_account",
    "crm_get_contract":       "crm.get_contract",
    "billing_get_invoice":    "billing.get_invoice",
    "billing_get_subscription": "billing.get_subscription",
    "access_get_org_state":   "access.get_org_state",
    "access_get_auth_events": "access.get_auth_events",
    "policy_search":          "policy.search",
    "workflow_set_priority":  "workflow.set_priority",
    "workflow_assign_team":   "workflow.assign_team",
    "workflow_set_status":    "workflow.set_status",
    "workflow_add_tags":      "workflow.add_tags",
    "comms_draft_reply":      "comms.draft_reply",
}


# ============================================================
# Environment wrapper — TRL will introspect this class
# ============================================================

class SupportOpsToolEnv:
    """TRL-compatible wrapper that exposes SupportOps tools as Python methods.

    Each public method becomes a tool TRL passes to the model via the chat template.
    The docstring (+ type hints) is the tool schema. Method names use snake_case
    because tool names are method identifiers. Dotted wire names like
    ``inbox.open_case`` are restored via ``_METHOD_TO_TOOL`` before calling the env.
    """

    _env_url: str = "http://localhost:8000"
    _task_id: Optional[str] = None

    def __init__(self) -> None:
        # EnvClient.step / reset are async; .sync() returns a blocking adapter that
        # TRL's environment_factory can call directly from its rollout loop.
        self.client = SupportOpsEnv(base_url=self._env_url).sync()
        self.reward = 0.0
        self._component_breakdown: Dict[str, float] = {}
        self._penalty_breakdown: Dict[str, float] = {}
        self._done = False
        self._reset()

    def _reset(self) -> None:
        task_id = self._task_id or TASK_IDS[0]
        result = self.client.reset(task_id=task_id)
        self._done = bool(result.done)
        obs = result.observation
        self._component_breakdown = dict(obs.reward_breakdown or {})
        self._penalty_breakdown = dict(obs.penalty_breakdown or {})
        self.reward = float(obs.progress_score or 0.0)

    def _dispatch(
        self,
        method_name: str,
        *,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        answer: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
    ) -> str:
        """Send one SupportOpsAction and return a compact tool-response string."""
        if self._done:
            return "episode already ended — submit_resolution was the last valid action"

        action = SupportOpsAction(
            assistant_message=message or f"calling {method_name}",
            tool_calls=tool_calls or [],
            answer=answer,
        )
        step = self.client.step(action)
        obs = step.observation
        self._done = bool(step.done)
        self._component_breakdown = dict(obs.reward_breakdown or {})
        self._penalty_breakdown = dict(obs.penalty_breakdown or {})
        self.reward = float(obs.progress_score or 0.0)

        if step.reward is not None:
            tool_results = obs.tool_results[-1:] if obs.tool_results else []
            if tool_results:
                tr = tool_results[0]
                payload = {"ok": tr.ok, "result": tr.result, "surfaced_facts": tr.surfaced_fact_ids}
                if tr.error:
                    payload["error"] = tr.error
                return json.dumps(payload, ensure_ascii=True)[:800]
        return json.dumps({"ok": True, "note": "action applied"}, ensure_ascii=True)

    # ---- observe-only tools ----

    def inbox_list_cases(self) -> str:
        """List every case currently visible in the inbox."""
        return self._dispatch("inbox_list_cases", tool_calls=[{"name": "inbox.list_cases", "args": {}}])

    def inbox_open_case(self, case_id: str) -> str:
        """Open a case by its id.

        Args:
            case_id: The id of the case to open (e.g. ``"A-1001"``).
        """
        return self._dispatch(
            "inbox_open_case",
            tool_calls=[{"name": "inbox.open_case", "args": {"case_id": case_id}}],
        )

    def crm_get_account(self, account_id: str) -> str:
        """Fetch CRM account details by account id.

        Args:
            account_id: Account identifier (e.g. ``"acct_northstar"``).
        """
        return self._dispatch(
            "crm_get_account",
            tool_calls=[{"name": "crm.get_account", "args": {"account_id": account_id}}],
        )

    def crm_get_contract(self, account_id: str) -> str:
        """Fetch the support contract (SLA, renewal date) for an account.

        Args:
            account_id: Account identifier.
        """
        return self._dispatch(
            "crm_get_contract",
            tool_calls=[{"name": "crm.get_contract", "args": {"account_id": account_id}}],
        )

    def billing_get_invoice(self, invoice_id: str) -> str:
        """Fetch an invoice by id.

        Args:
            invoice_id: Invoice identifier (e.g. ``"INV-18391"``).
        """
        return self._dispatch(
            "billing_get_invoice",
            tool_calls=[{"name": "billing.get_invoice", "args": {"invoice_id": invoice_id}}],
        )

    def billing_get_subscription(self, account_id: str) -> str:
        """Fetch the subscription plan and seat info for an account.

        Args:
            account_id: Account identifier.
        """
        return self._dispatch(
            "billing_get_subscription",
            tool_calls=[{"name": "billing.get_subscription", "args": {"account_id": account_id}}],
        )

    def access_get_org_state(self, account_id: str) -> str:
        """Return SSO / session / admin-fallback state for the account.

        Args:
            account_id: Account identifier.
        """
        return self._dispatch(
            "access_get_org_state",
            tool_calls=[{"name": "access.get_org_state", "args": {"account_id": account_id}}],
        )

    def access_get_auth_events(self, account_id: str) -> str:
        """Return recent auth events for the account.

        Args:
            account_id: Account identifier.
        """
        return self._dispatch(
            "access_get_auth_events",
            tool_calls=[{"name": "access.get_auth_events", "args": {"account_id": account_id}}],
        )

    def policy_search(self, query: str) -> str:
        """Search support policies by keyword.

        Args:
            query: Free-text query (e.g. ``"gdpr 30 day"``).
        """
        return self._dispatch(
            "policy_search",
            tool_calls=[{"name": "policy.search", "args": {"query": query}}],
        )

    # ---- mutating tools ----

    def inbox_merge_case(self, source_case_id: str, target_case_id: str) -> str:
        """Merge ``source_case_id`` into ``target_case_id``; the source is closed.

        Use this only when two cases describe the same incident from the same customer.

        Args:
            source_case_id: The duplicate case that will be closed.
            target_case_id: The owning case that will receive the merge.
        """
        return self._dispatch(
            "inbox_merge_case",
            tool_calls=[
                {
                    "name": "inbox.merge_case",
                    "args": {"source_case_id": source_case_id, "target_case_id": target_case_id},
                }
            ],
        )

    def workflow_set_priority(self, case_id: str, priority: str) -> str:
        """Set case priority.

        Args:
            case_id: The case to update.
            priority: One of ``"low"``, ``"normal"``, ``"high"``, ``"urgent"``.
        """
        return self._dispatch(
            "workflow_set_priority",
            tool_calls=[{"name": "workflow.set_priority", "args": {"case_id": case_id, "priority": priority}}],
        )

    def workflow_assign_team(self, case_id: str, team: str) -> str:
        """Assign the case to a support team.

        Args:
            case_id: The case to update.
            team: One of ``"general"``, ``"billing"``, ``"access"``, ``"product"``,
                ``"security"``, ``"compliance"``, ``"success"``.
        """
        return self._dispatch(
            "workflow_assign_team",
            tool_calls=[{"name": "workflow.assign_team", "args": {"case_id": case_id, "team": team}}],
        )

    def workflow_set_status(self, case_id: str, status: str) -> str:
        """Set case status.

        Args:
            case_id: The case to update.
            status: One of ``"open"``, ``"pending_customer"``, ``"escalated"``,
                ``"resolved"``, ``"closed"``.
        """
        return self._dispatch(
            "workflow_set_status",
            tool_calls=[{"name": "workflow.set_status", "args": {"case_id": case_id, "status": status}}],
        )

    def workflow_add_tags(self, case_id: str, tags: List[str]) -> str:
        """Add workflow tags to a case.

        Args:
            case_id: The case to update.
            tags: List of tag strings (e.g. ``["vip_customer", "sso"]``).
        """
        return self._dispatch(
            "workflow_add_tags",
            tool_calls=[{"name": "workflow.add_tags", "args": {"case_id": case_id, "tags": tags}}],
        )

    def comms_draft_reply(self, case_id: str, reply_text: str) -> str:
        """Draft the customer-facing reply for a case.

        Write one concise, grounded reply. Reference only facts surfaced via
        earlier tool calls. Do not promise refunds or deletions without policy.

        Args:
            case_id: The case the reply is for.
            reply_text: Full reply body (plain text).
        """
        return self._dispatch(
            "comms_draft_reply",
            tool_calls=[{"name": "comms.draft_reply", "args": {"case_id": case_id, "reply_text": reply_text}}],
        )

    # ---- terminal tool ----

    def submit_resolution(
        self,
        primary_case_id: str,
        resolved_case_ids: List[str],
        final_team: str,
        final_priority: str,
        final_status: str,
        final_tags: List[str],
        reply_text: str,
    ) -> str:
        """Submit the final resolution and end the episode.

        Call this exactly once, after all routing + the draft reply are in place.

        Args:
            primary_case_id: The main case id you resolved.
            resolved_case_ids: Every case id touched (including merged ones).
            final_team: The support team the primary case should belong to.
            final_priority: Final priority string.
            final_status: Final case status string.
            final_tags: Final tag list for the primary case.
            reply_text: The customer-facing reply text you drafted.
        """
        answer = {
            "primary_case_id": primary_case_id,
            "resolved_case_ids": resolved_case_ids,
            "final_team": final_team,
            "final_priority": final_priority,
            "final_status": final_status,
            "final_tags": final_tags,
            "reply_text": reply_text,
            "done": True,
        }
        out = self._dispatch("submit_resolution", answer=answer, message="submitting resolution")
        return out


# ============================================================
# Reward functions
# ----------------------------------------------------------------
# Keys below come from the deployed grader's reward_breakdown payload:
#   {investigation, routing, reply_quality, groundedness, submission}
# Penalty breakdown carries: unsupported_claim_penalty, unsafe_reply_penalty,
#   repeat_penalty, irrelevant_penalty, invalid_action_penalty,
#   no_progress_penalty, efficiency_penalty, disallowed_tool_penalty.
# ================================================================

def reward_total(completions, environments, **kwargs) -> List[float]:
    """Overall grader score (progress_score) — sums components minus penalties."""
    del completions, kwargs
    return [float(env.reward) for env in environments]


def reward_investigation(completions, environments, **kwargs) -> List[float]:
    """Investigation credit — did the agent surface the right facts via tool calls?"""
    del completions, kwargs
    return [float(env._component_breakdown.get("investigation", 0.0)) for env in environments]


def reward_routing(completions, environments, **kwargs) -> List[float]:
    """Routing correctness — priority + team + status + tags + merges."""
    del completions, kwargs
    return [float(env._component_breakdown.get("routing", 0.0)) for env in environments]


def reward_reply(completions, environments, **kwargs) -> List[float]:
    """Reply quality — keyword/paraphrase coverage minus forbidden phrases."""
    del completions, kwargs
    return [float(env._component_breakdown.get("reply_quality", 0.0)) for env in environments]


def reward_groundedness(completions, environments, **kwargs) -> List[float]:
    """Groundedness — did the reply claim only facts surfaced via tools?"""
    del completions, kwargs
    return [float(env._component_breakdown.get("groundedness", 0.0)) for env in environments]


# Back-compat aliases kept for older notebooks that imported these names.
reward_fields = reward_routing
reward_merge = reward_investigation


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for Support Ops Env (Gemma 4)")
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--env-url", default="http://localhost:8000")
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--dataset-size", type=int, default=50)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--max-tool-calling-iterations", type=int, default=12)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Turn on Gemma 4 reasoning mode during rollouts.")
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--report-to", default="none", choices=("wandb", "trackio", "none"))
    parser.add_argument("--trackio-space-id", default=None)
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-private-repo", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the trainer but skip trainer.train() for smoke-testing wiring.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    # Imports that require the [gemma] extra.
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    # Wire env config into the class (TRL instantiates it per-generation with no args).
    SupportOpsToolEnv._env_url = args.env_url
    SupportOpsToolEnv._task_id = args.task_id

    # Dataset: one user-prompt seed per episode. The env drives the rest.
    dataset = Dataset.from_dict(
        {
            "prompt": [
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Triage and resolve every real case in the inbox. "
                            "Investigate before you route. Draft one grounded reply. "
                            "Call submit_resolution when done."
                        ),
                    },
                ]
                for _ in range(args.dataset_size)
            ]
        }
    )

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir or f"outputs/support-ops-grpo-gemma4-{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    reward_log = output_dir / "reward_log.csv"
    if not reward_log.exists():
        with open(reward_log, "w", newline="") as fh:
            csv.writer(fh).writerow(
                ["episode", "total_reward", "field_reward", "reply_reward", "merge_reward", "timestamp"]
            )

    peft_config = None
    if args.use_lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
            exclude_modules=["vision_tower", "multi_modal_projector", "audio_tower"],
        )

    per_device_bs = args.per_device_train_batch_size or args.num_generations

    trainer = GRPOTrainer(
        model=args.model,
        train_dataset=dataset,
        reward_funcs=[reward_total, reward_investigation, reward_routing, reward_reply, reward_groundedness],
        peft_config=peft_config,
        args=GRPOConfig(
            chat_template_kwargs={"enable_thinking": args.enable_thinking},
            log_completions=True,
            logging_steps=args.logging_steps,
            num_completions_to_print=1,
            max_completion_length=args.max_completion_length,
            max_tool_calling_iterations=args.max_tool_calling_iterations,
            per_device_train_batch_size=per_device_bs,
            steps_per_generation=1,
            num_generations=args.num_generations,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_steps=args.max_steps,
            output_dir=str(output_dir),
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
            hub_private_repo=args.hub_private_repo,
            report_to=args.report_to,
            trackio_space_id=args.trackio_space_id,
            save_strategy="no" if args.max_steps <= 0 else "steps",
        ),
        environment_factory=SupportOpsToolEnv,
    )

    if args.dry_run:
        logger.info("--dry-run: trainer constructed OK, skipping trainer.train()")
        return

    trainer.train()
    trainer.save_model(str(output_dir))
    logger.info("training complete, artifacts in %s", output_dir)


if __name__ == "__main__":
    main()
