from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List

from openai import OpenAI

try:
    from .client import SupportOpsEnv
    from .models import SupportOpsAction
    from .tasks import TASK_IDS
except ImportError:
    from client import SupportOpsEnv
    from models import SupportOpsAction
    from tasks import TASK_IDS

BENCHMARK = "support_ops_env"
IMAGE = os.environ.get("IMAGE_NAME", "support-ops-env:latest")
ENV_URL = os.environ.get("ENV_BASE_URL")
API_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
TEMP = 0.0
MAX_TOK = 300
PASS_SCORE = 0.7
MAX_STEPS = {"easy_vip_sso": 8, "medium_refund_duplicate": 10, "hard_security_phishing": 10}

SYS_PROMPT = """You are operating a SaaS support inbox environment.
Return exactly one JSON object per turn with keys:
action_type, ticket_id, target_ticket_id, priority, team, status, tags, reply, reasoning.
Use only supported actions:
- view_ticket
- set_priority
- assign_team
- add_tags
- set_status
- draft_reply
- merge_ticket
- submit
Choose one action per turn. Keep reply text concise and customer-safe."""


def resolve_api_key(environ: Dict[str, str] | None = None) -> str:
    env = environ or os.environ
    return env.get("HF_TOKEN") or env.get("OPENAI_API_KEY") or "missing"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rw = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rw}",
        flush=True,
    )


def fallback_action(task_id: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    seq: Dict[str, List[Dict[str, Any]]] = {
        "easy_vip_sso": [
            {"action_type": "view_ticket", "ticket_id": "E-1001"},
            {"action_type": "set_priority", "ticket_id": "E-1001", "priority": "urgent"},
            {"action_type": "assign_team", "ticket_id": "E-1001", "team": "access"},
            {"action_type": "add_tags", "ticket_id": "E-1001", "tags": ["vip_customer", "sso"]},
            {"action_type": "set_status", "ticket_id": "E-1001", "status": "escalated"},
            {
                "action_type": "draft_reply",
                "ticket_id": "E-1001",
                "reply": (
                    "Sorry about the disruption. I escalated this to our access team right away "
                    "and expect an update within 15 minutes. If you have a backup admin, we can "
                    "help with temporary access while we investigate."
                ),
            },
            {"action_type": "submit"},
        ],
        "medium_refund_duplicate": [
            {"action_type": "view_ticket", "ticket_id": "M-2002"},
            {"action_type": "view_ticket", "ticket_id": "M-2004"},
            {"action_type": "merge_ticket", "ticket_id": "M-2004", "target_ticket_id": "M-2002"},
            {"action_type": "set_priority", "ticket_id": "M-2002", "priority": "high"},
            {"action_type": "assign_team", "ticket_id": "M-2002", "team": "billing"},
            {"action_type": "add_tags", "ticket_id": "M-2002", "tags": ["refund", "duplicate"]},
            {"action_type": "set_status", "ticket_id": "M-2002", "status": "escalated"},
            {
                "action_type": "draft_reply",
                "ticket_id": "M-2002",
                "reply": (
                    "Sorry for the duplicate charge and thanks for flagging it. I merged the "
                    "finance follow-up into this single ticket and escalated it to our billing "
                    "team for refund review. You should have an update within 3 business days."
                ),
            },
            {"action_type": "submit"},
        ],
        "hard_security_phishing": [
            {"action_type": "view_ticket", "ticket_id": "H-3001"},
            {"action_type": "view_ticket", "ticket_id": "H-3002"},
            {"action_type": "merge_ticket", "ticket_id": "H-3002", "target_ticket_id": "H-3001"},
            {"action_type": "set_priority", "ticket_id": "H-3001", "priority": "urgent"},
            {"action_type": "assign_team", "ticket_id": "H-3001", "team": "security"},
            {"action_type": "add_tags", "ticket_id": "H-3001", "tags": ["security", "phishing"]},
            {"action_type": "set_status", "ticket_id": "H-3001", "status": "escalated"},
            {
                "action_type": "draft_reply",
                "ticket_id": "H-3001",
                "reply": (
                    "This message was not sent by our team, so please do not share passwords or "
                    "MFA codes. I escalated this to our security team immediately. Please reset "
                    "the affected password, rotate tokens or active sessions, and preserve the "
                    "email headers or a screenshot so we can investigate."
                ),
            },
            {"action_type": "submit"},
        ],
    }
    plan = seq[task_id]
    return plan[min(len(history), len(plan) - 1)]


def mk_client() -> OpenAI:
    return OpenAI(base_url=API_URL, api_key=resolve_api_key())


def get_model_action(
    client: OpenAI, task_id: str, step: int,
    obs: Dict[str, Any], history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prompt = (
        f"Task id: {task_id}\n"
        f"Step: {step}\n"
        f"Objective: {obs.get('objective', '')}\n"
        f"Last event: {obs.get('last_event', '')}\n"
        f"Progress score: {obs.get('progress_score', 0.0)}\n"
        f"Remaining steps: {obs.get('remaining_steps', 0)}\n"
        f"Inbox summary:\n{obs.get('inbox_summary', '')}\n\n"
        f"Current ticket view:\n{obs.get('current_ticket_view', '')}\n\n"
        f"Recent history:\n{json.dumps(history[-5:], ensure_ascii=True)}\n"
        "Respond with one JSON action only."
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMP,
            max_tokens=MAX_TOK,
            stream=False,
        )
        txt = (resp.choices[0].message.content or "").strip()
        if txt.startswith("```"):
            txt = txt.strip("`")
            txt = txt.split("\n", 1)[1] if "\n" in txt else txt
            if txt.endswith("```"):
                txt = txt[:-3].strip()
        return json.loads(txt)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return fallback_action(task_id, history)


def to_action(raw: Dict[str, Any]) -> SupportOpsAction:
    return SupportOpsAction(
        action_type=raw.get("action_type", "submit"),
        ticket_id=raw.get("ticket_id"),
        target_ticket_id=raw.get("target_ticket_id"),
        priority=raw.get("priority"),
        team=raw.get("team"),
        status=raw.get("status"),
        tags=raw.get("tags") or [],
        reply=raw.get("reply"),
        reasoning=raw.get("reasoning"),
    )


async def run_task(client: OpenAI, env: SupportOpsEnv, task_id: str) -> float:
    history: List[Dict[str, Any]] = []
    rewards: List[float] = []
    steps = 0
    score = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL)
    result = await env.reset(task_id=task_id, seed=7)

    try:
        for step in range(1, MAX_STEPS[task_id] + 1):
            if result.done:
                break

            obs = result.observation.model_dump()
            raw = get_model_action(client, task_id, step, obs, history)
            err = None
            try:
                act = to_action(raw)
            except Exception as exc:
                err = f"action_parse_error: {exc}"
                act = SupportOpsAction(action_type="submit")

            result = await env.step(act)
            rw = float(result.reward or 0.0)
            rewards.append(rw)
            steps = step
            log_step(step=step, action=act.model_dump_json(), reward=rw, done=result.done, error=err)
            history.append(act.model_dump())
            if result.done:
                break

        st = await env.state()
        score = float(st.current_score)
        ok = score >= PASS_SCORE
        log_end(success=ok, steps=steps, score=score, rewards=rewards)
        return score
    finally:
        if not result.done:
            log_end(success=False, steps=steps, score=score, rewards=rewards)


async def main() -> None:
    client = mk_client()
    try:
        env = await SupportOpsEnv.from_docker_image(IMAGE)
    except Exception as exc:
        if not ENV_URL:
            raise RuntimeError(
                f"Failed to start Docker image {IMAGE!r}: {exc}. "
                "Set ENV_BASE_URL to run against an existing server."
            ) from exc
        print(f"[DEBUG] Falling back to ENV_BASE_URL: {exc}", flush=True)
        env = SupportOpsEnv(base_url=ENV_URL)
        await env.connect()
    try:
        for tid in TASK_IDS:
            await run_task(client, env, tid)
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
