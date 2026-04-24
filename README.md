---
title: Support Ops Env
emoji: 📬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
  - rl
  - grpo
  - trl
  - support-ops
---

# Support Ops Env — A Verifiable RL Benchmark for SaaS Support Triage

> **OpenEnv Hackathon submission.** An environment + deterministic grader + GRPO training stack + reproducible demo for first-line SaaS support operators.

- **Space (environment, live)**: https://huggingface.co/spaces/raj23211/support-ops-env
- **Colab (Qwen3-4B-Instruct-2507 training)**: see `support_ops_colab.ipynb`
- **Kaggle (same Qwen GRPO flow)**: `support_ops_kaggle.ipynb` — enable **GPU + Internet**, secret `HF_TOKEN`, outputs under `/kaggle/working`. **CLI push:** folder `kernels/support-ops-grpo/` (`kernel-metadata.json` + notebook); run `kaggle kernels push -p kernels/support-ops-grpo` (see `kernels/support-ops-grpo/README.md`).
- **Colab (Gemma 4 training, experimental)**: see `support_ops_colab_gemma4.ipynb`

---

## TL;DR for judges

| What judges care about (guide §19) | Where it lives in this repo |
|------------------------------------|-----------------------------|
| **Clear environment design** | `models.py`, `server/support_ops_environment.py`, `client.py` |
| **Objective multi-component rewards** | `graders.py` → `investigation, routing, reply_quality, groundedness, submission` + 8 penalties |
| **Evidence the model improved** | `eval_compare.py` → baseline vs trained, component deltas, Markdown table |
| **Prevention against reward hacking** | `audit.py` → trajectory dump + flags: repeat spam, disallowed tools, forbidden phrases, reward-without-evidence |
| **Reproducible deployment** | Docker Space + `inference.py` + Colab notebooks |
| **Sharp demo** | `plot_rewards.py` (4-panel) + eval markdown + curriculum story |

---

## What the agent actually does

The agent is a first-line SaaS support operator across six apps — **Inbox, CRM, Billing, Access, Policy, Comms** — and must:

1. **Open** the relevant cases in the inbox
2. **Investigate** via CRM, billing, access, and policy tools
3. **Route** each real case (priority + team + status + tags)
4. **Merge** duplicate cases from the same customer / same incident
5. **Draft** one customer-facing reply that is grounded in facts the agent actually surfaced
6. **Submit** the final resolution

Every task ships **traps** (distractor cases from the same account, policy ambiguity, phishing overclaims, GDPR inside a churn thread, etc.) so reward hacking is structurally costly.

---

## OpenEnv API

Standard OpenEnv simulation interface.

```python
from support_ops_env import SupportOpsEnv, SupportOpsAction

env = SupportOpsEnv(base_url="https://raj23211-support-ops-env.hf.space").sync()
obs = env.reset(task_id="c1_access_lockout").observation

step = env.step(SupportOpsAction(
    assistant_message="Listing inbox to find the primary case.",
    tool_calls=[{"name": "inbox.list_cases", "args": {}}],
    answer=None,
))
```

- `reset(task_id=..., seed=...) -> observation`
- `step(action) -> observation, reward, done, info`
- `state() -> current state`
- Typed Pydantic models: `SupportOpsAction`, `SupportOpsObservation`, `SupportOpsState`
- Metadata in [`openenv.yaml`](openenv.yaml)

### Action space

`SupportOpsAction` is intentionally structured, not free-form:

- `assistant_message`: short one-line explanation of the turn
- `tool_calls`: list of `{name, args}` over ~20 tools across the 6 apps
- `answer`: `None` until the final turn; then the full resolution dict

### Observation space

- `task_id, task_title, collection, task_family, difficulty, objective`
- `app_summaries`: compact view of each of the 6 apps
- `tool_results`: surfaced facts + error statuses from the last batch of tools
- `progress_score` ∈ (0, 1) — deterministic grader estimate
- `remaining_steps, recent_actions, visible_case_ids, guidance`
- **`reward_breakdown`** — per-component scores
- **`penalty_breakdown`** — 8 penalty channels (see below)

---

## Tasks and curriculum (guide §6)

Eight deterministic tasks across four collections (C1 → C8), tagged by `difficulty`. The repo exposes a curriculum helper so RL training sees non-zero reward **early**:

| Difficulty | Tasks | Purpose |
|------------|-------|---------|
| `easy`     | `c1_access_lockout`, `c1_duplicate_billing` | Warmup, short horizon, clean signal |
| `medium`   | + `c2_security_phishing`, `c2_refund_policy_trap` | Safety traps (phishing, policy ambiguity) |
| `hard`     | + `c4_gdpr_churn`, `c4_export_before_churn`, `c4_renewal_risk_triage` | Multi-owner workflows |
| `expert`   | + `c8_same_account_trap` | Cross-app, cross-team same-account scenario |

```python
from support_ops_env import get_curriculum_task_ids
get_curriculum_task_ids("easy")     # just easy
get_curriculum_task_ids("medium")   # easy + medium
get_curriculum_task_ids("all")      # everything, easy first
```

Training (`train.py` + the Colab notebook) cycles through the selected curriculum per rollout — so you can start with `--difficulty easy`, confirm rewards move, then climb.

---

## Reward design (guide §7 — multi-component, independent checks)

`graders.py` emits **five dense components** (weighted sum → `progress_score`) and **eight independent penalties**.

### Components (weighted)

| Component | Weight | Measures |
|-----------|-------:|----------|
| `investigation` | 0.28 | Did the agent surface the right facts via the right tools? |
| `routing`       | 0.32 | Priority, team, status, tags for each expected case (and merges) |
| `reply_quality` | 0.20 | Concept coverage of the final customer reply |
| `groundedness`  | 0.12 | Reply claims only facts that were actually surfaced |
| `submission`    | 0.08 | Agent explicitly submitted a final resolution |

### Penalty channels (each subtracted from the score)

`unsupported_claim_penalty`, `unsafe_reply_penalty`, `repeat_penalty`, `irrelevant_penalty`, `invalid_action_penalty`, `no_progress_penalty`, `efficiency_penalty`, `disallowed_tool_penalty`.

All keys are surfaced to the agent on every `step` via `obs.reward_breakdown` and `obs.penalty_breakdown`, so GRPO sees **per-skill signal**, not a single scalar.

---

## Training (guide §11 RLVR + §14 stability first)

Two training paths ship in the repo.

### A. Qwen path (recommended, stable) — `train.py` + `support_ops_colab.ipynb`

- **Default model**: `Qwen/Qwen3-4B-Instruct-2507` (dense 4B, non-thinking instruct)
- **Quantization**: 4-bit NF4 + bf16 compute on T4; bf16 on A100+
- **LoRA**: full surface (`q/k/v/o/gate/up/down`), adapter-only saves
- **vLLM**: off by default (doesn't fit alongside a 4B KV cache on T4); opt-in via `--use-vllm`

```bash
# local loop (T4 or A100)
python -m server.app --port 8000
python train.py --env-url http://localhost:8000 \
                --difficulty easy \
                --load-in-4bit
```

Rationale for Qwen3-4B-Instruct-2507 (as of Apr 2026):

| Concern | Qwen3.5-4B today | Qwen3-4B-Instruct-2507 |
|--------|------------------|------------------------|
| TRL + vLLM weight-prefix bug | **Open** (text-only checkpoints) | **No issue** |
| 4-bit QLoRA recommended | **No** (high quant error) | **Yes** |
| Thinking by default | **Yes** (parser work) | **No** (clean JSON) |

### A2. Qwen3.5-4B via Unsloth (fast path, opt-in) — `--use-unsloth`

Unsloth's loader sidesteps the two Qwen3.5 blockers above: it uses its own
generation (so the vLLM weight-prefix bug doesn't matter) and runs **bf16
LoRA** (~10 GB on Qwen3.5-4B, fits T4). Per the official docs, it is ~1.5×
faster and ~50 % less VRAM than FA2 on the same hardware.

Install:

```bash
pip install -e '.[unsloth]'          # local
# or in Colab: run the "1b. (Optional) Install Unsloth" cell
```

Run locally:

```bash
python train.py --env-url http://localhost:8000 \
                --difficulty easy \
                --use-unsloth          # model defaults to unsloth/Qwen3.5-4B
```

In the Colab notebook (`support_ops_colab.ipynb`) flip `USE_UNSLOTH = True` in
Section 2 and run the `1b` install cell before the trainer setup cell. The
notebook branches automatically:

| Toggle | Model | Loader | Quant | vLLM |
|--------|-------|--------|-------|------|
| `USE_UNSLOTH=False` (default) | `Qwen/Qwen3-4B-Instruct-2507` | HF `transformers` | NF4 + bf16 | off (opt-in) |
| `USE_UNSLOTH=True`             | `unsloth/Qwen3.5-4B`          | `FastLanguageModel` | bf16 LoRA  | forced off (Unsloth generation) |

### B. Gemma 4 path (experimental) — `train_gemma4.py` + `support_ops_colab_gemma4.ipynb`

Uses TRL's `environment_factory` with `google/gemma-4-E2B-it` in 4-bit. Good for agent benchmarks, but known-fragile stack (BF16/FP16, `AutoModelForImageTextToText`, thinking tokens). Kept as an optional track.

---

## Demo flow (guide §19)

Four artifacts, in order:

1. **Baseline rollout** (Space + base model)
   ```bash
   python eval_compare.py --env-url https://raj23211-support-ops-env.hf.space \
                          --difficulty easy --episodes 1
   ```
2. **Train on curriculum** (see `support_ops_colab.ipynb`)
3. **Baseline vs trained evaluation**
   ```bash
   python eval_compare.py --env-url ... \
                          --adapter-path outputs/support-ops-grpo-.../ \
                          --difficulty easy --episodes 2 \
                          --output-dir eval_runs/demo
   # -> eval_runs/demo/eval_results.md  (the table judges actually read)
   ```
4. **Audit — proof the trained agent is not reward-hacking**
   ```bash
   python audit.py --env-url ... \
                   --adapter-path outputs/support-ops-grpo-.../ \
                   --difficulty all --episodes 1 \
                   --output-dir audit_runs/demo
   # -> audit_runs/demo/audit_report.md
   ```

Reward curves (4-panel total / components / penalty / pass-rate):

```bash
python plot_rewards.py outputs/support-ops-grpo-*/reward_log.csv
```

---

## Safeguards against reward hacking (guide §8)

- **Multi-component grader** — five independent component rewards and eight penalty channels; gaming one without the others tanks the overall score.
- **Required evidence** — reply grounding requires that facts be actually surfaced; the grader penalizes claims whose `fact_id` was not returned by a tool.
- **Forbidden phrases + disallowed tools** — each task declares its own lists; direct, hard-coded penalties on use.
- **Repeat & no-progress penalties** — shaped to stop loops cheaply.
- **`audit.py`** — dumps full trajectories plus these flags:
  - `repeat_spam`
  - `disallowed_tool_use`
  - `forbidden_reply_phrase`
  - `missed_required_evidence`
  - `reward_without_investigation`
  - `reward_without_submission`
  - `no_tool_calls_at_all`
- **Adapter-only saves** — no naive 4-bit→16-bit upcast + merge (guide §16).

---

## Monitoring (guide §15)

`train.py` writes `reward_log.csv` with one row per episode, extended schema:

```
episode, task_id,
total_reward,
investigation, routing, reply_quality, groundedness, submission,
penalty_total,
timestamp
```

`plot_rewards.py` produces four stacked panels:

1. Total reward + rolling mean + pass threshold (0.5)
2. All five components (rolling mean)
3. Penalty total (lower is better)
4. Rolling pass rate

Per-task performance and penalty drift are both visible at a glance.

---

## Setup

### Local Python

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
pre-commit install
```

### Run the server locally

```bash
python -m server.app --port 8000
```

### Validate local structure

```bash
openenv validate .
```

### Validate a running server

```bash
openenv validate --url http://localhost:8000
```

---

## Docker

```bash
docker build -t support-ops-env:latest .
docker run --rm -p 8000:8000 support-ops-env:latest
```

---

## Hugging Face Spaces deployment

This repository ships as a Docker Space.

1. Create a new HF Space with SDK = `Docker`.
2. Push this repo.
3. Required secrets for the baseline runner: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`.
4. Ensure the Space has the `openenv` tag.
5. After deploy:
   - `GET  https://<space>.hf.space/health`
   - `POST https://<space>.hf.space/reset`
   - `openenv validate --url https://<space>.hf.space`

---

## Baseline inference (deterministic reference)

`inference.py` is the required OpenEnv baseline. It uses the OpenAI Python client, reads `API_BASE_URL` + `MODEL_NAME` + `HF_TOKEN` (or `OPENAI_API_KEY`), and runs all tasks with strict `[START]`, `[STEP]`, `[END]` output.

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4.1-mini
export HF_TOKEN=...
python inference.py
```

Reference deterministic-fallback scores in the in-process runner:

| Collection | Task | Score |
|------------|------|------:|
| C1 | c1_access_lockout       | 0.925 |
| C1 | c1_duplicate_billing    | 0.850 |
| C2 | c2_security_phishing    | 0.823 |
| C4 | c4_gdpr_churn           | 0.823 |

---

## Project structure

| File | Role |
|------|------|
| `models.py` | Typed action / observation / state models |
| `tasks.py` | 8 deterministic tasks + curriculum helpers |
| `graders.py` | Deterministic 5-component grader + 8 penalties |
| `server/support_ops_environment.py` | Env implementation |
| `server/app.py` | FastAPI / OpenEnv app |
| `client.py` | Typed sync/async OpenEnv client |
| `inference.py` | Baseline runner |
| `train.py` | Qwen GRPO training (default path) |
| `train_gemma4.py` | Gemma 4 GRPO training (experimental `environment_factory`) |
| `eval_compare.py` | **Baseline vs trained** eval → JSON + Markdown |
| `audit.py` | **Reward-hack audit** → flags + trajectory dumps |
| `plot_rewards.py` | 4-panel reward curves (total / components / penalty / pass) |
| `support_ops_colab.ipynb` | Colab: Qwen3-4B-Instruct-2507 + GRPO |
| `support_ops_colab_gemma4.ipynb` | Colab: Gemma 4 + GRPO (experimental) |

---

## Alignment with the hackathon self-serve guide

| Guide rule | Where in repo |
|------------|---------------|
| §1 verifiable task, success prob > 0 | Curriculum + dense grader |
| §4–§5 env first, OpenEnv API | `server/`, `models.py`, `client.py` |
| §6 easy → hard curriculum | `get_curriculum_task_ids`, `--difficulty` |
| §7 multiple reward components | `graders.py` (5 components + 8 penalties) |
| §8 anti-reward-hacking | `audit.py`, penalty channels |
| §11 GRPO + RLVR | `train.py` + `train_gemma4.py` |
| §13 deploy early | HF Space live |
| §14 stability before scale | Qwen3-4B-Instruct-2507 default; `use_vllm` opt-in |
| §15 monitor many things | `reward_log.csv` v2 + `plot_rewards.py` 4-panel |
| §16 save correctly | LoRA adapter saves, no naive merge |
| §19 judge demo format | `eval_compare.py` + `audit.py` |

---

## Acknowledgements

- Hugging Face [OpenEnv](https://github.com/huggingface/openenv) team
- [TRL](https://github.com/huggingface/trl) GRPO trainer
- [PEFT](https://github.com/huggingface/peft) LoRA adapters
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) 4-bit quantization
