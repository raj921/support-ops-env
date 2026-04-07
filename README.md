---
title: Support Ops Env
emoji: 📬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
---

# Support Ops Env

`support_ops_env` is a real-world OpenEnv benchmark for SaaS support operations. It simulates the work an internal support operator actually does: inspect an inbox, triage the right case, assign the correct team, apply operational tags, merge duplicates, draft a safe customer reply, and submit the case for review.

The environment is designed for agent learning rather than one-shot static evaluation. Each episode exposes partial progress through the reward signal, tracks queue state through `step() / reset() / state()`, and scores final outcomes with deterministic task graders.

## Why judges should care

This benchmark targets a practical gap in agent evaluation. Many environments test planning in abstract settings; this one tests whether an agent can complete a recognizable business workflow with real operational tradeoffs. The agent must prioritize, route, avoid unsafe behavior, preserve customer trust, and finish within a constrained action budget.

Additional context:

- [docs/BENCHMARK_BRIEF.md](docs/BENCHMARK_BRIEF.md)
- [docs/GLOSSARY.md](docs/GLOSSARY.md)
- [docs/IMPLEMENTATION_CHECKLIST.md](docs/IMPLEMENTATION_CHECKLIST.md)

## Real-world task modeled

This environment models first-line SaaS customer support triage:

- decide which ticket matters most
- inspect the relevant ticket history
- assign the right internal team
- set an urgency level
- add the right routing tags
- merge duplicate tickets
- draft a safe, accurate reply
- submit the final case state

## OpenEnv API

The environment follows the standard OpenEnv simulation interface:

- `reset(task_id=..., seed=...) -> observation`
- `step(action) -> observation, reward, done, info`
- `state() -> current state`

Typed models are implemented with Pydantic:

- `SupportOpsAction`
- `SupportOpsObservation`
- `SupportOpsState`

Metadata is declared in [openenv.yaml](openenv.yaml).

## Action space

The action model is structured and intentionally narrow:

- `view_ticket`
- `set_priority`
- `assign_team`
- `add_tags`
- `set_status`
- `draft_reply`
- `merge_ticket`
- `submit`

Action fields include `ticket_id`, `target_ticket_id`, `priority`, `team`, `status`, `tags`, and `reply`.

## Observation space

Each observation includes:

- `task_id`, `task_title`, `objective`
- `inbox_summary`: compact queue view across all tickets
- `current_ticket_id`, `current_ticket_view`: focused ticket context
- `last_event`: feedback from the last action
- `progress_score`: current deterministic grader estimate in `[0.0, 1.0]`
- `remaining_steps`
- `recent_actions`
- `score_breakdown`, `penalty_breakdown`
- standard OpenEnv `reward`, `done`, `metadata`

## State space

`state()` returns the full visible queue state:

- all tickets with status, priority, assigned team, tags, draft reply, and merge target
- viewed ticket ids
- action history
- cumulative reward
- invalid-action and no-progress counts
- current grader score
- grader component and penalty breakdowns
- episode metadata

## Tasks

The environment ships with three deterministic tasks, graded from easy to hard.

### 1. Easy: VIP SSO lockout

The agent must correctly triage an enterprise admin who is locked out during a board-meeting deadline while ignoring benign distractors from the same account. The correct solution requires urgent priority, the access team, relevant tags, an escalation status, and a reply that shows empathy, ownership, speed, and a viable workaround.

### 2. Medium: Duplicate refund case

The agent must identify the real billing issue among distractors, inspect both relevant tickets, merge the finance duplicate into the main thread, route to billing, and communicate a realistic refund-review path without mishandling unrelated tickets or overpromising a refund outcome.

### 3. Hard: Phishing and credential exposure

The agent must respond to a likely spoofed-support incident with related follow-up context and same-company distractors. The correct reply must be security-safe: urgent routing to security, correct tagging, duplicate consolidation, containment guidance, evidence preservation, and no false claim of a confirmed breach.

## Reward design

The reward function emits dense partial progress:

- positive reward for newly correct views, routing fields, merges, and reply improvements
- low or zero reward for no-op and repeated actions
- score degradation for invalid actions, inefficient looping, touching irrelevant tickets, wrong merges, unsafe reply content, and destructive handling

Per-step reward is clamped to `[0.0, 1.0]`. Final task scores are deterministic and also clamped to `[0.0, 1.0]`.

## Graders

Each task uses a deterministic grader in [graders.py](graders.py). The grader checks:

- required ticket views
- field alignment: priority, team, status, tags
- merge correctness
- reply concept coverage through keyword sets
- forbidden reply phrases
- invalid-action and no-progress pressure
- efficiency and irrelevant-ticket penalties
- whether the agent explicitly submitted the case

## Setup

### Local Python

```bash
python3 -m venv .venv
. .venv/bin/activate
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

### Run the local submission checklist

```bash
scripts/validate-local.sh --skip-docker
```

Run the full script on a machine with Docker:

```bash
scripts/validate-local.sh
```

## Docker

Build and run locally:

```bash
docker build -t support-ops-env:latest .
docker run --rm -p 8000:8000 support-ops-env:latest
```

## Hugging Face Spaces deployment

This repository is ready for a Docker Space.

1. Create a new Hugging Face Space with SDK set to `Docker`.
2. Push this repository.
3. Add required secrets: `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`.
4. Optionally set `OPENAI_API_KEY` only as a local-development fallback.
5. Ensure the Space is tagged with `openenv`.
6. After deploy, verify:
   - `https://<space>.hf.space/health`
   - `POST https://<space>.hf.space/reset`
   - `openenv validate --url https://<space>.hf.space`

The container serves the OpenEnv HTTP API on port `8000`.

## Baseline inference

The required baseline script is [inference.py](inference.py). It:

- uses the OpenAI Python client
- reads `API_BASE_URL`, `MODEL_NAME`, and API credentials with precedence:
  `HF_TOKEN` first, then `OPENAI_API_KEY`
- runs all three benchmark tasks
- emits strict `[START]`, `[STEP]`, and `[END]` stdout records

Example:

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4.1-mini
export HF_TOKEN=...
python inference.py
```

The script defaults to the local Docker image `support-ops-env:latest`. Override with `IMAGE_NAME` if needed, or use `ENV_BASE_URL` to target an already-running local server.

## Baseline scores

Verified locally against the running server using the built-in deterministic fallback path in [inference.py](inference.py):

- easy: `0.9250`
- medium: `0.8500`
- hard: `0.8227`

When a real model endpoint is available, scores may differ, but the environment graders themselves remain deterministic.

## Project structure

- [models.py](models.py): typed action, observation, and state models
- [tasks.py](tasks.py): task fixtures and expectations
- [graders.py](graders.py): deterministic scoring logic
- [server/support_ops_environment.py](server/support_ops_environment.py): environment implementation
- [server/app.py](server/app.py): FastAPI/OpenEnv app
- [client.py](client.py): typed OpenEnv client
- [inference.py](inference.py): baseline runner
- [scripts/validate-local.sh](scripts/validate-local.sh): local submission checklist
