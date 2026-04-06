# Support Ops Env Benchmark Brief

## What this benchmark is

`support_ops_env` evaluates how well an agent handles realistic SaaS support triage under operational constraints. It is not a toy environment. The agent must inspect inbox items, determine which ticket matters, route ownership correctly, apply workflow updates, consolidate duplicates, and write a customer-safe response.

## Target user

- Agent researchers evaluating multi-step reasoning in business operations
- Hackathon judges looking for real-world utility and deterministic grading
- Applied AI teams building support or operations copilots

## Why this is useful

Support triage sits in an important gap between toy benchmarks and full production systems. It requires prioritization under limited context, policy-aware routing, duplicate handling, safe external communication, and multi-step stateful interaction.

## Rubric alignment

### Real-world utility

The benchmark models a real first-line support workflow common to SaaS businesses.

### Task and grader quality

There are three tasks with deterministic graders from easy to hard, each scored in `[0.0, 1.0]`.

### Environment design

The environment exposes meaningful state transitions, dense partial-progress reward, and sensible episode boundaries.

### Code quality and spec compliance

The repo includes typed models, `openenv.yaml`, validation tooling, tests, Docker packaging, and Hugging Face deployment guidance.

### Creativity and novelty

The environment focuses on support operations instead of games or toy chat loops, with security- and billing-aware task mechanics.
