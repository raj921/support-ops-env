#!/usr/bin/env python3
"""
Plot Support Ops Env training rewards from a CSV log.

Usage (while training is running or after)::

    python plot_rewards.py                              # auto-find latest reward_log.csv
    python plot_rewards.py outputs/*/reward_log.csv     # specific file
    python plot_rewards.py --live                       # refresh every 30s
    python plot_rewards.py --table                      # ASCII table, no matplotlib

Supports **two CSV schemas** (auto-detected from the header row):

* **Legacy (v1)** — 6 columns::

    episode, total_reward, field_reward, reply_reward, grounding_reward, timestamp

* **Extended (v2, current)** — 10 columns::

    episode, task_id,
    total_reward,
    investigation, routing, reply_quality, groundedness, submission,
    penalty_total, timestamp

The extended schema unlocks four stacked panels (total / components /
penalty / rolling success-rate), which is what the guide asks for in §15:
don't just watch the overall reward — watch each column.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ----------------------------------------------------------------------
# CSV loading
# ----------------------------------------------------------------------

@dataclass
class RewardLog:
    schema: str  # "v1" or "v2"
    episodes: List[int] = field(default_factory=list)
    task_ids: List[str] = field(default_factory=list)
    total: List[float] = field(default_factory=list)
    investigation: List[float] = field(default_factory=list)
    routing: List[float] = field(default_factory=list)
    reply: List[float] = field(default_factory=list)
    grounding: List[float] = field(default_factory=list)
    submission: List[float] = field(default_factory=list)
    penalty: List[float] = field(default_factory=list)

    def n(self) -> int:
        return len(self.episodes)


def find_latest_csv() -> Optional[Path]:
    csvs = sorted(
        Path("outputs").glob("*/reward_log.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if csvs:
        return csvs[0]
    if Path("reward_log.csv").exists():
        return Path("reward_log.csv")
    return None


def load_csv(path: Path) -> RewardLog:
    with open(path) as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return RewardLog(schema="v1")
        # v2 has "task_id" as the 2nd column.
        is_v2 = len(header) >= 10 and header[1] == "task_id"
        log = RewardLog(schema="v2" if is_v2 else "v1")
        for row in reader:
            if is_v2 and len(row) >= 10:
                log.episodes.append(int(row[0]))
                log.task_ids.append(row[1])
                log.total.append(float(row[2]))
                log.investigation.append(float(row[3]))
                log.routing.append(float(row[4]))
                log.reply.append(float(row[5]))
                log.grounding.append(float(row[6]))
                log.submission.append(float(row[7]))
                log.penalty.append(float(row[8]))
            elif not is_v2 and len(row) >= 5:
                # v1: episode, total, field, reply, grounding, timestamp
                log.episodes.append(int(row[0]))
                log.task_ids.append("")
                log.total.append(float(row[1]))
                log.investigation.append(0.0)
                log.routing.append(float(row[2]))
                log.reply.append(float(row[3]))
                log.grounding.append(float(row[4]))
                log.submission.append(0.0)
                log.penalty.append(0.0)
        return log


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def rolling_avg(values: List[float], window: int) -> List[float]:
    window = max(1, min(window, len(values)))
    return [
        sum(values[max(0, i - window + 1):i + 1]) / min(i + 1, window)
        for i in range(len(values))
    ]


def rolling_success_rate(totals: List[float], window: int, threshold: float = 0.5) -> List[float]:
    window = max(1, min(window, len(totals)))
    out: List[float] = []
    for i in range(len(totals)):
        chunk = totals[max(0, i - window + 1):i + 1]
        out.append(sum(1 for v in chunk if v >= threshold) / max(1, len(chunk)))
    return out


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------

def plot(path: Path, save_path: Optional[Path] = None, window: int = 10) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = load_csv(path)
    if log.n() == 0:
        print("No data yet.")
        return

    eps = log.episodes
    w = min(window, log.n())

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    ax_total, ax_comp, ax_pen, ax_succ = axes

    # --- Total ---
    ax_total.plot(eps, log.total, alpha=0.25, color="#1f77b4", label="per episode")
    ax_total.plot(eps, rolling_avg(log.total, w), color="#1f77b4", linewidth=2,
                  label=f"rolling({w})")
    ax_total.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax_total.axhline(y=0.5, color="green", linestyle=":", alpha=0.5, label="pass threshold")
    ax_total.set_ylabel("Total reward")
    ax_total.set_title(f"Support Ops Env — GRPO training ({log.n()} episodes, schema={log.schema})")
    ax_total.legend(loc="lower right")
    ax_total.grid(True, alpha=0.3)

    # --- Components ---
    ax_comp.plot(eps, rolling_avg(log.investigation, w), linewidth=2, label="investigation", color="#d62728")
    ax_comp.plot(eps, rolling_avg(log.routing, w), linewidth=2, label="routing", color="#ff7f0e")
    ax_comp.plot(eps, rolling_avg(log.reply, w), linewidth=2, label="reply_quality", color="#2ca02c")
    ax_comp.plot(eps, rolling_avg(log.grounding, w), linewidth=2, label="groundedness", color="#9467bd")
    ax_comp.plot(eps, rolling_avg(log.submission, w), linewidth=2, label="submission", color="#17becf")
    ax_comp.set_ylabel("Component reward")
    ax_comp.legend(loc="lower right", ncol=3)
    ax_comp.grid(True, alpha=0.3)

    # --- Penalties (lower is better) ---
    ax_pen.plot(eps, log.penalty, alpha=0.25, color="#8c564b")
    ax_pen.plot(eps, rolling_avg(log.penalty, w), linewidth=2, color="#8c564b",
                label=f"penalty total rolling({w})")
    ax_pen.set_ylabel("Penalty total (↓)")
    ax_pen.legend(loc="upper right")
    ax_pen.grid(True, alpha=0.3)

    # --- Rolling success rate ---
    ax_succ.plot(eps, rolling_success_rate(log.total, w, threshold=0.5),
                 linewidth=2, color="#2ca02c", label=f"rolling({w}) pass rate (total≥0.5)")
    ax_succ.set_xlabel("Episode")
    ax_succ.set_ylabel("Pass rate")
    ax_succ.set_ylim(-0.05, 1.05)
    ax_succ.legend(loc="lower right")
    ax_succ.grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_path or path.with_suffix(".png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved to {out}")

    print(f"\nSchema        : {log.schema}")
    print(f"Episodes      : {log.n()}")
    print(f"Latest reward : {log.total[-1]:+.3f}")
    print(f"Avg (last 10) : {sum(log.total[-10:]) / min(10, log.n()):+.3f}")
    print(f"Best reward   : {max(log.total):+.3f}")
    print(f"Pass rate (≥0.5): {sum(1 for v in log.total if v >= 0.5) / log.n():.1%}")
    if any(log.penalty):
        print(f"Avg penalty   : {sum(log.penalty) / log.n():.3f}")


# ----------------------------------------------------------------------
# ASCII table
# ----------------------------------------------------------------------

def print_table(path: Path) -> None:
    log = load_csv(path)
    if log.n() == 0:
        print("No data yet.")
        return

    print(
        f"\n{'Ep':>4} | {'Task':<22} | {'Total':>7} | {'Inv':>6} | "
        f"{'Rout':>6} | {'Reply':>6} | {'Gnd':>6} | {'Sub':>6} | {'Pen':>6} | {'Avg10':>7}"
    )
    print("-" * 112)
    for i in range(log.n()):
        avg10 = sum(log.total[max(0, i - 9):i + 1]) / min(i + 1, 10)
        marker = " *" if log.total[i] == max(log.total[:i + 1]) else ""
        tid = (log.task_ids[i] or "-")[:22]
        print(
            f"{log.episodes[i]:>4} | {tid:<22} | {log.total[i]:>+7.2f} | "
            f"{log.investigation[i]:>+6.2f} | {log.routing[i]:>+6.2f} | "
            f"{log.reply[i]:>+6.2f} | {log.grounding[i]:>+6.2f} | "
            f"{log.submission[i]:>+6.2f} | {log.penalty[i]:>+6.2f} | "
            f"{avg10:>+7.2f}{marker}"
        )

    best_idx = log.total.index(max(log.total))
    print(f"\nBest   : {max(log.total):+.3f}  (ep {log.episodes[best_idx]}"
          f", task={log.task_ids[best_idx] or 'n/a'})")
    print(f"Avg    : {sum(log.total) / log.n():+.3f}")
    passes = sum(1 for v in log.total if v >= 0.5)
    print(f"Pass   : {passes}/{log.n()} ({passes / log.n():.1%})")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Support Ops GRPO rewards")
    parser.add_argument("csv_path", nargs="?", help="Path to reward_log.csv")
    parser.add_argument("--live", action="store_true", help="Refresh every 30s")
    parser.add_argument("--table", action="store_true", help="Print ASCII table instead of plot")
    parser.add_argument("--window", type=int, default=10, help="Rolling-window size")
    parser.add_argument("--out", default=None, help="Output image path")
    args = parser.parse_args()

    path = Path(args.csv_path) if args.csv_path else find_latest_csv()
    if not path or not path.exists():
        print("No reward_log.csv found. Run training first or specify path.")
        sys.exit(1)

    print(f"Reading: {path}")

    if args.table:
        print_table(path)
        return

    if args.live:
        while True:
            try:
                plot(path, Path(args.out) if args.out else None, window=args.window)
                time.sleep(30)
            except KeyboardInterrupt:
                break
    else:
        plot(path, Path(args.out) if args.out else None, window=args.window)


if __name__ == "__main__":
    main()
