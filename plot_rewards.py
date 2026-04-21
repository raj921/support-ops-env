#!/usr/bin/env python3
"""
Plot Support Ops Env training rewards from a CSV log.

Usage (while training is running or after)::

    python plot_rewards.py                              # auto-find latest reward_log.csv
    python plot_rewards.py outputs/*/reward_log.csv     # specific file
    python plot_rewards.py --live                       # refresh every 30s
    python plot_rewards.py --table                      # ASCII table, no matplotlib

Columns expected in ``reward_log.csv``:

    episode, total_reward, field_reward, reply_reward, grounding_reward, timestamp
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple


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


def load_csv(
    path: Path,
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    episodes: List[int] = []
    total: List[float] = []
    field: List[float] = []
    reply: List[float] = []
    ground: List[float] = []
    with open(path) as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue
            episodes.append(int(row[0]))
            total.append(float(row[1]))
            field.append(float(row[2]))
            reply.append(float(row[3]))
            ground.append(float(row[4]))
    return episodes, total, field, reply, ground


def rolling_avg(values: List[float], window: int = 10) -> List[float]:
    window = min(window, len(values))
    return [
        sum(values[max(0, i - window):i + 1]) / min(i + 1, window)
        for i in range(len(values))
    ]


def plot(path: Path, save_path: Optional[Path] = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes, total, field, reply, ground = load_csv(path)
    if not episodes:
        print("No data yet.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    window = min(10, len(episodes))

    ax1.plot(episodes, total, alpha=0.3, color="#1f77b4", label="per episode")
    ax1.plot(
        episodes,
        rolling_avg(total, window),
        color="#1f77b4",
        linewidth=2,
        label=f"rolling({window})",
    )
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Total reward")
    ax1.set_title(f"Support Ops Env — GRPO reward ({len(episodes)} episodes)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, rolling_avg(field, window), color="#ff7f0e", linewidth=2, label="field alignment")
    ax2.plot(episodes, rolling_avg(reply, window), color="#2ca02c", linewidth=2, label="reply quality")
    ax2.plot(episodes, rolling_avg(ground, window), color="#9467bd", linewidth=2, label="grounding")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Component reward")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_path or path.with_suffix(".png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved to {out}")

    print(f"\nEpisodes      : {len(episodes)}")
    print(f"Latest reward : {total[-1]:+.3f}")
    print(f"Avg (last 10) : {sum(total[-10:]) / min(10, len(total)):+.3f}")
    print(f"Best reward   : {max(total):+.3f}")
    print(f"Worst reward  : {min(total):+.3f}")


def print_table(path: Path) -> None:
    episodes, total, field, reply, ground = load_csv(path)
    if not episodes:
        print("No data yet.")
        return

    print(f"\n{'Ep':>4} | {'Total':>7} | {'Field':>6} | {'Reply':>6} | {'Ground':>6} | {'Avg(10)':>8}")
    print("-" * 60)
    for i in range(len(episodes)):
        avg10 = sum(total[max(0, i - 9):i + 1]) / min(i + 1, 10)
        marker = " *" if total[i] == max(total[:i + 1]) else ""
        print(
            f"{episodes[i]:>4} | {total[i]:>+7.2f} | {field[i]:>+6.2f} | "
            f"{reply[i]:>+6.2f} | {ground[i]:>+6.2f} | {avg10:>+8.2f}{marker}"
        )

    best_idx = total.index(max(total))
    print(f"\nBest : {max(total):+.3f}  (ep {episodes[best_idx]})")
    print(f"Avg  : {sum(total) / len(total):+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Support Ops GRPO rewards")
    parser.add_argument("csv_path", nargs="?", help="Path to reward_log.csv")
    parser.add_argument("--live", action="store_true", help="Refresh every 30s")
    parser.add_argument("--table", action="store_true", help="Print ASCII table instead of plot")
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
                plot(path, Path(args.out) if args.out else None)
                time.sleep(30)
            except KeyboardInterrupt:
                break
    else:
        plot(path, Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
