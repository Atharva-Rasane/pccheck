from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from microbench_utils import (
    BASELINE_COLORS,
    parse_microbenchmark_time,
    run_microbenchmark,
    tiny_matplotlib,
)

tiny_matplotlib()
import matplotlib.pyplot as plt


DEFAULT_CFREQS = [0, 1, 5, 10, 25]
DEFAULT_ASYNC = [1, 2, 4]


def slowdown_from_microbenchmark(
    checkpoint_ms: float,
    cfreq: int,
    max_async: int,
    base_iter_ms: float,
) -> float:
    if cfreq == 0:
        return 1.0
    visible_checkpoint_ms = checkpoint_ms / max_async
    return (base_iter_ms + visible_checkpoint_ms / cfreq) / base_iter_ms


def plot(cfreqs: list[int], data: dict[int, list[float]], output_file: Path) -> None:
    width = 0.18
    x = np.arange(len(cfreqs[1:]))

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for idx, (max_async, values) in enumerate(data.items()):
        ax.bar(
            x + width * idx,
            values[1:],
            width,
            label=f"{max_async} async",
            align="edge",
        )

    ax.set_xticks(x + width * len(data) / 2, labels=cfreqs[1:])
    ax.set_yscale("log", base=10)
    ax.set_ylabel("Slowdown over no checkpointing")
    ax.set_xlabel("Checkpoint interval")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight", dpi=150, pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate tiny Figure 12 from a PCcheck microbenchmark."
    )
    parser.add_argument("--size-mb", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--base-iter-ms", type=float, default=100.0)
    parser.add_argument("--cfreqs", type=int, nargs="+", default=DEFAULT_CFREQS)
    parser.add_argument("--max-async", type=int, nargs="+", default=DEFAULT_ASYNC)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.cfreqs or args.cfreqs[0] != 0:
        raise ValueError("--cfreqs must start with 0 for the no-checkpoint baseline")

    output_dir = Path.cwd()
    log_file = run_microbenchmark(
        "pccheck",
        args.size_mb,
        args.iterations,
        output_dir,
        num_threads=args.num_threads,
        force=args.force,
        tag="async",
    )
    checkpoint_ms = parse_microbenchmark_time(log_file, args.iterations)

    data = {
        max_async: [
            slowdown_from_microbenchmark(
                checkpoint_ms,
                cfreq,
                max_async,
                args.base_iter_ms,
            )
            for cfreq in args.cfreqs
        ]
        for max_async in args.max_async
    }

    pd.DataFrame(
        [data[max_async] for max_async in args.max_async],
        index=[str(max_async) for max_async in args.max_async],
        columns=[str(cfreq) for cfreq in args.cfreqs],
    ).to_csv("fig12_tiny.csv")
    plot(args.cfreqs, data, output_dir / "fig12_tiny.png")
    print("Generated fig12_tiny.csv and fig12_tiny.png")


if __name__ == "__main__":
    main()
