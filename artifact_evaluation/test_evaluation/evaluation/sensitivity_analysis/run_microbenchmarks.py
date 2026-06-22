from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from microbench_utils import (
    BASELINE_COLORS,
    BASELINE_KEYS,
    BASELINE_LABELS,
    collect_microbenchmark_matrix,
    tiny_matplotlib,
)

tiny_matplotlib()
import matplotlib.pyplot as plt


def plot(
    sizes_mb: list[int],
    data: dict[str, list[float]],
    output_file: Path,
    baseline_keys: list[str],
) -> None:
    labels = [BASELINE_LABELS[key] for key in baseline_keys]
    width = min(0.18, 0.8 / len(labels))
    x = np.arange(len(sizes_mb))

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for idx, label in enumerate(labels):
        ax.bar(
            x + width * idx,
            data[label],
            width,
            label=label,
            color=BASELINE_COLORS[label],
            align="edge",
        )

    ax.set_xticks(x + width * len(labels) / 2, labels=sizes_mb)
    ax.set_yscale("log", base=10)
    ax.set_ylabel("Checkpoint time (ms)")
    ax.set_xlabel("Checkpoint size (MB)")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight", dpi=150, pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate tiny Figure 11 using only microbenchmarks."
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 10, 100])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--baselines", nargs="+", choices=BASELINE_KEYS, default=BASELINE_KEYS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fakegpu", action="store_true")
    args = parser.parse_args()

    output_dir = Path.cwd()
    data = collect_microbenchmark_matrix(
        args.sizes,
        args.iterations,
        output_dir,
        baselines=args.baselines,
        num_threads=args.num_threads,
        force=args.force,
        fakegpu=args.fakegpu,
    )

    pd.DataFrame(
        [data[BASELINE_LABELS[key]] for key in args.baselines],
        index=[BASELINE_LABELS[key] for key in args.baselines],
        columns=[str(size) for size in args.sizes],
    ).to_csv("fig11_tiny.csv")
    plot(args.sizes, data, output_dir / "fig11_tiny.png", args.baselines)
    print("Generated fig11_tiny.csv and fig11_tiny.png")


if __name__ == "__main__":
    main()
