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


DEFAULT_CFREQS = [0, 1, 5, 10, 25]


def throughput_from_checkpoint_ms(
    checkpoint_ms: float,
    cfreq: int,
    base_iter_ms: float,
) -> float:
    if cfreq == 0:
        return 1000.0 / base_iter_ms
    return 1000.0 / (base_iter_ms + checkpoint_ms / cfreq)


def plot(
    cfreqs: list[int],
    data: dict[str, list[float]],
    output_file: Path,
    baseline_keys: list[str],
) -> None:
    labels = [BASELINE_LABELS[key] for key in baseline_keys]
    width = min(0.18, 0.8 / len(labels))
    x = np.arange(len(cfreqs[1:]))

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for idx, label in enumerate(labels):
        ax.bar(
            x + width * idx,
            data[label][1:],
            width,
            label=label,
            color=BASELINE_COLORS[label],
            align="edge",
        )

    if "PCcheck" in data:
        ax.plot(
            x + width * (len(labels) - 1),
            [data["PCcheck"][0]] * len(x),
            color="black",
            marker="s",
            linewidth=1,
            markersize=3,
        )
    ax.set_xticks(x + width * len(labels) / 2, labels=cfreqs[1:])
    ax.set_ylabel("Throughput (iter/sec)")
    ax.set_xlabel("Checkpoint interval")
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight", dpi=150, pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate tiny Figure 8 from microbenchmark measurements."
    )
    parser.add_argument("--size-mb", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--base-iter-ms", type=float, default=100.0)
    parser.add_argument("--cfreqs", type=int, nargs="+", default=DEFAULT_CFREQS)
    parser.add_argument("--baselines", nargs="+", choices=BASELINE_KEYS, default=BASELINE_KEYS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.cfreqs or args.cfreqs[0] != 0:
        raise ValueError("--cfreqs must start with 0 for the no-checkpoint baseline")

    output_dir = Path.cwd()
    measured = collect_microbenchmark_matrix(
        [args.size_mb],
        args.iterations,
        output_dir,
        baselines=args.baselines,
        num_threads=args.num_threads,
        force=args.force,
    )

    checkpoint_ms = {label: values[0] for label, values in measured.items()}
    throughput = {
        label: [
            throughput_from_checkpoint_ms(ms, cfreq, args.base_iter_ms)
            for cfreq in args.cfreqs
        ]
        for label, ms in checkpoint_ms.items()
    }

    pd.DataFrame(
        [throughput[BASELINE_LABELS[key]] for key in args.baselines],
        index=[BASELINE_LABELS[key] for key in args.baselines],
        columns=[str(cfreq) for cfreq in args.cfreqs],
    ).to_csv("fig8_tiny.csv")
    pd.DataFrame(
        [checkpoint_ms],
        columns=[BASELINE_LABELS[key] for key in args.baselines],
    ).to_csv("fig8_tiny_inputs.csv", index=False)

    plot(args.cfreqs, throughput, output_dir / "fig8_tiny.png", args.baselines)
    print("Generated fig8_tiny.csv and fig8_tiny.png")


if __name__ == "__main__":
    main()
