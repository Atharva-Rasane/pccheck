from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from microbench_utils import BASELINE_COLORS, tiny_matplotlib

tiny_matplotlib()
import matplotlib.pyplot as plt


SUPPORTED_BASELINES = ["CheckFreq", "GPM", "Gemini", "PCcheck"]
N_PCCHECK_TINY = 2


def redo_time_sec(
    baseline: str,
    cfreq: int,
    time_no_checkpoint: float,
    loading_time: float,
    tw_pccheck: float,
) -> float:
    if baseline in ["CheckFreq", "Gemini"]:
        return cfreq * time_no_checkpoint + loading_time
    if baseline == "GPM":
        return cfreq * time_no_checkpoint / 2.0 + loading_time
    if baseline == "PCcheck":
        return (
            loading_time
            + cfreq * time_no_checkpoint / 2.0
            + time_no_checkpoint
            * min(tw_pccheck / time_no_checkpoint, cfreq * N_PCCHECK_TINY)
            / 2.0
        )
    if baseline == "Ideal":
        return cfreq * time_no_checkpoint / 2.0 + loading_time
    raise ValueError(f"unknown baseline {baseline}")


def goodput(
    baseline: str,
    cfreq: int,
    total_time_sec: float,
    num_failures: int,
    avg_iter_time: float,
    loading_time: float,
    time_no_checkpoint: float,
    tw_pccheck: float,
) -> float:
    redo = redo_time_sec(
        baseline,
        cfreq,
        time_no_checkpoint,
        loading_time,
        tw_pccheck,
    )
    useful_time = max(0.0, total_time_sec - redo * num_failures)
    return (useful_time / avg_iter_time) / total_time_sec


def read_tw_pccheck_sec(default_tw_sec: float) -> float:
    inputs = Path("fig8_tiny_inputs.csv")
    if not inputs.exists():
        return default_tw_sec
    df = pd.read_csv(inputs)
    if "PCcheck" not in df.columns:
        return default_tw_sec
    return float(df["PCcheck"].iloc[0]) / 1000.0


def plot(cfreqs: list[int], data: dict[str, list[float]], output_file: Path) -> None:
    x = np.arange(len(cfreqs[1:]))
    fig, ax = plt.subplots(figsize=(4.8, 2.8))

    markers = {"CheckFreq": "*", "GPM": "s", "PCcheck": "o"}
    plotted_labels = [label for label in SUPPORTED_BASELINES if label in data]
    for label in plotted_labels:
        ax.plot(
            x,
            data[label][1:],
            label=label,
            linewidth=1.3,
            marker=markers.get(label, "D"),
            markersize=4,
            color=BASELINE_COLORS[label],
        )
    ax.plot(
        x,
        data["Ideal"][1:],
        label="Ideal",
        linewidth=1.2,
        linestyle="--",
        color=BASELINE_COLORS["Ideal"],
    )

    ax.set_xticks(x, labels=cfreqs[1:])
    ax.set_ylabel("Goodput (batches/sec)")
    ax.set_xlabel("Checkpoint interval")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight", dpi=150, pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate tiny Figure 9 from fig8_tiny.csv."
    )
    parser.add_argument("--fig8-csv", default="fig8_tiny.csv")
    parser.add_argument("--total-time-sec", type=float, default=600.0)
    parser.add_argument("--num-failures", type=int, default=2)
    parser.add_argument("--load-time-sec", type=float, default=0.25)
    parser.add_argument("--tw-pccheck-sec", type=float, default=0.01)
    args = parser.parse_args()

    fig8_csv = Path(args.fig8_csv)
    if not fig8_csv.exists():
        raise SystemExit("fig8_tiny.csv not found. Run get_throughput_single_node.sh first.")

    throughput_df = pd.read_csv(fig8_csv, header=0, index_col=0)
    cfreqs = [int(col) for col in throughput_df.columns]
    baselines = [label for label in SUPPORTED_BASELINES if label in throughput_df.index]
    if "PCcheck" not in baselines:
        raise SystemExit("fig8_tiny.csv must include PCcheck to compute Ideal goodput.")
    iter_times = {
        label: [1.0 / value for value in throughput_df.loc[label].tolist()]
        for label in baselines
    }
    tw_pccheck = read_tw_pccheck_sec(args.tw_pccheck_sec)

    data: dict[str, list[float]] = {}
    for label in baselines + ["Ideal"]:
        data[label] = []
        for idx, cfreq in enumerate(cfreqs):
            source_label = "PCcheck" if label == "Ideal" else label
            data[label].append(
                goodput(
                    label,
                    cfreq,
                    args.total_time_sec,
                    args.num_failures,
                    iter_times[source_label][idx],
                    args.load_time_sec,
                    iter_times[source_label][0],
                    tw_pccheck,
                )
            )

    pd.DataFrame(
        [data[label] for label in baselines + ["Ideal"]],
        index=baselines + ["Ideal"],
        columns=[str(cfreq) for cfreq in cfreqs],
    ).to_csv("fig9_tiny.csv")
    plot(cfreqs, data, Path.cwd() / "fig9_tiny.png")
    print("Generated fig9_tiny.csv and fig9_tiny.png")


if __name__ == "__main__":
    main()
