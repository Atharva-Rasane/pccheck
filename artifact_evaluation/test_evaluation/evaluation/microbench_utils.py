from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
MICROBENCH_DIR = REPO_ROOT / "checkpoint_eval" / "models" / "microbenchmarks"
PCCHECK_LIB_PATH = Path(
    os.environ.get(
        "PCCHECK_LIB_PATH",
        str(REPO_ROOT / "checkpoint_eval" / "pccheck" / "libtest_ssd.so"),
    )
)
PYTHON = os.environ.get("PCCHECK_PYTHON", sys.executable)

BASELINE_KEYS = ["cfreq", "gpm", "pccheck"]
BASELINE_LABELS = {
    "cfreq": "CheckFreq",
    "gpm": "GPM",
    "pccheck": "PCcheck",
}
BASELINE_COLORS = {
    "CheckFreq": "#4392B8",
    "GPM": "#E27733",
    "PCcheck": "#A7B972",
    "Ideal": "#777777",
}


def ensure_iterations(iterations: int) -> None:
    if iterations <= 3:
        raise ValueError("iterations must be greater than the 3 warmup iterations")


def microbench_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not existing
        else f"{REPO_ROOT}{os.pathsep}{existing}"
    )
    return env


def parse_last_float(line: str) -> float:
    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", line)
    if not matches:
        raise ValueError(f"no numeric value found in line: {line.strip()}")
    return float(matches[-1])


def parse_microbenchmark_time(log_file: Path, iterations: int) -> float:
    average_ms = None
    mmap_unmap_total_ms = 0.0

    with log_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if "AVERAGE Checkpoint" in line:
                average_ms = parse_last_float(line)
            elif "MMAP/UMAP" in line:
                mmap_unmap_total_ms = parse_last_float(line)

    if average_ms is None:
        raise RuntimeError(f"could not find average checkpoint time in {log_file}")

    adjusted_ms = average_ms - (mmap_unmap_total_ms / iterations)
    return max(adjusted_ms, 0.001)


def run_microbenchmark(
    baseline: str,
    size_mb: int,
    iterations: int,
    output_dir: Path,
    *,
    num_threads: int = 2,
    force: bool = False,
    tag: Optional[str] = None,
) -> Path:
    ensure_iterations(iterations)
    if baseline not in BASELINE_KEYS:
        raise ValueError(f"unknown microbenchmark baseline: {baseline}")

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    log_file = output_dir / f"{baseline}_{size_mb}mb{suffix}.txt"
    if log_file.exists() and not force:
        return log_file

    script = MICROBENCH_DIR / f"test_{baseline}.py"
    command = [
        PYTHON,
        str(script),
        "--size",
        str(size_mb),
        "--iterations",
        str(iterations),
    ]

    if baseline == "pccheck":
        if not PCCHECK_LIB_PATH.exists():
            raise FileNotFoundError(
                f"PCcheck library not found at {PCCHECK_LIB_PATH}. "
                "Run install.sh or set PCCHECK_LIB_PATH."
            )
        command.extend(
            [
                "--num-threads",
                str(num_threads),
                "--c_lib_path",
                str(PCCHECK_LIB_PATH),
            ]
        )

    with log_file.open("w", encoding="utf-8") as stdout:
        result = subprocess.run(
            command,
            cwd=output_dir,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            env=microbench_env(),
            check=False,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"{baseline} microbenchmark failed with exit code "
            f"{result.returncode}. See {log_file}."
        )
    return log_file


def collect_microbenchmark_matrix(
    sizes_mb: list[int],
    iterations: int,
    output_dir: Path,
    *,
    baselines: Optional[list[str]] = None,
    num_threads: int = 2,
    force: bool = False,
) -> dict[str, list[float]]:
    baselines = baselines or BASELINE_KEYS
    data: dict[str, list[float]] = {}

    for baseline in baselines:
        label = BASELINE_LABELS[baseline]
        data[label] = []
        for size_mb in sizes_mb:
            log_file = run_microbenchmark(
                baseline,
                size_mb,
                iterations,
                output_dir,
                num_threads=num_threads,
                force=force,
            )
            data[label].append(parse_microbenchmark_time(log_file, iterations))

    return data


def tiny_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
