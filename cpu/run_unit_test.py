#!/usr/bin/env python3
"""Run PCcheck's original Python checkpoint pipeline on CPU-only hardware.

The original Checkpoint.write_pipelined()/write_batch() functions execute. Only
the compiled Writer boundary is replaced in this unit test by a small file
writer so the Python pipeline can be tested before the original C++ library is
built on the VM.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpu.gpu_compat import CpuGpuCompat, TimingProfile
from checkpoint_eval.pccheck.chk_checkpoint_pipeline import Checkpoint


class CpuPersistenceWriter:
    """Unit-test implementation of PCcheck Writer's Python-facing methods."""
    def __init__(self, path: Path, total_floats: int):
        self.path = Path(path)
        self.total_floats = int(total_floats)
        self._checkpoint_id = 0
        self._lock = threading.Lock()
        with self.path.open("wb") as f:
            f.truncate(self.total_floats * 4)

    def register(self):
        with self._lock:
            value = self._checkpoint_id
            self._checkpoint_id += 1
            return value

    def savenvm_new(self, arr, total_size, num_threads, checkp_info,
                    batch_num, batch_size, last_batch):
        del num_threads, checkp_info
        data = np.asarray(arr, dtype=np.float32).reshape(-1).tobytes(order="C")
        expected_floats = min(
            int(batch_size),
            int(total_size) - (int(batch_num) - 1) * int(batch_size),
        )
        expected_bytes = max(0, expected_floats) * 4
        if len(data) != expected_bytes:
            raise AssertionError(
                f"writer got {len(data)} bytes; expected {expected_bytes}"
            )
        offset = (int(batch_num) - 1) * int(batch_size) * 4
        with self._lock:
            with self.path.open("r+b", buffering=0) as f:
                f.seek(offset)
                f.write(data)
                if last_batch:
                    os.fsync(f.fileno())


def _storage_ptr(tensor):
    try:
        return int(tensor.untyped_storage().data_ptr())
    except AttributeError:
        return int(tensor.storage().data_ptr())


def _transfers(path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if json.loads(line).get("event") == "transfer"]


def run_test(size_mb, batches, profile_path):
    total_bytes = int(size_mb) * 1024 * 1024
    total_floats = total_bytes // 4
    if total_bytes <= 0 or total_floats % int(batches):
        raise ValueError("size must be positive and divide evenly into batches")
    batch_floats = total_floats // int(batches)
    batch_bytes = batch_floats * 4
    profile = TimingProfile.from_json(profile_path)

    with tempfile.TemporaryDirectory(prefix="pccheck-cpu-") as td_raw:
        td = Path(td_raw)
        trace = td / "compat.jsonl"
        output = td / "checkpoint.bin"

        with CpuGpuCompat(
            timing_profile=profile,
            strict_timing=profile_path is not None,
            trace_path=str(trace),
        ) as runtime:
            gpu_ar = torch.arange(total_floats, dtype=torch.float32).cuda()
            cpu_ar = torch.empty(total_floats, dtype=torch.float32, device="cpu")
            assert runtime.is_device_surrogate(gpu_ar)
            assert not runtime.is_device_surrogate(cpu_ar)
            assert _storage_ptr(gpu_ar) != _storage_ptr(cpu_ar)

            checkpoint = Checkpoint(
                total_size=total_floats,
                num_threads=1,
                filename=str(output),
                lib_path="",
                max_async=1,
                gpu_ar=gpu_ar,
                bsize=batch_floats,
                memory_saving=False,
            )
            checkpoint.writer = CpuPersistenceWriter(output, total_floats)
            progress = SimpleNamespace(value=1)
            lock = threading.Lock()
            checkpoint.write_pipelined(
                cpu_ar, 1, total_floats, batch_floats, lock, progress
            )
            assert progress.value == 0
            assert torch.equal(
                cpu_ar, torch.arange(total_floats, dtype=torch.float32)
            )

        expected = np.arange(total_floats, dtype=np.float32).tobytes(order="C")
        actual = output.read_bytes()
        assert actual == expected
        d2h = [r for r in _transfers(trace)
               if r.get("kind") == "d2h" and r.get("bytes") == batch_bytes]
        assert len(d2h) >= int(batches)

        print("PASS PCcheck CPU compatibility")
        print(f"checkpoint_bytes={total_bytes}")
        print(f"batches={batches} batch_bytes={batch_bytes}")
        print("original_pipeline=Checkpoint.write_pipelined")
        print(f"persisted_bytes={len(actual)} fsync=yes")
        print(f"timing_source={profile.source}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mb", type=int, default=1)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--profile", default=None)
    args = parser.parse_args()
    run_test(args.size_mb, args.batches, args.profile)


if __name__ == "__main__":
    main()
