import os
from platform import node
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import models, datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.multiprocessing import Pool, Process, set_start_method, Manager, Value, Lock
from datetime import timedelta
import random
import numpy as np
import time
import os
import argparse

import ctypes
from checkpoint_eval.pccheck.chk_monitor import Chk_monitor
from checkpoint_eval.pccheck_utils import initialize, get_total_size, set_storage
from checkpoint_eval.fakegpu import (
    add_fakegpu_argument,
    fake_checkpoint_loop,
    fakegpu_context,
    maybe_set_affinity,
)

parser = argparse.ArgumentParser(description="CheckFreq microbenchmark")
parser.add_argument(
    "--size", default=1, type=int, help="size of the object to checkpoint (in MB)"
)
parser.add_argument("--iterations", default=10, type=int, help="iterations to simulate")
parser.add_argument(
    "--num-threads", default=1, type=int, help="Number of CPU threads writing at NVM"
)
parser.add_argument(
    "--c_lib_path",
    default="",
    type=str,
    required=False,
    help="path to the libtest.so library",
)
add_fakegpu_argument(parser)


def run(args):
    if not args.fakegpu and not args.c_lib_path:
        raise ValueError("--c_lib_path is required unless --fakegpu is used")

    num_floats = args.size * 1000000 / 4
    print(f"allocate tensor of {int(num_floats)} floats")

    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            tensor = torch.ones(int(num_floats), dtype=torch.float32)
            self.a = torch.nn.Parameter(tensor)

    model = TestModel()
    model.cuda()
    gpu_ar, total_size = initialize(model, [])

    start_idx = 0
    # assume gpu_ar is big 1D GPU tensor
    set_storage(model, [], gpu_ar)
    torch.cuda.empty_cache()

    if args.fakegpu:
        fake_checkpoint_loop(args.iterations, prefix="----------------- CHECKPOINT")
        return

    chk_monitor = Chk_monitor(
        args.c_lib_path,
        total_size,
        args.num_threads,
        1,
        True,
        gpu_ar=gpu_ar,
        is_sync=True,
        bsize=total_size,
        model=model.state_dict(),
        optimizer={},
        memory_saving=True,
    )

    warmup = 3
    checkpoint_time_list = []
    for it in range(args.iterations):
        time.sleep(2)
        print(f"-------------------------- Start ITER {it}")

        start_time = time.time()
        chk_monitor.save()
        while chk_monitor.checkpoint_in_progress():
            continue

        end_time = time.time()
        duration = (end_time - start_time) * 1000
        if it >= warmup:
            checkpoint_time_list.append(duration)

        print(f"----------------- CHECKPOINT {it} TOOK {duration} ms")

    if chk_monitor:
        chk_monitor.kill_checkpoint()

    print(f"AVERAGE Checkpoint time is {np.average(checkpoint_time_list)} ms")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    maybe_set_affinity({0})
    args = parser.parse_args()
    with fakegpu_context(args.fakegpu):
        run(args)
