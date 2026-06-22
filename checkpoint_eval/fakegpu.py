import os
from contextlib import ExitStack, nullcontext
from unittest import mock

import torch


FAKEGPU_ENV = "PCCHECK_FAKEGPU"


def add_fakegpu_argument(parser):
    parser.add_argument(
        "--fakegpu",
        action="store_true",
        help=(
            "Run without a real GPU by mocking CUDA calls and returning CPU "
            "garbage tensors with matching shape and dtype."
        ),
    )


def fakegpu_enabled(args=None):
    return bool(getattr(args, "fakegpu", False) or os.environ.get(FAKEGPU_ENV) == "1")


def maybe_set_affinity(cpus):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)


def _is_fake_device(device):
    if device is None:
        return False
    if isinstance(device, int):
        return True
    text = str(device)
    return text == "cuda" or text.startswith("cuda:")


def _extract_to_dtype(args, kwargs, fallback):
    if "dtype" in kwargs and kwargs["dtype"] is not None:
        return kwargs["dtype"]
    for arg in args:
        if isinstance(arg, torch.dtype):
            return arg
        if isinstance(arg, torch.Tensor):
            return arg.dtype
    return fallback


def _to_requests_fake_device(args, kwargs):
    if _is_fake_device(kwargs.get("device")):
        return True
    for arg in args:
        if isinstance(arg, torch.dtype):
            continue
        if isinstance(arg, torch.Tensor):
            return _is_fake_device(arg.device)
        if _is_fake_device(arg):
            return True
    return False


def _empty_like_garbage(tensor, dtype=None):
    try:
        return torch.empty_like(
            tensor,
            dtype=dtype or tensor.dtype,
            device="cpu",
            memory_format=torch.preserve_format,
        )
    except TypeError:
        return torch.empty_like(tensor, dtype=dtype or tensor.dtype, device="cpu")


def _clean_factory_kwargs(kwargs):
    cleaned = dict(kwargs)
    if _is_fake_device(cleaned.get("device")):
        cleaned["device"] = "cpu"
    if cleaned.get("pin_memory"):
        cleaned["pin_memory"] = False
    return cleaned


def _factory_wrapper(original):
    def wrapped(*args, **kwargs):
        return original(*args, **_clean_factory_kwargs(kwargs))

    return wrapped


class FakeGradScaler:
    def __init__(self, *args, **kwargs):
        self.enabled = kwargs.get("enabled", True)

    def scale(self, loss):
        return loss

    def step(self, optimizer, *args, **kwargs):
        return optimizer.step(*args, **kwargs)

    def update(self, *args, **kwargs):
        return None

    def unscale_(self, optimizer):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return None


def fake_checkpoint_loop(iterations, prefix="CHECKPOINT"):
    import time
    import numpy as np

    warmup = 3
    checkpoint_time_list = []
    for it in range(iterations):
        start_time = time.time()
        garbage = torch.empty(1, dtype=torch.float32)
        _ = garbage.shape
        duration = (time.time() - start_time) * 1000
        if it >= warmup:
            checkpoint_time_list.append(duration)
        print(f"{prefix} {it} TOOK {duration} ms")

    average = np.average(checkpoint_time_list) if checkpoint_time_list else 0.0
    print(f"AVERAGE Checkpoint time is {average} ms")


def fakegpu_context(enabled, *, fake_distributed=False, rank=0, world_size=1):
    stack = ExitStack()
    if not enabled:
        return stack

    stack.enter_context(mock.patch.dict(os.environ, {FAKEGPU_ENV: "1"}))

    tensor_cuda = torch.Tensor.cuda
    tensor_to = torch.Tensor.to
    module_to = torch.nn.Module.to
    torch_load = torch.load

    def fake_tensor_cuda(self, *args, **kwargs):
        return _empty_like_garbage(self)

    def fake_tensor_to(self, *args, **kwargs):
        if _to_requests_fake_device(args, kwargs):
            return _empty_like_garbage(self, dtype=_extract_to_dtype(args, kwargs, self.dtype))
        return tensor_to(self, *args, **kwargs)

    def fake_module_cuda(self, *args, **kwargs):
        return self._apply(lambda tensor: _empty_like_garbage(tensor))

    def fake_module_to(self, *args, **kwargs):
        if _to_requests_fake_device(args, kwargs):
            dtype = _extract_to_dtype(args, kwargs, None)
            return self._apply(lambda tensor: _empty_like_garbage(tensor, dtype=dtype))
        return module_to(self, *args, **kwargs)

    def fake_torch_load(*args, **kwargs):
        map_location = kwargs.get("map_location")
        if _is_fake_device(map_location):
            kwargs = dict(kwargs)
            kwargs["map_location"] = "cpu"
        return torch_load(*args, **kwargs)

    stack.enter_context(mock.patch.object(torch.Tensor, "cuda", fake_tensor_cuda))
    stack.enter_context(mock.patch.object(torch.Tensor, "to", fake_tensor_to))
    stack.enter_context(mock.patch.object(torch.nn.Module, "cuda", fake_module_cuda))
    stack.enter_context(mock.patch.object(torch.nn.Module, "to", fake_module_to))
    stack.enter_context(mock.patch.object(torch, "load", fake_torch_load))

    for name in ["empty", "empty_like", "ones", "zeros", "full", "arange", "tensor"]:
        stack.enter_context(mock.patch.object(torch, name, _factory_wrapper(getattr(torch, name))))

    stack.enter_context(mock.patch.object(torch.cuda, "is_available", return_value=True))
    stack.enter_context(mock.patch.object(torch.cuda, "device_count", return_value=1))
    stack.enter_context(mock.patch.object(torch.cuda, "current_device", return_value=0))
    stack.enter_context(mock.patch.object(torch.cuda, "set_device", return_value=None))
    stack.enter_context(mock.patch.object(torch.cuda, "synchronize", return_value=None))
    stack.enter_context(mock.patch.object(torch.cuda, "empty_cache", return_value=None))
    stack.enter_context(mock.patch.object(torch.cuda, "manual_seed_all", return_value=None))
    stack.enter_context(mock.patch.object(torch.cuda, "IntTensor", torch.IntTensor))
    stack.enter_context(mock.patch.object(torch.cuda.amp, "autocast", lambda *a, **k: nullcontext()))
    stack.enter_context(mock.patch.object(torch.cuda.amp, "GradScaler", FakeGradScaler))
    stack.enter_context(
        mock.patch.object(torch.backends.cuda, "sdp_kernel", lambda *a, **k: nullcontext())
    )

    if fake_distributed:
        import torch.distributed as dist

        def fake_recv(tensor, src=None, *args, **kwargs):
            if torch.is_tensor(tensor):
                tensor.copy_(torch.empty_like(tensor))
            return None

        stack.enter_context(mock.patch.object(dist, "init_process_group", return_value=None))
        stack.enter_context(mock.patch.object(dist, "barrier", return_value=None))
        stack.enter_context(mock.patch.object(dist, "send", return_value=None))
        stack.enter_context(mock.patch.object(dist, "recv", fake_recv))
        stack.enter_context(mock.patch.object(dist, "destroy_process_group", return_value=None))
        stack.enter_context(mock.patch.object(dist, "get_rank", return_value=rank))
        stack.enter_context(mock.patch.object(dist, "get_world_size", return_value=world_size))
        stack.enter_context(mock.patch.object(dist, "is_initialized", return_value=True))

    return stack
