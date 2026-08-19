# PCcheck CPU compatibility: what is preserved and why

## Objective

Run PCcheck's original checkpoint control path on CPU-only VMs. The `cpu/`
code is a scoped hardware-compatibility layer, not a second implementation of
PCcheck.

The rule is:

> Preserve original control flow and real CPU/network/storage work. Substitute
> only primitives that require CUDA/NCCL hardware.

## Exact mapping to the original PCcheck code

### 1. Flat checkpoint state: GPU allocation -> device-surrogate allocation

`checkpoint_eval/pccheck_utils.py::initialize()` computes the real model plus
optimizer state size and creates:

```python
gpu_ar = torch.zeros(total_size).cuda()
```

The compatibility layer leaves the call in place. `Tensor.cuda()` is redirected
to a new CPU allocation with the same shape, dtype, values and number of bytes.
That storage is tagged as a **device surrogate** so later code can distinguish
it from host checkpoint DRAM even though both are physically CPU memory.

Why a new allocation instead of returning the same tensor: a real CUDA move
creates device-resident storage. Aliasing the original host tensor would make a
later GPU->DRAM checkpoint copy disappear.

### 2. GPU -> DRAM: retain PCcheck's original `copy_`

The original `Checkpoint.write_batch()` does:

```python
gpu_ar_new = self.gpu_ar[start_idx:end_idx]
cpu_ar_new = cpu_ar[start_idx:end_idx]
cpu_ar_new.copy_(gpu_ar_new)
```

That code is unchanged. `gpu_ar_new` is a view into tagged device-surrogate
storage. `cpu_ar_new` is a separate ordinary CPU staging buffer. Therefore
`copy_` still moves the complete checkpoint batch between two distinct
allocations.

The compatibility layer measures the real CPU copy. If an exact-size measured
GPU D2H target exists, it sleeps only the residual:

```text
emulated D2H time = max(real CPU copy time, measured GPU D2H time)
```

It never shortens a slow CPU operation and never extrapolates a timing from a
different buffer size.

### 3. Pinned memory

PCcheck normally allocates its host staging buffers with `pin_memory=True`.
Pinned memory is useful for CUDA DMA, but a CPU-only VM has no CUDA DMA engine
to pin for. The shim therefore creates a **separate ordinary host allocation**.
The important algorithmic property — a distinct checkpoint staging buffer — is
preserved. CUDA page-locking itself is not claimed to be reproduced.

### 4. Pipeline and concurrency

The following PCcheck code is not rewritten:

```text
Checkpoint.write_pipelined()
  -> per-batch Barrier objects
  -> per-batch threads
  -> Checkpoint.write_batch()
  -> D2H staging copy
  -> Writer.savenvm_new()
```

Batch slicing, barriers, thread ordering, `cp_in_progress`, and `max_async`
remain PCcheck behavior.

### 5. Why the unit test substitutes only `Writer`

The real Python `Writer` loads PCcheck's C++ persistence engine through ctypes.
That engine contains mmap/msync persistence, checkpoint metadata, CAS, DRAM
buffer management, and optional distributed checkpoint coordination.

A clean CPU VM may not have that shared library built yet. Therefore
`cpu/run_unit_test.py` injects a test-local `CpuPersistenceWriter` **only at the
Python/C++ boundary**. It receives exactly the batches produced by the original
PCcheck pipeline, writes them at their proper offsets, and calls `fsync` when
the checkpoint completes.

This fixture validates the Python checkpoint pipeline. It is not presented as
an equivalent implementation of PCcheck's C++ mmap/msync/CAS engine. For the
full experiment, build and use the original C++ library.

### 6. Distributed communication

The old fake-GPU test mocked `send`, `recv`, `barrier`, and process-group setup.
That is not used by this compatibility design.

The shared layer changes only a requested process-group backend from `nccl` to
`gloo`. PyTorch send/recv/collective/barrier calls remain real. PCcheck's C++
distributed path also remains untouched when the original library is used.
Importantly, PCcheck does **not** send the whole checkpoint to its peer: its
checkpoint payload is persisted locally and the C++ code exchanges small
coordination messages to establish a globally consistent completed checkpoint.
That difference from Gemini and the baseline is intentional and must remain.

## Timing profile

`cpu/gpu_profile.example.json` shows the schema. Its zeros are synthetic unit
-test placeholders, not GPU measurements. With a real profile and strict mode,
a missing exact `(kind, bytes)` measurement is an error rather than an invented
bandwidth estimate.

## Run

```bash
python cpu/run_unit_test.py --size-mb 1 --batches 4
```

The test succeeds only if the original PCcheck Python pipeline performs every
batch copy into distinct host staging memory and the resulting complete payload
is persisted by the unit-test Writer boundary.

## Remaining fidelity limits

1. CPU memcpy plus a calibrated wait does not physically recreate PCIe/NVLink
   contention or a GPU DMA engine.
2. The default unit test does not execute PCcheck's original C++ persistence
   implementation; build that library for persistence/coordination experiments.
3. The shared CUDA stream surrogate is synchronous. It preserves ordering but
   does not reproduce CUDA copy-engine/kernel overlap.
4. Function-level GPU training latency still requires measured GPU traces. No
   forward/backward/optimizer timing is invented by this patch.
