# CPU VM system requirements

## Focused one-node unit test

`python cpu/run_unit_test.py --size-mb 1 --batches 4` does not require CUDA,
an NVIDIA driver, NCCL, PMDK, Docker, or the compiled PCcheck writer. It uses
the test-local persistence fixture only to validate the original Python
`Checkpoint.write_pipelined()` / `write_batch()` path.

The VM should expose at least 2 logical CPUs for consistency with PCcheck's
original affinity assumptions, although the focused writer-fixture test itself
does not enter `start_chk()`.

## Full PCcheck persistence path on a CPU VM

The original SSD writer is Linux/x86-oriented and requires:

- `g++` / `build-essential`;
- PMDK development headers/library (`libpmem-dev` on Ubuntu);
- a CPU/VM exposing the `clwb` instruction used by the original `BARRIER()`;
- at least 2 logical CPUs because the original checkpoint process pins itself
  to CPU 1;
- a writable checkpoint path with enough address-space/storage capacity for the
  original mmap-backed writer.

The repository's root `checkpoint_eval/pccheck/Makefile` hardcodes CUDA 12.1
paths and links `-lcudart`. Do not install CUDA merely to satisfy that Makefile
on a CPU-only VM. Use:

```bash
sudo apt-get update
sudo apt-get install -y build-essential libpmem-dev
bash cpu/build_pccheck_writer.sh
```

That script compiles the unchanged original `main_ssd_memory.cpp` and
`socket_work.cpp` without the unnecessary CUDA include/link flags.

For distributed PCcheck coordination, rank 0 and peer ranks must additionally
be able to reach the original coordination socket (port 1235 in the current C++
source) and `PCCHECK_COORDINATOR` must resolve to rank 0.
