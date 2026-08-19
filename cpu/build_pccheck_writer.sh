#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/checkpoint_eval/pccheck"
BUILD="$ROOT/cpu/build"
CUDA_STUB="$ROOT/cpu/cuda_stub"
OUT="${1:-$BUILD/libtest_ssd_cpu.so}"

command -v g++ >/dev/null || {
  echo "ERROR: g++ is required (Ubuntu: sudo apt-get install build-essential)" >&2
  exit 2
}
[[ -f /usr/include/libpmem.h ]] || {
  echo "ERROR: libpmem development headers are required (Ubuntu: sudo apt-get install libpmem-dev)" >&2
  exit 2
}
[[ -f "$CUDA_STUB/cuda_runtime.h" ]] || {
  echo "ERROR: missing CPU CUDA compatibility header: $CUDA_STUB/cuda_runtime.h" >&2
  exit 2
}

# PCcheck's persistence path executes an explicit x86 CLWB instruction in
# BARRIER(). Fail before a long experiment instead of risking SIGILL later.
if [[ -r /proc/cpuinfo ]] && ! grep -m1 -qw clwb /proc/cpuinfo; then
  echo "ERROR: CPU/VM does not expose the CLWB instruction required by PCcheck's original writer." >&2
  exit 2
fi

cpus="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
if (( cpus < 2 )); then
  echo "ERROR: PCcheck's original checkpoint process pins itself to CPU 1; expose at least 2 logical CPUs." >&2
  exit 2
fi

mkdir -p "$BUILD" "$(dirname "$OUT")"

CXXFLAGS=(
  -O3 -fPIC -pthread
  -march=native -mtune=native -ffast-math
  -I"$CUDA_STUB"
  -I"$SRC"
)

# The original DRAMAlloc.h includes <cuda_runtime.h> only for cudaMallocHost()
# and cudaSuccess. The CPU compatibility include directory is intentionally
# first, so the original source compiles unchanged while cudaMallocHost maps to
# a distinct page-aligned CPU allocation. No CUDA SDK or libcudart is required.
g++ "${CXXFLAGS[@]}" -c "$SRC/main_ssd_memory.cpp" -o "$BUILD/main_ssd_memory.o"
g++ "${CXXFLAGS[@]}" -c "$SRC/socket_work.cpp" -o "$BUILD/socket_work.o"
g++ -shared -pthread -o "$OUT" \
  "$BUILD/main_ssd_memory.o" "$BUILD/socket_work.o" \
  -latomic -lpmem

echo "PCcheck CPU writer built: $OUT"
echo "CUDA SDK required: no (cpu/cuda_stub/cuda_runtime.h supplies cudaMallocHost compatibility)"
echo "Use this path as c_lib_path when running the original PCcheck checkpoint monitor."
