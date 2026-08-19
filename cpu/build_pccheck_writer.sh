#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/checkpoint_eval/pccheck"
BUILD="$ROOT/cpu/build"
OUT="${1:-$BUILD/libtest_ssd_cpu.so}"

command -v g++ >/dev/null || {
  echo "ERROR: g++ is required (Ubuntu: sudo apt-get install build-essential)" >&2
  exit 2
}
[[ -f /usr/include/libpmem.h ]] || {
  echo "ERROR: libpmem development headers are required (Ubuntu: sudo apt-get install libpmem-dev)" >&2
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
  -I"$SRC"
)

# Deliberately no CUDA include path and no -lcudart. main_ssd_memory.cpp and
# socket_work.cpp contain the persistence/coordination implementation and do
# not require CUDA; GPU capture is handled above this boundary in Python.
g++ "${CXXFLAGS[@]}" -c "$SRC/main_ssd_memory.cpp" -o "$BUILD/main_ssd_memory.o"
g++ "${CXXFLAGS[@]}" -c "$SRC/socket_work.cpp" -o "$BUILD/socket_work.o"
g++ -shared -pthread -o "$OUT" \
  "$BUILD/main_ssd_memory.o" "$BUILD/socket_work.o" \
  -latomic -lpmem

echo "PCcheck CPU writer built: $OUT"
echo "Use this path as c_lib_path when running the original PCcheck checkpoint monitor."
