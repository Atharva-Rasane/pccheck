#!/bin/bash
set -euo pipefail

PCCHECK_HOME="${PCCHECK_HOME:-$HOME/pccheck}"
PYTHON="${PYTHON:-python3.9}"
MICROBENCH_DIR="$PCCHECK_HOME/checkpoint_eval/models/microbenchmarks"
PCCHECK_LIB="$PCCHECK_HOME/checkpoint_eval/pccheck/libtest_ssd.so"
GPM_LIB="$PCCHECK_HOME/checkpoint_eval/gpm/libtest.so"

SIZE_MB="${SIZE_MB:-1}"
ITERATIONS="${ITERATIONS:-4}"
NUM_THREADS="${NUM_THREADS:-2}"
GEMINI_MASTER_IP="${GEMINI_MASTER_IP:-127.0.0.1}"
GEMINI_MASTER_PORT="${GEMINI_MASTER_PORT:-29501}"
GEMINI_TIMEOUT="${GEMINI_TIMEOUT:-120}"

print_section() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

if [ ! -f "$PCCHECK_LIB" ]; then
    echo "Missing $PCCHECK_LIB. Run bash install.sh first."
    exit 1
fi

if [ ! -f "$GPM_LIB" ]; then
    echo "Missing $GPM_LIB. Run bash install.sh first."
    exit 1
fi

print_section "PCcheck microbenchmark smoke test"
echo "PCCHECK_HOME: $PCCHECK_HOME"
echo "size: ${SIZE_MB} MB"
echo "iterations: $ITERATIONS"
echo "PCcheck writer threads: $NUM_THREADS"

print_section "Running CheckFreq"
"$PYTHON" "$MICROBENCH_DIR/test_cfreq.py" \
    --size "$SIZE_MB" \
    --iterations "$ITERATIONS"

print_section "Running GPM"
"$PYTHON" "$MICROBENCH_DIR/test_gpm.py" \
    --size "$SIZE_MB" \
    --iterations "$ITERATIONS"

print_section "Running PCcheck"
"$PYTHON" "$MICROBENCH_DIR/test_pccheck.py" \
    --size "$SIZE_MB" \
    --iterations "$ITERATIONS" \
    --num-threads "$NUM_THREADS" \
    --c_lib_path "$PCCHECK_LIB"

print_section "Running Gemini"
timeout "$GEMINI_TIMEOUT" "$PYTHON" "$MICROBENCH_DIR/test_gemini.py" \
    --size "$SIZE_MB" \
    --iterations "$ITERATIONS" \
    --rank 0 \
    --world_size 1 \
    --master_ip "$GEMINI_MASTER_IP" \
    --master_port "$GEMINI_MASTER_PORT"

print_section "Smoke test complete"
