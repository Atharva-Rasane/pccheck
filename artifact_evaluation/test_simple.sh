#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PCCHECK_HOME="${PCCHECK_HOME:-$REPO_ROOT}"
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
FAKEGPU=0
EXTRA_ARGS=()

usage() {
    echo "Usage: $0 [--fakegpu]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --fakegpu)
            FAKEGPU=1
            EXTRA_ARGS+=(--fakegpu)
            export PCCHECK_FAKEGPU=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
    shift
done

export PYTHONPATH="$PCCHECK_HOME${PYTHONPATH:+:$PYTHONPATH}"

print_section() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

if [ "$FAKEGPU" -eq 0 ] && [ ! -f "$PCCHECK_LIB" ]; then
    echo "Missing $PCCHECK_LIB. Run bash install.sh first."
    exit 1
fi

if [ "$FAKEGPU" -eq 0 ] && [ ! -f "$GPM_LIB" ]; then
    echo "Missing $GPM_LIB. Run bash install.sh first."
    exit 1
fi

PCCHECK_ARG="$PCCHECK_LIB"
if [ "$FAKEGPU" -eq 1 ]; then
    PCCHECK_ARG="fakegpu"
fi

print_section "PCcheck microbenchmark smoke test"
echo "PCCHECK_HOME: $PCCHECK_HOME"
echo "size: ${SIZE_MB} MB"
echo "iterations: $ITERATIONS"
echo "PCcheck writer threads: $NUM_THREADS"
echo "fake GPU: $FAKEGPU"

print_section "Running CheckFreq"
"$PYTHON" "$MICROBENCH_DIR/test_cfreq.py" \
    --size "$SIZE_MB" \
    --iterations "$ITERATIONS" \
    "${EXTRA_ARGS[@]}"

print_section "Running GPM"
"$PYTHON" "$MICROBENCH_DIR/test_gpm.py" \
    --size "$SIZE_MB" \
    --iterations "$ITERATIONS" \
    "${EXTRA_ARGS[@]}"

print_section "Running PCcheck"
"$PYTHON" "$MICROBENCH_DIR/test_pccheck.py" \
    --size "$SIZE_MB" \
    --iterations "$ITERATIONS" \
    --num-threads "$NUM_THREADS" \
    --c_lib_path "$PCCHECK_ARG" \
    "${EXTRA_ARGS[@]}"

print_section "Running Gemini"
timeout "$GEMINI_TIMEOUT" "$PYTHON" "$MICROBENCH_DIR/test_gemini.py" \
    --size "$SIZE_MB" \
    --iterations "$ITERATIONS" \
    --rank 0 \
    --world_size 1 \
    --master_ip "$GEMINI_MASTER_IP" \
    --master_port "$GEMINI_MASTER_PORT" \
    "${EXTRA_ARGS[@]}"

print_section "Smoke test complete"
