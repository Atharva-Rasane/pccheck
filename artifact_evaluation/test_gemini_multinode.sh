#!/bin/bash
set -euo pipefail

MASTER_IP="${1:-${GEMINI_MASTER_IP:-}}"
REMOTE_HOST="${2:-${GEMINI_REMOTE_HOST:-}}"
MASTER_PORT="${GEMINI_MASTER_PORT:-29501}"

PCCHECK_HOME="${PCCHECK_HOME:-$HOME/pccheck}"
REMOTE_PCCHECK_HOME="${REMOTE_PCCHECK_HOME:-$PCCHECK_HOME}"
PYTHON="${PYTHON:-python3.9}"
SIZE_MB="${SIZE_MB:-1}"
ITERATIONS="${ITERATIONS:-4}"
GEMINI_TIMEOUT="${GEMINI_TIMEOUT:-120}"

if [ -z "$MASTER_IP" ] || [ -z "$REMOTE_HOST" ]; then
    echo "Usage: $0 MASTER_IP REMOTE_HOST"
    echo "Example: $0 10.128.0.19 10.128.0.20"
    exit 1
fi

MICROBENCH_DIR="$PCCHECK_HOME/checkpoint_eval/models/microbenchmarks"
REMOTE_MICROBENCH_DIR="$REMOTE_PCCHECK_HOME/checkpoint_eval/models/microbenchmarks"

echo "Starting Gemini rank 1 on $REMOTE_HOST"
ssh "$REMOTE_HOST" \
    "cd '$REMOTE_PCCHECK_HOME' && timeout '$GEMINI_TIMEOUT' '$PYTHON' '$REMOTE_MICROBENCH_DIR/test_gemini.py' --size '$SIZE_MB' --iterations '$ITERATIONS' --rank 1 --world_size 2 --master_ip '$MASTER_IP' --master_port '$MASTER_PORT'" &
remote_rank_pid=$!

echo "Starting Gemini rank 0 on local node"
timeout "$GEMINI_TIMEOUT" "$PYTHON" "$MICROBENCH_DIR/test_gemini.py" \
    --size "$SIZE_MB" \
    --iterations "$ITERATIONS" \
    --rank 0 \
    --world_size 2 \
    --master_ip "$MASTER_IP" \
    --master_port "$MASTER_PORT" &
local_rank_pid=$!

gemini_status=0
wait "$local_rank_pid" || gemini_status=$?
wait "$remote_rank_pid" || gemini_status=$?

if [ "$gemini_status" -ne 0 ]; then
    echo "Gemini multinode microbenchmark failed."
    exit "$gemini_status"
fi
