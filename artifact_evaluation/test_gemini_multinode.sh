#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 MASTER_IP REMOTE_HOST"
    echo "       $0 --hostfile PATH"
    echo "Example: $0 10.128.0.19 10.128.0.20"
    echo "Example: $0 --hostfile ~/pccheck/hostfile"
}

HOSTFILE="${GEMINI_HOSTFILE:-}"
hosts=()

if [ "${1:-}" = "--hostfile" ]; then
    HOSTFILE="${2:-}"
    shift 2
fi

if [ -n "$HOSTFILE" ]; then
    if [ ! -f "$HOSTFILE" ]; then
        echo "Missing hostfile: $HOSTFILE"
        exit 1
    fi

    while read -r host _; do
        case "$host" in
            ""|\#*) continue ;;
            *) hosts+=("$host") ;;
        esac
    done < "$HOSTFILE"
else
    hosts=("$@")
fi

if [ "${#hosts[@]}" -ne 2 ]; then
    usage
    echo "Current test_gemini.py supports exactly two ranks: rank 0 sends, rank 1 receives."
    exit 1
fi

WORLD_SIZE="${#hosts[@]}"
MASTER_IP="${GEMINI_MASTER_IP:-${hosts[0]#*@}}"
MASTER_PORT="${GEMINI_MASTER_PORT:-29501}"

PCCHECK_HOME="${PCCHECK_HOME:-$HOME/pccheck}"
REMOTE_PCCHECK_HOME="${REMOTE_PCCHECK_HOME:-$PCCHECK_HOME}"
PYTHON="${PYTHON:-python3.9}"
SIZE_MB="${SIZE_MB:-1}"
ITERATIONS="${ITERATIONS:-4}"
GEMINI_TIMEOUT="${GEMINI_TIMEOUT:-120}"

MICROBENCH_DIR="$PCCHECK_HOME/checkpoint_eval/models/microbenchmarks"
REMOTE_MICROBENCH_DIR="$REMOTE_PCCHECK_HOME/checkpoint_eval/models/microbenchmarks"

pids=()
cleanup() {
    for pid in "${pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup INT TERM

for ((rank = 1; rank < WORLD_SIZE; rank++)); do
    remote_host="${hosts[$rank]}"
    echo "Starting Gemini rank $rank on $remote_host"
    ssh "$remote_host" \
        "cd '$REMOTE_PCCHECK_HOME' && timeout '$GEMINI_TIMEOUT' '$PYTHON' '$REMOTE_MICROBENCH_DIR/test_gemini.py' --size '$SIZE_MB' --iterations '$ITERATIONS' --rank '$rank' --world_size '$WORLD_SIZE' --master_ip '$MASTER_IP' --master_port '$MASTER_PORT'" &
    pids+=("$!")
done

echo "Starting Gemini rank 0 on local node"
timeout "$GEMINI_TIMEOUT" "$PYTHON" "$MICROBENCH_DIR/test_gemini.py" \
    --size "$SIZE_MB" \
    --iterations "$ITERATIONS" \
    --rank 0 \
    --world_size "$WORLD_SIZE" \
    --master_ip "$MASTER_IP" \
    --master_port "$MASTER_PORT" &
pids+=("$!")

gemini_status=0
for pid in "${pids[@]}"; do
    wait "$pid" || gemini_status=$?
done

if [ "$gemini_status" -ne 0 ]; then
    echo "Gemini multinode microbenchmark failed."
    exit "$gemini_status"
fi
