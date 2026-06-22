#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 OUTPUT_LOG SIZE_MB ITERATIONS [--fakegpu]"
    echo
    echo "Configure nodes with either:"
    echo "  GEMINI_HOSTFILE=/path/to/hostfile"
    echo "or:"
    echo "  GEMINI_MASTER_IP=<rank0-internal-ip> GEMINI_REMOTE_HOST=<rank1-ssh-host>"
}

if [ "$#" -ne 3 ] && [ "$#" -ne 4 ]; then
    usage
    exit 1
fi

OUTPUT_LOG="$1"
SIZE_MB="$2"
ITERATIONS="$3"
FAKEGPU=0
if [ "$#" -eq 4 ]; then
    if [ "$4" != "--fakegpu" ]; then
        usage
        exit 1
    fi
    FAKEGPU=1
fi
REMOTE_LOG="${OUTPUT_LOG%.txt}_rank1.txt"
LAUNCH_LOG="${OUTPUT_LOG%.txt}_launcher.txt"

PCCHECK_HOME="${PCCHECK_HOME:-$HOME/pccheck}"
REMOTE_PCCHECK_HOME="${REMOTE_PCCHECK_HOME:-$PCCHECK_HOME}"
PYTHON_BIN="${PCCHECK_PYTHON:-${PYTHON:-python3.9}}"
GEMINI_TIMEOUT="${GEMINI_TIMEOUT:-180}"
MASTER_PORT="${GEMINI_MASTER_PORT:-29501}"

MICROBENCH_DIR="$PCCHECK_HOME/checkpoint_eval/models/microbenchmarks"
REMOTE_MICROBENCH_DIR="$REMOTE_PCCHECK_HOME/checkpoint_eval/models/microbenchmarks"

mkdir -p "$(dirname "$OUTPUT_LOG")"
: > "$LAUNCH_LOG"

if [ "$FAKEGPU" -eq 1 ]; then
    {
        echo "Starting Gemini fake GPU single-rank run"
        echo "Writing log to $OUTPUT_LOG"
    } | tee -a "$LAUNCH_LOG"

    PCCHECK_FAKEGPU=1 timeout "$GEMINI_TIMEOUT" "$PYTHON_BIN" "$MICROBENCH_DIR/test_gemini.py" \
        --size "$SIZE_MB" \
        --iterations "$ITERATIONS" \
        --rank 0 \
        --world_size 1 \
        --master_ip 127.0.0.1 \
        --master_port "$MASTER_PORT" \
        --fakegpu \
        > "$OUTPUT_LOG" 2>&1
    echo "Gemini fake GPU microbenchmark completed." | tee -a "$LAUNCH_LOG"
    exit 0
fi

HOSTFILE="${GEMINI_HOSTFILE:-}"
hosts=()

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
    if [ -z "${GEMINI_REMOTE_HOST:-}" ]; then
        usage
        echo "Set GEMINI_REMOTE_HOST or GEMINI_HOSTFILE before running Gemini."
        exit 1
    fi
    if [ -z "${GEMINI_MASTER_IP:-}" ]; then
        usage
        echo "Set GEMINI_MASTER_IP to rank 0's internal IP."
        exit 1
    fi
    hosts=("$GEMINI_MASTER_IP" "$GEMINI_REMOTE_HOST")
fi

if [ "${#hosts[@]}" -ne 2 ]; then
    usage
    echo "Current test_gemini.py supports exactly two ranks: rank 0 sends, rank 1 receives."
    exit 1
fi

WORLD_SIZE="${#hosts[@]}"
MASTER_IP="${GEMINI_MASTER_IP:-${hosts[0]#*@}}"

pids=()
cleanup() {
    for pid in "${pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup INT TERM

remote_host="${hosts[1]}"
{
    echo "Starting Gemini rank 1 on $remote_host"
    echo "Master: $MASTER_IP:$MASTER_PORT"
} | tee -a "$LAUNCH_LOG"

ssh "$remote_host" \
    "cd '$REMOTE_PCCHECK_HOME' && timeout '$GEMINI_TIMEOUT' '$PYTHON_BIN' '$REMOTE_MICROBENCH_DIR/test_gemini.py' --size '$SIZE_MB' --iterations '$ITERATIONS' --rank 1 --world_size '$WORLD_SIZE' --master_ip '$MASTER_IP' --master_port '$MASTER_PORT'" \
    > "$REMOTE_LOG" 2>&1 &
pids+=("$!")

{
    echo "Starting Gemini rank 0 on local node"
    echo "Writing rank 0 log to $OUTPUT_LOG"
} | tee -a "$LAUNCH_LOG"

timeout "$GEMINI_TIMEOUT" "$PYTHON_BIN" "$MICROBENCH_DIR/test_gemini.py" \
    --size "$SIZE_MB" \
    --iterations "$ITERATIONS" \
    --rank 0 \
    --world_size "$WORLD_SIZE" \
    --master_ip "$MASTER_IP" \
    --master_port "$MASTER_PORT" \
    > "$OUTPUT_LOG" 2>&1 &
pids+=("$!")

gemini_status=0
for pid in "${pids[@]}"; do
    wait "$pid" || gemini_status=$?
done

if [ "$gemini_status" -ne 0 ]; then
    {
        echo "Gemini multinode microbenchmark failed."
        echo "Rank 0 log: $OUTPUT_LOG"
        echo "Rank 1 log: $REMOTE_LOG"
    } | tee -a "$LAUNCH_LOG"
    exit "$gemini_status"
fi

echo "Gemini multinode microbenchmark completed." | tee -a "$LAUNCH_LOG"
