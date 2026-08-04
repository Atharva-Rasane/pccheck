#!/bin/bash
set -euo pipefail

MODE="${1:-baseline}"

PCCHECK_HOME="${PCCHECK_HOME:-$HOME/pccheck}"
TRANSFORMERS_DIR="${TRANSFORMERS_DIR:-$HOME/transformers}"
SCRIPT_DIR="$TRANSFORMERS_DIR/examples/pytorch/language-modeling"
TRANSFORMERS_SRC_DIR="$TRANSFORMERS_DIR/src/transformers"
LLM_DIR="$PCCHECK_HOME/checkpoint_eval/models/llm_distr"

PYTHON="${PYTHON:-/opt/conda/bin/python}"
HOSTFILE="${HOSTFILE:-$PCCHECK_HOME/hostfile}"
NUM_GPUS="${NUM_GPUS:-1}"
NUM_NODES="${NUM_NODES:-}"
MASTER_PORT="${MASTER_PORT:-1234}"
MASTER_ADDR="${MASTER_ADDR:-}"
GEMINI_MASTER_PORT="${GEMINI_MASTER_PORT:-1235}"
SSH_PORT="${SSH_PORT:-22}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

if [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
    NCCL_SOCKET_IFNAME="$(awk '$2 == "00000000" { print $1; exit }' /proc/net/route)"
fi
if [ -z "$NCCL_SOCKET_IFNAME" ]; then
    echo "Unable to detect the default network interface; set NCCL_SOCKET_IFNAME explicitly."
    exit 1
fi
export HF_HOME HF_DATASETS_CACHE HUGGINGFACE_HUB_CACHE NCCL_SOCKET_IFNAME
# DeepSpeed 0.12.6 appends the configured SSH port to this variable without
# initializing it first.  Keep it defined for pdsh launches on container SSH
# ports (for example 2222).
export PDSH_SSH_ARGS_APPEND="${PDSH_SSH_ARGS_APPEND:-}"

TOKENIZER_NAME="${TOKENIZER_NAME:-facebook/opt-125m}"
TINY_OPT_CONFIG_OVERRIDES="${TINY_OPT_CONFIG_OVERRIDES:-vocab_size=50272,max_position_embeddings=128,hidden_size=128,ffn_dim=512,num_hidden_layers=2,num_attention_heads=4,word_embed_proj_dim=128,dropout=0.0,attention_dropout=0.0}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
BENCH_TOTAL_STEPS="${BENCH_TOTAL_STEPS:-8}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
CFREQ="${CFREQ:-4}"
SEED="${SEED:-1234}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output-tiny-$MODE}"
CHECKFREQ_PATH="${CHECKFREQ_PATH:-$OUTPUT_DIR/checkfreq}"
PCCHECK_CHECKPOINT_PATH="${PCCHECK_CHECKPOINT_PATH:-}"
if [ -z "$PCCHECK_CHECKPOINT_PATH" ]; then
    PCCHECK_CHECKPOINT_PATH="$OUTPUT_DIR/pccheck_checkpoint.rank-{rank}.chk"
fi
DS_CONFIG="${DS_CONFIG:-$SCRIPT_DIR/ds_config.json}"
TRAIN_FILE="${TRAIN_FILE:-$SCRIPT_DIR/tiny_train.txt}"

PCCHECK_LIB="${PCCHECK_LIB:-$PCCHECK_HOME/checkpoint_eval/pccheck/libtest_ssd.so}"
MAX_ASYNC="${MAX_ASYNC:-2}"
NUM_THREADS="${NUM_THREADS:-2}"
CHUNK_SIZE_MB="${CHUNK_SIZE_MB:-4}"
SYNC_LLM_FILES="${SYNC_LLM_FILES:-1}"
export PCCHECK_CHECKPOINT_PATH
export BENCH_SEQUENCE_LENGTH="$BLOCK_SIZE"

usage() {
    echo "Usage: $0 [baseline|gemini|pccheck|gpm|cfreq]"
    echo
    echo "Common env overrides:"
    echo "  HOSTFILE=$HOSTFILE"
    echo "  MASTER_ADDR=<rank0 internal IP>"
    echo "  NUM_NODES=$NUM_NODES NUM_GPUS=$NUM_GPUS"
    echo "  BENCH_TOTAL_STEPS=$BENCH_TOTAL_STEPS WARMUP_STEPS=$WARMUP_STEPS CFREQ=$CFREQ"
    echo "  PCCHECK_CHECKPOINT_PATH=$PCCHECK_CHECKPOINT_PATH"
}

case "$MODE" in
    --copy-only) ;;
    baseline) TARGET_SCRIPT="run_clm_pp.py" ;;
    gemini) TARGET_SCRIPT="run_clm_pp_gemini.py" ;;
    pccheck) TARGET_SCRIPT="run_clm_pp_pccheck.py" ;;
    gpm) TARGET_SCRIPT="run_clm_pp_gpm.py" ;;
    cfreq) TARGET_SCRIPT="run_clm_pp_cfreq.py" ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage
        echo "Unknown mode: $MODE"
        exit 1
        ;;
esac

copy_llm_files() {
    mkdir -p "$SCRIPT_DIR"
    rm -f "$SCRIPT_DIR/deepspeed.py" "$SCRIPT_DIR/trainer_pp.py"
    rm -f "$SCRIPT_DIR"/__pycache__/deepspeed*.pyc "$SCRIPT_DIR"/__pycache__/trainer_pp*.pyc 2>/dev/null || true

    cp "$LLM_DIR"/bloom_ds.py "$SCRIPT_DIR"/
    cp "$LLM_DIR"/convert_to_ds.py "$SCRIPT_DIR"/
    cp "$LLM_DIR"/llama_ds.py "$SCRIPT_DIR"/
    cp "$LLM_DIR"/opt_ds.py "$SCRIPT_DIR"/
    cp "$LLM_DIR"/run_clm_pp.py "$SCRIPT_DIR"/
    cp "$LLM_DIR"/run_clm_pp_cfreq.py "$SCRIPT_DIR"/
    cp "$LLM_DIR"/run_clm_pp_gemini.py "$SCRIPT_DIR"/
    cp "$LLM_DIR"/run_clm_pp_gpm.py "$SCRIPT_DIR"/
    cp "$LLM_DIR"/run_clm_pp_pccheck.py "$SCRIPT_DIR"/
    cp "$LLM_DIR"/ds_config.json "$SCRIPT_DIR"/
    write_tiny_train_file "$TRAIN_FILE"

    mkdir -p "$TRANSFORMERS_SRC_DIR"
    cp "$LLM_DIR"/trainer_pp.py "$TRANSFORMERS_SRC_DIR"/
    cp "$LLM_DIR"/deepspeed.py "$TRANSFORMERS_SRC_DIR"/
    cp "$PCCHECK_HOME"/checkpoint_eval/models/opt/__init__.py "$TRANSFORMERS_SRC_DIR"/

    deepspeed_path="$("$PYTHON" -c 'import deepspeed; print(deepspeed.__path__[0])' | tail -1)"
    cp "$PCCHECK_HOME/checkpoint_eval/deepspeed/__init__.py" "$deepspeed_path"/

    patch_runtime_scripts
}

write_tiny_train_file() {
    train_path="$1"
    mkdir -p "$(dirname "$train_path")"
    cat > "$train_path" <<'EOF'
PCcheck tiny language model smoke test.
This corpus is intentionally small and repetitive.
It exercises tokenization, a short causal language modeling pass, DeepSpeed pipeline setup, and checkpoint hooks.
The model used by run.sh is randomly initialized from a very small OPT configuration.
This file is not intended for accuracy or convergence.

PCcheck tiny language model smoke test.
This corpus is intentionally small and repetitive.
It exercises tokenization, a short causal language modeling pass, DeepSpeed pipeline setup, and checkpoint hooks.
The model used by run.sh is randomly initialized from a very small OPT configuration.
This file is not intended for accuracy or convergence.

PCcheck tiny language model smoke test.
This corpus is intentionally small and repetitive.
It exercises tokenization, a short causal language modeling pass, DeepSpeed pipeline setup, and checkpoint hooks.
The model used by run.sh is randomly initialized from a very small OPT configuration.
This file is not intended for accuracy or convergence.

PCcheck tiny language model smoke test.
This corpus is intentionally small and repetitive.
It exercises tokenization, a short causal language modeling pass, DeepSpeed pipeline setup, and checkpoint hooks.
The model used by run.sh is randomly initialized from a very small OPT configuration.
This file is not intended for accuracy or convergence.
EOF
}

patch_runtime_scripts() {
    for script in "$SCRIPT_DIR"/run_clm_pp*.py; do
        [ -f "$script" ] || continue
        sed -i \
            -e '/use_auth_token=True if model_args\.use_auth_token else None,/d' \
            -e '/"use_auth_token": True if model_args\.use_auth_token else None,/d' \
            "$script"
    done
}

copy_llm_files_remote() {
    host="$1"
    ssh -p "$SSH_PORT" "$host" "mkdir -p '$SCRIPT_DIR' '$TRANSFORMERS_SRC_DIR'"
    ssh -p "$SSH_PORT" "$host" "rm -f '$SCRIPT_DIR/deepspeed.py' '$SCRIPT_DIR/trainer_pp.py'; rm -f '$SCRIPT_DIR'/__pycache__/deepspeed*.pyc '$SCRIPT_DIR'/__pycache__/trainer_pp*.pyc 2>/dev/null || true"

    scp -P "$SSH_PORT" \
        "$SCRIPT_DIR"/bloom_ds.py \
        "$SCRIPT_DIR"/convert_to_ds.py \
        "$SCRIPT_DIR"/llama_ds.py \
        "$SCRIPT_DIR"/opt_ds.py \
        "$SCRIPT_DIR"/run_clm_pp.py \
        "$SCRIPT_DIR"/run_clm_pp_cfreq.py \
        "$SCRIPT_DIR"/run_clm_pp_gemini.py \
        "$SCRIPT_DIR"/run_clm_pp_gpm.py \
        "$SCRIPT_DIR"/run_clm_pp_pccheck.py \
        "$SCRIPT_DIR"/ds_config.json \
        "$host:$SCRIPT_DIR"/

    scp -P "$SSH_PORT" \
        "$TRANSFORMERS_SRC_DIR"/trainer_pp.py \
        "$TRANSFORMERS_SRC_DIR"/deepspeed.py \
        "$TRANSFORMERS_SRC_DIR"/__init__.py \
        "$host:$TRANSFORMERS_SRC_DIR"/

    write_tiny_train_file_remote "$host"

    ssh -p "$SSH_PORT" "$host" "deepspeed_path=\"\$('$PYTHON' -c 'import deepspeed; print(deepspeed.__path__[0])' | tail -1)\" && cp '$PCCHECK_HOME/checkpoint_eval/deepspeed/__init__.py' \"\$deepspeed_path\"/"
}

write_tiny_train_file_remote() {
    host="$1"
    train_dir="$(dirname "$TRAIN_FILE")"
    ssh -p "$SSH_PORT" "$host" "mkdir -p '$train_dir'"
    scp -P "$SSH_PORT" "$TRAIN_FILE" "$host:$TRAIN_FILE"
}

verify_train_file() {
    if [ ! -f "$TRAIN_FILE" ]; then
        echo "Missing train file on local node: $TRAIN_FILE"
        exit 1
    fi
}

verify_train_file_remote() {
    host="$1"
    ssh -p "$SSH_PORT" "$host" "test -f '$TRAIN_FILE'" || {
        echo "Missing train file on $host: $TRAIN_FILE"
        exit 1
    }
}

write_deepspeed_env() {
    {
        echo "GEMINI_MASTER_ADDR=$MASTER_ADDR"
        echo "GEMINI_MASTER_PORT=$GEMINI_MASTER_PORT"
        echo "PCCHECK_COORDINATOR=$MASTER_ADDR"
        echo "HF_HOME=$HF_HOME"
        echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
        echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
        echo "PCCHECK_CHECKPOINT_PATH=$PCCHECK_CHECKPOINT_PATH"
    } > "$HOME/.deepspeed_env"
}

write_deepspeed_env_remote() {
    host="$1"
    ssh -p "$SSH_PORT" "$host" "printf '%s\n' 'GEMINI_MASTER_ADDR=$MASTER_ADDR' 'GEMINI_MASTER_PORT=$GEMINI_MASTER_PORT' 'PCCHECK_COORDINATOR=$MASTER_ADDR' 'HF_HOME=$HF_HOME' 'HF_DATASETS_CACHE=$HF_DATASETS_CACHE' 'HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE' 'PCCHECK_CHECKPOINT_PATH=$PCCHECK_CHECKPOINT_PATH' > ~/.deepspeed_env"
}

if [ "$MODE" = "--copy-only" ]; then
    copy_llm_files
    exit 0
fi

if [ ! -d "$TRANSFORMERS_DIR" ]; then
    echo "Missing $TRANSFORMERS_DIR. Run setup_models_and_datasets.sh or set TRANSFORMERS_DIR."
    exit 1
fi

if [ ! -f "$HOSTFILE" ]; then
    echo "Missing hostfile: $HOSTFILE"
    echo "Expected format:"
    echo "  10.128.0.19 slots=1"
    echo "  10.128.0.20 slots=1"
    exit 1
fi

if [ -z "$MASTER_ADDR" ]; then
    MASTER_ADDR="$(awk 'NF && $1 !~ /^#/ {print $1; exit}' "$HOSTFILE")"
fi

if [ -z "$NUM_NODES" ]; then
    NUM_NODES="$(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' "$HOSTFILE")"
fi

if [ "$SYNC_LLM_FILES" = "1" ]; then
    echo "Copying LLM distributed files into $TRANSFORMERS_DIR"
    copy_llm_files
    write_deepspeed_env

    while read -r host _; do
        case "$host" in
            ""|\#*) continue ;;
            *)
                if [ "${host#*@}" = "$MASTER_ADDR" ]; then
                    continue
                fi
                echo "Copying LLM distributed files on $host"
                copy_llm_files_remote "$host" < /dev/null
                write_deepspeed_env_remote "$host" < /dev/null
                ;;
        esac
    done < "$HOSTFILE"
else
    write_deepspeed_env
fi

verify_train_file
while read -r host _; do
    case "$host" in
        ""|\#*) continue ;;
        *)
            if [ "${host#*@}" = "$MASTER_ADDR" ]; then
                continue
            fi
            verify_train_file_remote "$host" < /dev/null
            ;;
    esac
done < "$HOSTFILE"

mode_args=(
    --cfreq "$CFREQ"
    --bench_total_steps "$BENCH_TOTAL_STEPS"
    --warmup_steps "$WARMUP_STEPS"
)

if [ "$MODE" = "pccheck" ]; then
    mode_args+=(
        --c_lib_path "$PCCHECK_LIB"
        --max_async "$MAX_ASYNC"
        --num_threads "$NUM_THREADS"
    )
elif [ "$MODE" = "cfreq" ]; then
    mkdir -p "$CHECKFREQ_PATH"
    mode_args+=(
        --path_to_pmem "$CHECKFREQ_PATH"
    )
elif [ "$MODE" = "gemini" ]; then
    mode_args+=(
        --chunk_size_mb "$CHUNK_SIZE_MB"
    )
fi

echo "Running tiny LLM $MODE smoke test"
echo "hostfile: $HOSTFILE"
echo "master: $MASTER_ADDR:$MASTER_PORT"
echo "nodes: $NUM_NODES"
echo "script: $SCRIPT_DIR/$TARGET_SCRIPT"
echo "train file: $TRAIN_FILE"

cd "$SCRIPT_DIR"
deepspeed \
    --num_gpus "$NUM_GPUS" \
    --num_nodes "$NUM_NODES" \
    --hostfile "$HOSTFILE" \
    --ssh_port "$SSH_PORT" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    "$TARGET_SCRIPT" \
    --deepspeed "$DS_CONFIG" \
    --ds_config "$DS_CONFIG" \
    --model_type opt \
    --config_overrides "$TINY_OPT_CONFIG_OVERRIDES" \
    --tokenizer_name "$TOKENIZER_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --train_file "$TRAIN_FILE" \
    --validation_file "$TRAIN_FILE" \
    --do_train \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --block_size "$BLOCK_SIZE" \
    --max_train_samples "$MAX_TRAIN_SAMPLES" \
    --overwrite_output_dir \
    --overwrite_cache \
    --logging_steps 1 \
    --report_to none \
    --seed "$SEED" \
    "${mode_args[@]}"
