#!/bin/bash
set -euo pipefail

detect_accelerator_backend() {
	python - <<'PY'
import torch

backend = "cpu"
if torch.cuda.is_available():
	if getattr(torch.version, "hip", None):
		backend = "rocm"
	elif getattr(torch.version, "cuda", None):
		backend = "cuda"

print(backend)
PY
}

export ACCELERATOR_BACKEND=${ACCELERATOR_BACKEND:-auto}
export ENABLE_CHUNKING=true
export MAX_SEQ_LENGTH=65536
export CHUNK_STRIDE=49152
export MODEL_NAME=${MODEL_NAME:-gemma-4-26B-A4B-it-hermes-reasoning-OpenCodeInstruct}
export MODEL_OUT_DIR=${MODEL_OUT_DIR:-./output/$MODEL_NAME}
export OUTPUT_DIR=${OUTPUT_DIR:-$MODEL_OUT_DIR/results}
export ADAPTER_DIR=${ADAPTER_DIR:-$MODEL_OUT_DIR/adapter}
export NUM_EPOCHS=0.5
export CODE_MIN_TEST_SCORE=0.9
export MIX_REASONING_RATIO=0.35
export LEARNING_RATE=5e-5
# Attention-only LoRA by default to better preserve base coding ability.
export TARGET_MODULES=q_proj,k_proj,v_proj,o_proj
export REPORT_TO=tensorboard
export TENSORBOARD_LOG_DIR=${TENSORBOARD_LOG_DIR:-$MODEL_OUT_DIR/results/runs}
export START_TENSORBOARD=${START_TENSORBOARD:-true}
export TENSORBOARD_HOST=${TENSORBOARD_HOST:-127.0.0.1}
export TENSORBOARD_PORT=${TENSORBOARD_PORT:-6006}

DEFAULT_BATCH_SIZE=2
DEFAULT_GRADIENT_ACCUMULATION_STEPS=8

if [[ "$ACCELERATOR_BACKEND" == "auto" ]]; then
	ACCELERATOR_BACKEND=$(detect_accelerator_backend)
	if [[ -z "$ACCELERATOR_BACKEND" ]]; then
		echo "Failed to auto-detect accelerator backend."
		exit 1
	fi
	export ACCELERATOR_BACKEND
fi

case "$ACCELERATOR_BACKEND" in
	rocm)
		export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}
		;;
	cuda)
		export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
		DEFAULT_BATCH_SIZE=1
		DEFAULT_GRADIENT_ACCUMULATION_STEPS=16
		unset TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL || true
		;;
	cpu)
		unset TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL || true
		;;
	*)
		echo "Unsupported ACCELERATOR_BACKEND: $ACCELERATOR_BACKEND"
		echo "Expected one of: auto, rocm, cuda, cpu"
		exit 1
		;;
esac

export BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-$DEFAULT_GRADIENT_ACCUMULATION_STEPS}

TB_PID=""

cleanup() {
	if [[ -n "${TB_PID:-}" ]] && kill -0 "$TB_PID" 2>/dev/null; then
		echo "Stopping TensorBoard (PID: $TB_PID)..."
		kill "$TB_PID" 2>/dev/null || true
		wait "$TB_PID" 2>/dev/null || true
	fi
}

trap cleanup EXIT

mkdir -p "$MODEL_OUT_DIR/logs"
mkdir -p "$TENSORBOARD_LOG_DIR"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$MODEL_OUT_DIR/logs/train-$(date +%Y%m%d-%H%M%S).log"
TB_LOG_FILE="$MODEL_OUT_DIR/logs/tensorboard-$(date +%Y%m%d-%H%M%S).log"
echo "Logging to: $LOG_FILE"
echo "Accelerator backend: $ACCELERATOR_BACKEND"
echo "Model name: $MODEL_NAME"
echo "Batch config: BATCH_SIZE=$BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION_STEPS"
if [[ "$ACCELERATOR_BACKEND" == "cuda" ]]; then
	echo "CUDA allocator config: ${PYTORCH_CUDA_ALLOC_CONF:-unset}"
fi
echo "Training output dir: $OUTPUT_DIR"
LATEST_CHECKPOINT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)
if [[ -n "$LATEST_CHECKPOINT" ]]; then
	echo "Latest checkpoint detected: $LATEST_CHECKPOINT"
	echo "Resume behavior: trainer should auto-resume from this checkpoint."
else
	echo "No checkpoint detected in $OUTPUT_DIR"
	echo "Resume behavior: trainer will start a fresh run."
fi
echo "TensorBoard command: tensorboard --logdir $TENSORBOARD_LOG_DIR --host $TENSORBOARD_HOST --port $TENSORBOARD_PORT"
echo "TensorBoard URL: http://$TENSORBOARD_HOST:$TENSORBOARD_PORT"
if [[ "$TENSORBOARD_HOST" == "0.0.0.0" ]]; then
	echo "TensorBoard is exposed on all interfaces for LAN access."
fi

if [[ "$START_TENSORBOARD" == "true" ]]; then
	echo "Starting TensorBoard in background (logs: $TB_LOG_FILE)"
	tensorboard --logdir "$TENSORBOARD_LOG_DIR" --host "$TENSORBOARD_HOST" --port "$TENSORBOARD_PORT" \
		>"$TB_LOG_FILE" 2>&1 &
	TB_PID=$!
	echo "TensorBoard started with PID: $TB_PID"
fi

# Stream output in real time and also save it to a log file.
set +e
python -u ./train_lora.py 2>&1 | tee -a "$LOG_FILE"
TRAIN_EXIT_CODE=${PIPESTATUS[0]}
set -e

if [[ "$TRAIN_EXIT_CODE" -eq 0 ]]; then
	echo "Training completed successfully."
else
	echo "Training exited with code $TRAIN_EXIT_CODE."
fi

if [[ -t 0 ]]; then
	echo "Press any key to exit (TensorBoard will be stopped)..."
	read -r -n 1 -s _
	echo
else
	echo "Non-interactive shell detected; exiting now."
fi

exit "$TRAIN_EXIT_CODE"
