#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-runs}"
mkdir -p "$OUTPUT_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUTPUT_ROOT/l_mnist_pipeline_$STAMP"
mkdir -p "$RUN_DIR"

CONDA_BASE="$(conda info --base)"
PYTHON_BIN="$CONDA_BASE/envs/pytorch-3.10/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "ERROR: python executable not found: $PYTHON_BIN" >&2
	exit 1
fi

"$PYTHON_BIN" -c "import matplotlib, torch, torchvision" >/dev/null 2>&1 || {
	echo "ERROR: matplotlib/torch/torchvision are not importable in $PYTHON_BIN" >&2
	exit 1
}

GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-10000}"
GPU_IDS="${GPU_IDS:-}"

if [[ -z "$GPU_IDS" ]]; then
	if command -v nvidia-smi >/dev/null 2>&1; then
		GPU_IDS="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
			| awk -F',' -v limit="$GPU_MEM_THRESHOLD_MB" '{gsub(/ /, "", $1); gsub(/ /, "", $2); if ($2 + 0 <= limit) print $1}' \
			| paste -sd, -)"
	fi
fi

if [[ -z "$GPU_IDS" ]]; then
	echo "ERROR: no GPUs selected. Set GPU_IDS, e.g. GPU_IDS=0,2,3,5 $0" >&2
	exit 1
fi

COMMAND=(
	"$PYTHON_BIN" -u l_mnist_training_pipeline.py
	--output-dir "$RUN_DIR"
	--gpu-ids "$GPU_IDS"
)

echo "Launching background run in: $RUN_DIR"
echo "Selected GPUs: $GPU_IDS"
echo "GPU auto-select memory threshold: ${GPU_MEM_THRESHOLD_MB} MiB"

printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
printf '%q ' "${COMMAND[@]}" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"

setsid env PYTHONUNBUFFERED=1 "${COMMAND[@]}" > "$RUN_DIR/live.log" 2>&1 < /dev/null &
PID=$!

echo "PID: $PID"
echo "$PID" > "$RUN_DIR/pid.txt"

echo "Live log: $RUN_DIR/live.log"
echo "Tail with: tail -f $RUN_DIR/live.log"
echo "Per-run progress: $RUN_DIR/progress/*.json"
echo "Progress peek: watch -n 5 'tail -n 20 $RUN_DIR/live.log; ls -1 $RUN_DIR/progress 2>/dev/null | tail'"
