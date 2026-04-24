#!/usr/bin/env sh
set -eu

PROJECT_DIR="/data/home/yxzhou/jydai/work/IFD_T"
SCRIPT_DIR="$PROJECT_DIR/script"
RUN_SCRIPT="$SCRIPT_DIR/eval_baseline_no_tools.sh"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/eval/output/baseline_no_tools}"

LOG_DIR="${LOG_DIR:-$OUTPUT_DIR}"
PID_FILE="${PID_FILE:-$OUTPUT_DIR/eval_baseline_no_tools.pid}"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/eval_baseline_no_tools.log}"

mkdir -p "$LOG_DIR"

if [ ! -f "$RUN_SCRIPT" ]; then
  echo "[ERROR] run script not found: $RUN_SCRIPT" >&2
  exit 1
fi

if [ -f "$PID_FILE" ]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[ERROR] process already running: pid=$OLD_PID" >&2
    echo "        log=$LOG_FILE" >&2
    exit 2
  fi
fi

cd "$PROJECT_DIR"

# Start in a new session and ignore SIGHUP so it can survive SSH disconnect.
setsid nohup sh "$RUN_SCRIPT" < /dev/null > "$LOG_FILE" 2>&1 &
PID="$!"
echo "$PID" > "$PID_FILE"

echo "[OK] launched background eval"
echo "[OK] pid=$PID"
echo "[OK] log=$LOG_FILE"
echo "[OK] pid_file=$PID_FILE"
