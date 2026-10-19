#!/usr/bin/env bash
# Requires the committed frozen DESIGN.md.
set -euo pipefail
phase=${1:?discover or heldout}
DESIGN_COMMIT=${2:?Full frozen source commit SHA is required}
TASK=${3:-numeric}
. "workspace/env.sh"
export TMPDIR="workspace/exp07/tmp" XDG_CACHE_HOME="workspace/exp07/cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "workspace/mech-interp-lab"
EXP=experiments/07_numeric_selection
DATA="workspace/exp07/stimuli"
OUT="workspace/exp07/stage2"
if [[ "$TASK" == table ]]; then
  python -c 'import json,pathlib; g=json.loads((pathlib.Path('workspace') / "./exp07/stage2/task_gate.json").read_text()); assert g["selected_task"]=="table" and not g["H5_passed"], "Numeric H5 must fail before table fallback"'
  DATA="workspace/exp07/table_stimuli"
  OUT="workspace/exp07/stage2_table"
fi
mkdir -p "$OUT"
run_gpu() {
  srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=02:00:00 \
    python -u "$EXP/stage2.py" "$@" --task "$TASK" --stimuli "$DATA" --output "$OUT" --expected-commit "$DESIGN_COMMIT"
}
run_cpu() {
  srun -p gpu --cpus-per-task=4 --mem=24G --time=00:10:00 \
    python -u "$EXP/stage2.py" "$@" --task "$TASK" --stimuli "$DATA" --output "$OUT" --expected-commit "$DESIGN_COMMIT"
}
wave() {
  local mode=$1
  shift
  local pids=()
  for seed in "$@"; do
    run_gpu "$mode" --seed "$seed" > "$OUT/${mode}_${seed}.log" 2>&1 &
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  return "$status"
}
if [[ "$phase" == discover ]]; then
  if [[ "$TASK" == table && ! -d "$DATA" ]]; then
    srun -p gpu --cpus-per-task=4 --mem=24G --time=00:10:00 \
      python "$EXP/table_stimuli.py" --output "$DATA"
  fi
  wave dev 20261101 20261102 20261103 20261104
  run_cpu task_gate > "$OUT/task_gate.log" 2>&1
  wave residual 20261101 20261102 20261103 20261104
  run_cpu bounds > "$OUT/bounds.log" 2>&1
  wave scan 20261101 20261102 20261103 20261104
  run_gpu freeze > "$OUT/freeze.log" 2>&1
  cat "$OUT/freeze.log"
  echo 'STOP: commit the FREEZE file before heldout.'
elif [[ "$phase" == heldout ]]; then
  wave heldout 20261201 20261202 20261203 20261204
  wave heldout 20261205 20261206 20261207 20261208
  run_cpu aggregate > "$OUT/aggregate.log" 2>&1
else
  echo 'Expected discover or heldout' >&2
  exit 2
fi
