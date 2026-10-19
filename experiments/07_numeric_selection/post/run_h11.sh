#!/usr/bin/env bash
set -euo pipefail
. "workspace/env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "workspace/mech-interp-lab"
commit=$(git rev-parse HEAD)
EXP=experiments/07_numeric_selection
OUT="workspace/exp07/stage6/h11"
mkdir -p "$OUT"
common=(--expected-commit "$commit" --stimuli "workspace/exp07/stimuli" --h7-output "workspace/exp07/stage2" --output "$OUT")
cpu() {
 srun -p gpu --cpus-per-task=2 --mem=8G --time=00:10:00 --job-name="exp07-h11-$1" python -u "$EXP/post/h11_transfer.py" "$1" "${common[@]}" > "$OUT/$1.log" 2>&1
}
wave() {
 local phase=$1
 shift
 local pids=() status=0
 for seed in "$@"; do
  srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=01:00:00 --job-name="exp07-h11-$phase-$seed" python -u "$EXP/post/h11_transfer.py" "$phase" --seed "$seed" "${common[@]}" > "$OUT/${phase}_${seed}.log" 2>&1 &
  pids+=("$!")
 done
 for pid in "${pids[@]}"; do wait "$pid" || status=1; done
 return "$status"
}
cpu prepare
wave clean 20261201 20261202 20261203 20261204
wave clean 20261205 20261206 20261207 20261208
cpu gate
if python -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["passed"])' "$OUT/gate.json"; then
 wave run 20261201 20261202 20261203 20261204
 wave run 20261205 20261206 20261207 20261208
fi
cpu aggregate
