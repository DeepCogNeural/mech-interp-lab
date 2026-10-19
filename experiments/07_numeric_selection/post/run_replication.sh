#!/usr/bin/env bash
# sanity: rerun original held-out seed 20261201 into replication/sanity and compare.
# replicate PREREG_COMMIT: fresh seeds, only after ADDENDUM_REPLICATION.md is committed.
set -euo pipefail
phase=${1:?sanity or replicate}
. "workspace/env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "workspace/mech-interp-lab"
commit=$(git rev-parse HEAD)
EXP=experiments/07_numeric_selection
OUT="workspace/exp07/replication"
mkdir -p "$OUT/sanity"
common=(--expected-commit "$commit" --h7-output "workspace/exp07/stage2" --output "$OUT")
gpu() {
 srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=02:00:00 --job-name="exp07-rep-$1-$2" \
  python -u "$EXP/post/replication.py" "$1" --seed "$2" "${@:3}" "${common[@]}"
}
cpu() {
 srun -p gpu --cpus-per-task=2 --mem=8G --time=00:20:00 --job-name="exp07-rep-$1" \
  python -u "$EXP/post/replication.py" "$1" "${@:2}" "${common[@]}"
}
if [[ "$phase" == sanity ]]; then
 gpu sanity 20261201 --stimuli "workspace/exp07/stimuli" > "$OUT/sanity/sanity.log" 2>&1
 cpu sanity_check --stimuli "workspace/exp07/stimuli" > "$OUT/sanity/sanity_check.log" 2>&1
 cat "$OUT/sanity/sanity_check.log"
elif [[ "$phase" == replicate ]]; then
 rep=(--prereg-commit "${2:?committed pre-registration SHA}" --stimuli "$OUT/stimuli")
 cpu prepare "${rep[@]}" > "$OUT/prepare.log" 2>&1
 wave() {
  local pids=() status=0
  for seed in "$@"; do
   gpu run "$seed" "${rep[@]}" > "$OUT/run_${seed}.log" 2>&1 &
   pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  return "$status"
 }
 wave 20261301 20261302 20261303 20261304
 wave 20261305 20261306 20261307 20261308
 cpu aggregate "${rep[@]}" > "$OUT/aggregate.log" 2>&1
 cat "$OUT/aggregate.log"
else
 echo 'Expected sanity or replicate' >&2
 exit 2
fi
