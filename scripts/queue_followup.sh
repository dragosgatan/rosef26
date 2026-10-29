#!/usr/bin/env bash
# Follow-up tests for the paper; finished steps are skipped, so it can be rerun after a crash.
#   1. replace mode with the confusion filter off, seeds 43 and 44 (the two builds with no protection):
#      does a full confusion pool fix them?
#   2. additive mode, seeds 45 and 46: five additive seeds against five replace seeds.
# Each build: defended RFA, the gate_up_cos detector, the harmless check, then its model is deleted
# (it can be rebuilt from artifacts/<name>/edits.safetensors).
# usage: scripts/queue_followup.sh   (HF_TOKEN set, from the repo root, inside tmux)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[[ -n ${HF_TOKEN:-} ]] || { echo "error: HF_TOKEN is not set" >&2; exit 1; }

one() {  # name, then build flags
  local name=$1; shift
  echo "== $name ($(date -u +%T)); disk free: $(df -h . | awk 'NR==2 {print $4}')"
  [[ -e artifacts/$name/manifest.json ]] || scripts/build_ddo_llama3.sh --name "$name" "$@" || return
  [[ -d models/$name ]] || $PY scripts/reapply_edits.py --edits artifacts/$name/edits.safetensors --out models/$name
  for t in "defended" "gate_up_cos_ddo --undo gate_up_cos --layers 6-14"; do
    set -- $t; local tag=$1; shift
    [[ -e results/$name/$tag.json ]] || $PY scripts/attack.py --model models/$name --tag $name/$tag \
      --edits artifacts/$name/edits.safetensors "$@"
  done
  [[ -e results/benign/$name.json ]] || $PY scripts/benign.py --model models/$name --tag benign/$name
  rm -rf models/$name
}

one nofilter_seed43 --seed 43 --no-confusion-filter
one nofilter_seed44 --seed 44 --no-confusion-filter
one additive_seed45 --seed 45 -- --compile_mode additive
one additive_seed46 --seed 46 -- --compile_mode additive
echo "== followup done ($(date -u +%T))"
