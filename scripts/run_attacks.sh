#!/usr/bin/env bash
# Run every attack on one build; finished runs are skipped, so it can be rerun after a crash.
# usage: scripts/run_attacks.sh [--seed 42]
set -euo pipefail

SEED=42
[[ ${1:-} == --seed ]] && SEED=${2:?--seed needs a value}

REPO=$(realpath "$(dirname "$0")/..")
PY=$REPO/.venv/bin/python
EDITS=$REPO/artifacts/ddo_llama3_seed$SEED/edits.safetensors
DEF=$REPO/models/ddo_llama3_seed$SEED
export HF_HOME=$REPO/models/hf_cache PYTHONUNBUFFERED=1

[[ -n ${HF_TOKEN:-} ]] || { echo "error: HF_TOKEN is not set (LlamaGuard-2 is gated too)" >&2; exit 1; }
[[ -d $DEF ]] || "$PY" "$REPO/scripts/reapply_edits.py" --edits "$EDITS" --out "$DEF"
ORIG=$(PYTHONPATH=$REPO/scripts "$PY" -c "import sys; from reapply_edits import original_snapshot; print(original_snapshot(sys.argv[1]))" "$EDITS")

run() {
  local tag=seed$SEED/$1; shift
  if [[ -e $REPO/results/$tag.json ]]; then echo "== skip $tag (done)"; return; fi
  echo "== $tag ($(date -u +%T))"
  "$PY" "$REPO/scripts/attack.py" --tag "$tag" --edits "$EDITS" "$@"
}

run undefended            --model "$ORIG"
run defended              --model "$DEF"
run oracle                --model "$DEF" --undo oracle
run gate_up_cos_ddo       --model "$DEF" --undo gate_up_cos --layers 6-14   # Table 21 setup
run gate_up_cos_all       --model "$DEF" --undo gate_up_cos
run down_norm_ddo         --model "$DEF" --undo down_norm --layers 6-14
run norm_ratio_ddo        --model "$DEF" --undo norm_ratio --layers 6-14
echo "== done: results/seed$SEED/"
