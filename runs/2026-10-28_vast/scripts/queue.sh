#!/usr/bin/env bash
# Tonight's camouflage queue (not part of the repo). Runs after run_attacks.sh --name additive_seed42.
set -uo pipefail
cd ~/rosef26
PY=.venv/bin/python
ORIG=$(PYTHONPATH=scripts $PY -c "from reapply_edits import original_snapshot; print(original_snapshot('artifacts/ddo_llama3_seed42/edits.safetensors'))")
atk() { local name=$1 tag=$2; shift 2
  [[ -e results/$name/$tag.json ]] && return
  echo "== $name/$tag ($(date -u +%T))"
  $PY scripts/attack.py --model models/$name --tag $name/$tag --edits artifacts/$name/edits.safetensors "$@"; }
ben() { [[ -e results/benign/$1.json ]] || { echo "== benign/$1 ($(date -u +%T))"; $PY scripts/benign.py --model "$2" --tag benign/$1; }; }

ben original "$ORIG"
ben ddo_llama3_seed42 models/ddo_llama3_seed42
ben additive_seed42 models/additive_seed42
atk additive_seed42 gate_up_cos_m10_ddo --undo gate_up_cos --m 10 --layers 6-14

for v in "noise_seed42 --up-norm 0.8 --down-norm 0.8" "split2_seed42 --split 2 --up-norm 0.8 --down-norm 0.8"; do
  set -- $v; name=$1; shift
  [[ -d artifacts/$name ]] || { echo "== camouflage $name ($(date -u +%T))"; $PY scripts/camouflage.py --name $name "$@" || continue; }
  ben $name models/$name
  atk $name defended
  atk $name gate_up_cos_ddo --undo gate_up_cos --layers 6-14
  atk $name gate_up_cos_m10_ddo --undo gate_up_cos --m 10 --layers 6-14
  atk $name oracle --undo oracle
done
echo "== queue done ($(date -u +%T))"
