#!/usr/bin/env bash
# Additive seeds 43-44 (not part of the repo). Frees disk first: the build refuses under 60 GB free.
set -uo pipefail
cd ~/rosef26
PY=.venv/bin/python
rm -rf models/noise_seed42 models/split2_seed42 models/ddo_llama3_seed42 models/additive_seed42   # all rebuildable from their edits
df -h ~ | tail -1
for s in 43 44; do
  name=additive_seed$s
  [[ -e artifacts/$name/manifest.json ]] || scripts/build_ddo_llama3.sh --seed $s --name $name -- --compile_mode additive || continue
  for t in "defended" "gate_up_cos_ddo --undo gate_up_cos --layers 6-14"; do
    set -- $t; tag=$1; shift
    [[ -e results/$name/$tag.json ]] || { echo "== $name/$tag ($(date -u +%T))"; $PY scripts/attack.py --model models/$name --tag $name/$tag --edits artifacts/$name/edits.safetensors "$@"; }
  done
  [[ -e results/benign/$name.json ]] || $PY scripts/benign.py --model models/$name --tag benign/$name
  rm -rf models/$name
done
echo "== queue2 done ($(date -u +%T))"
