#!/usr/bin/env bash
# OBLITERATUS (pinned 43b5d6d, method advanced, bf16, no quantization) on three models, then our judge.
set -uo pipefail
cd ~/rosef26
PY=.venv/bin/python
# Its admission check counts only "free" RAM, not reclaimable page cache (26 GB available on this box).
export OBLITERATUS_TELEMETRY=0 OBLITERATUS_PHASE_ADMISSION=off HF_HOME=$PWD/models/hf_cache
ORIG=$(PYTHONPATH=scripts $PY -c "from reapply_edits import original_snapshot; print(original_snapshot('artifacts/ddo_llama3_seed42/edits.safetensors'))")
for name in original ddo_llama3_seed42 additive_seed42; do
  src=$ORIG
  if [[ $name != original ]]; then
    src=models/$name
    [[ -d $src ]] || $PY scripts/reapply_edits.py --edits artifacts/$name/edits.safetensors --out $src || continue
  fi
  out=models/obl_advanced_$name
  if [[ ! -e results/obliteratus/advanced_$name.json ]]; then
    echo "== obliteratus advanced on $name ($(date -u +%T))"
    [[ -d $out ]] || .venv-obl/bin/python obl_run.py obliterate "$src" --method advanced --dtype bfloat16 --seed 42 --output-dir $out || { echo "obliteratus failed on $name"; rm -rf $out; continue; }
    ls $out
    # Saved by transformers 5.x; our judge runs 4.51. Same weights only: check names, reuse source config/tokenizer.
    PYTHONPATH=scripts $PY -c "
import sys; from collect_artifacts import tensor_files
a, b = tensor_files(sys.argv[1]).keys(), tensor_files(sys.argv[2]).keys()
assert a == b, ('tensor names differ', sorted(a ^ b)[:5]); print('tensor names match:', len(a))" "$src" $out || { echo "checkpoint mismatch on $name"; continue; }
    cp "$src"/config.json "$src"/tokenizer.json "$src"/tokenizer_config.json "$src"/special_tokens_map.json "$src"/generation_config.json $out/ 2>/dev/null
    $PY scripts/attack.py --model $out --tag obliteratus/advanced_$name --no-rfa || continue
  fi
  [[ -e results/benign/obl_advanced_$name.json ]] || $PY scripts/benign.py --model $out --tag benign/obl_advanced_$name
  rm -rf $out
  [[ $name != original ]] && rm -rf $src
  df -h ~ | tail -1
done
echo "== queue3 done ($(date -u +%T))"
