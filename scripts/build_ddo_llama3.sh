#!/usr/bin/env bash
# Build the DDO-defended Llama-3-8B-Instruct with the authors' published config.
# usage: scripts/build_ddo_llama3.sh --seed 42 [--out DIR] [--name NAME] [-- extra apply flags]
#   e.g. --name additive_seed42 -- --compile_mode additive   (NAME defaults to ddo_llama3_seed42)
set -euo pipefail

MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
# Pin to one commit. Get the current one with:
#   .venv/bin/python -c "from huggingface_hub import model_info; print(model_info('meta-llama/Meta-Llama-3-8B-Instruct').sha)"
MODEL_REVISION=8afb486c1db24fe5011ec46dfbe5b5dccdb575c2
DDO_COMMIT=a26f23e0149a0050b822a4913a2a147f67553810
MIN_VRAM_MIB=24000
MIN_DISK_GB=60

die() { echo "error: $*" >&2; exit 1; }

SEED= OUT= NAME= EXTRA=()
while (($#)); do
  case $1 in
    --seed) SEED=${2:?--seed needs a value}; shift 2 ;;
    --out) OUT=${2:?--out needs a value}; shift 2 ;;
    --name) NAME=${2:?--name needs a value}; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) die "unknown argument $1 (usage: --seed N [--out DIR] [--name NAME] [-- apply flags])" ;;
  esac
done
[[ $SEED =~ ^[0-9]+$ ]] || die "--seed N is required"

REPO=$(realpath "$(dirname "$0")/..")
NAME=${NAME:-ddo_llama3_seed$SEED}
OUT=$(realpath -m "${OUT:-$REPO/models/$NAME}")
ART=$REPO/artifacts/$NAME
PY=$REPO/.venv/bin/python
[[ -e $ART/manifest.json ]] && die "$ART already holds a finished build; move it away first"

mkdir -p "$ART"
exec > >(tee -a "$ART/build.log") 2>&1
echo "== build seed $SEED at $(date -u +%FT%TZ) =="

export HF_HOME=$REPO/models/hf_cache
export CUBLAS_WORKSPACE_CONFIG=:16:8
# Lets the allocator reuse fragmented memory on a 24 GB card; does not change results.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------- preflight
[[ -x $PY ]] || die "no .venv; run scripts/setup_env.sh first"
[[ $MODEL_REVISION =~ ^[0-9a-f]{40}$ ]] \
  || die "set MODEL_REVISION at the top of $0 (the command to get it is next to it)"
[[ $(git -C "$REPO/third_party/ddo" rev-parse HEAD) == "$DDO_COMMIT" ]] \
  || die "third_party/ddo is not at $DDO_COMMIT"
"$PY" - <<'EOF' || die "the installed ddo_defense is not the patched one; rerun scripts/setup_env.sh"
import inspect, ddo_defense.defense.optimizer as o
assert "_confusion_term" in inspect.getsource(o)
EOF

[[ -n ${HF_TOKEN:-} ]] || die "HF_TOKEN is not set"
"$PY" - "$MODEL_ID" "$MODEL_REVISION" <<'EOF'
import sys
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.errors import GatedRepoError, RevisionNotFoundError
repo, rev = sys.argv[1:]
try:
    get_hf_file_metadata(hf_hub_url(repo, "config.json", revision=rev))
except GatedRepoError:
    sys.exit(f"error: no access to {repo}. Accept the licence at https://huggingface.co/{repo} with the account that owns HF_TOKEN.")
except RevisionNotFoundError:
    sys.exit(f"error: {repo} has no revision {rev}")
print(f"licence ok: {repo}@{rev}")
EOF

VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0) \
  || die "nvidia-smi failed; is there a GPU?"
((VRAM >= MIN_VRAM_MIB)) || die "GPU 0 has ${VRAM} MiB; need at least ${MIN_VRAM_MIB} MiB"

free_gb() {  # free space on the filesystem that will hold $1
  local p=$1
  while [[ ! -e $p ]]; do p=$(dirname "$p"); done
  df -P -BG "$p" | awk 'NR == 2 { sub("G", "", $4); print $4 }'
}
for p in "$HF_HOME" "$OUT"; do
  ((MIN_DISK_GB <= $(free_gb "$p"))) || die "only $(free_gb "$p") GB free for $p; need $MIN_DISK_GB"
done

git -C "$REPO" check-ignore -q "$OUT" 2>/dev/null \
  || die "$OUT is not gitignored, so the checkpoint could be committed; use a path under models/"
if [[ -d $OUT && -n $(ls -A "$OUT") ]]; then die "$OUT is not empty"; fi
echo "preflight ok: GPU ${VRAM} MiB, output $OUT"

# -------------------------------------------------------------------- build
# The CLI has no --revision flag, so download the pinned snapshot and pass its folder.
SNAP=$("$PY" - "$MODEL_ID" "$MODEL_REVISION" <<'EOF'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2],
                        allow_patterns=["*.json", "model-*.safetensors"]))
EOF
)
echo "original model: $SNAP"

CMD=("$PY" -m ddo_defense.cli apply
  --model_path "$SNAP"
  --layer_start 6 --layer_end 15
  --init_beta 1.30 --init_scale 0.463 --confusion_lambda 1.12
  --lr 0.0026 --epochs 2 --compile_mode replace
  --seed "$SEED" --save_path "$OUT" "${EXTRA[@]}")
echo "running: ${CMD[*]}"
# Offline so nothing can silently fetch a different model revision.
HF_HUB_OFFLINE=1 "${CMD[@]}"

"$PY" "$REPO/scripts/collect_artifacts.py" \
  --original "$SNAP" --defended "$OUT" --artifacts "$ART" \
  --seed "$SEED" --model-id "$MODEL_ID" --model-revision "$MODEL_REVISION" \
  --command "${CMD[*]}"
echo "== done: model in $OUT, record in $ART =="
