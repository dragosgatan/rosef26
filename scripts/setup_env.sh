#!/usr/bin/env bash
# Create .venv with the authors' tested versions and the patched DDO library.
# usage: scripts/setup_env.sh        (PYTHON=python3.x to pick the interpreter)
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
DDO_COMMIT=a26f23e0149a0050b822a4913a2a147f67553810
PATCH=$REPO/patches/ddo-24gb-vram.patch
PYTHON=${PYTHON:-python3}

die() { echo "error: $*" >&2; exit 1; }

"$PYTHON" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] <= (3, 12))' \
  || die "torch 2.3.0 needs Python 3.10-3.12; rerun with PYTHON=python3.x"
[[ $(git -C "$REPO/third_party/ddo" rev-parse HEAD 2>/dev/null) == "$DDO_COMMIT" ]] \
  || die "third_party/ddo is not at $DDO_COMMIT; run: git submodule update --init"

"$PYTHON" -m venv "$REPO/.venv"
PY=$REPO/.venv/bin/python
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$REPO/requirements.txt"
# nnsight 0.3.7 declares torch>=2.4.0, but the authors tested it with torch 2.3.0.
# --no-deps stops pip from replacing torch; its other dependencies are in requirements.txt.
"$PY" -m pip install --no-deps nnsight==0.3.7

# Install the authors' code from a patched copy, so third_party/ddo stays untouched.
SRC=$(mktemp -d)
trap 'rm -rf "$SRC"' EXIT
git -C "$REPO/third_party/ddo" archive "$DDO_COMMIT" | tar -x -C "$SRC"
(cd "$SRC" && git apply "$PATCH")
"$PY" -m pip install --no-deps "$SRC"

# The authors' CPU test suite. One test reads the prompt data from this fixed path.
DDO_CACHE_DIR=/tmp/ddo_datacheck "$PY" -c "from ddo_defense.data import prefetch; prefetch()"
(cd "$SRC" && "$PY" -m pytest -q -p no:cacheprovider tests)
"$PY" -m pytest -q -p no:cacheprovider "$REPO/tests"

"$PY" -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
echo "setup done: $REPO/.venv"
