"""Rebuild a DDO-defended checkpoint: the original model plus edits.safetensors.

usage:
  .venv/bin/python scripts/reapply_edits.py \
      --edits artifacts/ddo_llama3_seed42/edits.safetensors \
      --out models/ddo_llama3_seed42_reapplied [--original DIR] [--check DIR]

GPUs are not fully deterministic, so rerunning the optimization would not give
the same decoys. This gives back the exact defended model instead, on CPU, one
shard at a time (about 5 GB of RAM). --original defaults to the pinned snapshot
recorded in the edits file (downloaded into models/hf_cache if missing).
--check compares every tensor of the result with another checkpoint, e.g. the
one apply saved, and fails unless all are bit-identical.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file

from collect_artifacts import REPO, differing_tensors


def original_snapshot(edits_path) -> str:
    """The pinned original model recorded in an edits file (downloaded if missing)."""
    os.environ.setdefault("HF_HOME", str(REPO / "models" / "hf_cache"))
    from huggingface_hub import snapshot_download

    with safe_open(edits_path, "pt") as f:
        meta = f.metadata()
    return snapshot_download(meta["model_id"], revision=meta["model_revision"],
                             allow_patterns=["*.json", "model-*.safetensors"])


def reapply(edits_path, original, out) -> None:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)

    edits = load_file(edits_path)
    pending = {}  # weight name -> (kind, neuron indices, values)
    for key, values in edits.items():
        layer, kind = key.split(".", 1)
        if kind != "neurons":
            pending[f"model.layers.{layer}.mlp.{kind}.weight"] = (kind, edits[f"{layer}.neurons"], values)

    for src in sorted(Path(original).iterdir()):
        if src.suffix == ".safetensors":
            tensors = load_file(src)
            for name in tensors.keys() & pending.keys():
                kind, idx, values = pending.pop(name)
                if kind == "down_proj":
                    tensors[name][:, idx] = values
                else:
                    tensors[name][idx] = values
            save_file(tensors, out / src.name, metadata={"format": "pt"})
        elif src.is_file():  # config, index, tokenizer
            shutil.copy(src, out / src.name)
    if pending:
        raise SystemExit(f"not found in {original}: {sorted(pending)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--edits", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--original", help="original model folder (default: pinned snapshot)")
    p.add_argument("--check", help="checkpoint the result must equal bit for bit")
    args = p.parse_args()

    reapply(args.edits, args.original or original_snapshot(args.edits), args.out)
    print(f"wrote {args.out}")

    if args.check:
        diffs = [name for name, _, _ in differing_tensors(args.out, args.check)]
        if diffs:
            raise SystemExit(f"differs from {args.check} in {len(diffs)} tensors: {diffs[:5]}")
        print(f"bit-identical to {args.check}")


if __name__ == "__main__":
    main()
