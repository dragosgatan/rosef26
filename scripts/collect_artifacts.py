"""Collect the commit-sized record of one DDO build into artifacts/.

Writes, next to build.log:
  ddo_defense.json   the record apply saved with the model
  config.json        the resolved configuration and the exact command
  edits.safetensors  every edited gate row, up row and down column, with indices
  manifest.json      versions, GPU, seed, timestamp and sha256 of every shard

The edits are found by diffing the defended checkpoint against the original one
tensor at a time (low RAM), and must match ddo_defense.json exactly: any other
changed tensor, neuron or layer is an error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parent.parent
PATCH = REPO / "patches" / "ddo-24gb-vram.patch"
PACKAGES = ("torch", "transformers", "tokenizers", "accelerate", "nnsight",
            "safetensors", "huggingface_hub", "numpy", "ddo-defense")
MLP_WEIGHT = re.compile(r"model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight")


def tensor_files(model_dir) -> dict:
    """``{tensor name: file}`` for a safetensors checkpoint, sharded or not."""
    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return {name: model_dir / shard for name, shard in weight_map.items()}
    single = model_dir / "model.safetensors"
    with safe_open(single, "pt") as f:
        return {name: single for name in f.keys()}


def differing_tensors(a_dir, b_dir):
    """Yield ``(name, a, b)`` for every tensor that differs, one pair in RAM at a time."""
    a_files, b_files = tensor_files(a_dir), tensor_files(b_dir)
    if a_files.keys() != b_files.keys():
        raise SystemExit(f"tensor names differ: {sorted(a_files.keys() ^ b_files.keys())[:5]}")
    for name in sorted(a_files):
        with safe_open(a_files[name], "pt") as fa, safe_open(b_files[name], "pt") as fb:
            a, b = fa.get_tensor(name), fb.get_tensor(name)
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            yield name, a, b


def extract_edits(original, defended, neuron_indices: dict) -> dict:
    """Tensors for edits.safetensors, checked against ddo_defense.json.

    ``neuron_indices`` is ddo_defense.json's ``{"layer": [neuron, ...]}``.
    Keys written: ``{layer}.neurons`` and ``{layer}.{gate_proj|up_proj|down_proj}``
    (gate/up rows as [k, d_model], down columns as [d_model, k]).
    """
    found: dict = {}
    edits: dict = {}
    for name, a, b in differing_tensors(original, defended):
        m = MLP_WEIGHT.fullmatch(name)
        if not m or a.dtype != b.dtype or a.shape != b.shape:
            raise SystemExit(f"unexpected change in {name}: DDO only edits MLP rows/columns")
        layer, kind = m.group(1), m.group(2)
        # gate/up neurons are rows of [intermediate, d_model]; down neurons are columns.
        axis = 0 if kind == "down_proj" else 1
        idx = (a != b).any(dim=axis).nonzero().flatten()
        found.setdefault(layer, {})[kind] = idx.tolist()
        values = b[:, idx] if kind == "down_proj" else b[idx]
        edits[f"{layer}.{kind}"] = values.contiguous()
        edits[f"{layer}.neurons"] = idx

    expected = {str(l): sorted(ix) for l, ix in neuron_indices.items()}
    if found.keys() != expected.keys():
        raise SystemExit(f"edited layers {sorted(found, key=int)} != ddo_defense.json {sorted(expected, key=int)}")
    for layer, ix in expected.items():
        if any(found[layer].get(k) != ix for k in ("gate_proj", "up_proj", "down_proj")):
            raise SystemExit(f"layer {layer}: diff found {found[layer]}, ddo_defense.json says {ix}")
    return edits


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu() -> str | None:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader", "-i", "0"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--original", required=True, help="pinned snapshot of the original model")
    p.add_argument("--defended", required=True, help="folder apply saved to")
    p.add_argument("--artifacts", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--model-revision", required=True)
    p.add_argument("--command", required=True, help="the exact apply command that was run")
    args = p.parse_args()

    defended, art = Path(args.defended), Path(args.artifacts)
    art.mkdir(parents=True, exist_ok=True)
    info = json.loads((defended / "ddo_defense.json").read_text())
    shutil.copy(defended / "ddo_defense.json", art / "ddo_defense.json")

    (art / "config.json").write_text(json.dumps({
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "seed": args.seed,
        "params": info["params"],
        "command": args.command,
    }, indent=2) + "\n")

    print("diffing defended weights against the original ...")
    edits = extract_edits(args.original, defended, info["neuron_indices"])
    save_file(edits, art / "edits.safetensors", metadata={
        "model_id": args.model_id, "model_revision": args.model_revision, "seed": str(args.seed),
    })
    print(f"  edits match ddo_defense.json: layers {info['target_layers']}, "
          f"neurons {info['neuron_indices']}")

    print("hashing shards ...")
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "ddo_submodule_commit": subprocess.run(
            ["git", "-C", str(REPO / "third_party" / "ddo"), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip(),
        "patch_sha256": sha256(PATCH),
        "python": platform.python_version(),
        "torch_cuda": torch.version.cuda,
        "packages": {pkg: version(pkg) for pkg in PACKAGES},
        "gpu": gpu(),
        "env": {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF")},
        "command": args.command,
        "shards_sha256": {f.name: sha256(f) for f in sorted(defended.glob("*.safetensors"))},
        "edits_sha256": sha256(art / "edits.safetensors"),
    }
    (art / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  wrote {art}")


if __name__ == "__main__":
    main()
