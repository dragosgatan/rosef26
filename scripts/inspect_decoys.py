"""Score every MLP neuron from the weights alone and rank the DDO decoys.

usage:
  .venv/bin/python scripts/inspect_decoys.py \
      --edits artifacts/ddo_llama3_seed42/edits.safetensors [--original DIR] [--out FILE]

The defended weights are the original model with the edits patched in, one
layer at a time on CPU (about 2 GB of RAM). Per neuron:
  gate_up_cos  |cos(gate row, up row)|  (the authors' gate_up_cos heuristic)
  norm_ratio   |gate row| / |up row|
  down_norm    |down column|            (the authors' down_norm heuristic)
Each decoy is ranked against the normal neurons only, ties counted against it:
in its own layer, and globally across every layer (an attacker who does not
know which layers DDO edited). Writes the numbers, no weights, to --out
(default: inspect.json next to the edits).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from collect_artifacts import MLP_WEIGHT, tensor_files
from reapply_edits import original_snapshot

SCORES = ("gate_up_cos", "norm_ratio", "down_norm")


def layer_scores(gate, up, down) -> dict:
    """``{score: [intermediate]}`` for one layer; higher means more decoy-like."""
    gate, up, down = gate.float(), up.float(), down.float()
    return {
        "gate_up_cos": F.cosine_similarity(gate, up, dim=1).abs(),
        "norm_ratio": gate.norm(dim=1) / up.norm(dim=1),
        "down_norm": down.norm(dim=0),
    }


def load_layer(files: dict, layer: int, edits: dict):
    """gate, up, down of one layer with the decoy rows/columns patched in."""
    w = {}
    for kind in ("gate_proj", "up_proj", "down_proj"):
        name = f"model.layers.{layer}.mlp.{kind}.weight"
        with safe_open(files[name], "pt") as f:
            w[kind] = f.get_tensor(name)
        if f"{layer}.neurons" in edits:
            idx = edits[f"{layer}.neurons"]
            if kind == "down_proj":
                w[kind][:, idx] = edits[f"{layer}.{kind}"]
            else:
                w[kind][idx] = edits[f"{layer}.{kind}"]
    return w["gate_proj"], w["up_proj"], w["down_proj"]


def inspect(original, edits_path) -> dict:
    files = tensor_files(original)
    edits = load_file(edits_path)
    layers = sorted({int(m.group(1)) for n in files if (m := MLP_WEIGHT.fullmatch(n))})

    report, natural = {}, {s: [] for s in SCORES}
    for layer in layers:
        scores = layer_scores(*load_layer(files, layer, edits))
        decoys = edits.get(f"{layer}.neurons", torch.tensor([], dtype=torch.long))
        is_normal = torch.ones(len(scores["gate_up_cos"]), dtype=torch.bool)
        is_normal[decoys] = False
        entry = {"natural": {}, "decoys": []}
        for s, v in scores.items():
            nat = v[is_normal]
            natural[s].append(nat)
            entry["natural"][s] = {"median": nat.median().item(),
                                   "p99": nat.quantile(0.99).item(),
                                   "max": nat.max().item()}
        for i in decoys.tolist():
            entry["decoys"].append({"neuron": i, **{
                s: {"value": v[i].item(), "rank": 1 + int((v[is_normal] >= v[i]).sum())}
                for s, v in scores.items()}})
        report[str(layer)] = entry
        print(f"  layer {layer:2d} scored" + (f", decoys {decoys.tolist()}" if len(decoys) else ""))

    natural = {s: torch.cat(v) for s, v in natural.items()}
    for entry in report.values():
        for d in entry["decoys"]:
            for s in SCORES:
                d[s]["global_rank"] = 1 + int((natural[s] >= d[s]["value"]).sum())
    return {"original": str(original), "edits": str(edits_path),
            "n_normal_neurons": len(natural["gate_up_cos"]), "layers": report}


def print_report(r: dict) -> None:
    for s in SCORES:
        print(f"\n{s}   (rank among normal neurons: in layer / global of {r['n_normal_neurons']})")
        print(f"  {'layer':>5} {'neuron':>7} {'decoy':>9} {'rank':>11} {'natural p99':>12} {'natural max':>12}")
        for layer, e in r["layers"].items():
            nat = e["natural"][s]
            decoys = e["decoys"] or [None]
            for d in decoys:
                neuron = d["neuron"] if d else ""
                value = f"{d[s]['value']:.4f}" if d else ""
                rank = f"{d[s]['rank']}/{d[s]['global_rank']}" if d else ""
                print(f"  {layer:>5} {neuron:>7} {value:>9} {rank:>11} {nat['p99']:12.4f} {nat['max']:12.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--edits", required=True)
    p.add_argument("--original", help="original model folder (default: pinned snapshot)")
    p.add_argument("--out", help="default: inspect.json next to --edits")
    args = p.parse_args()

    r = inspect(args.original or original_snapshot(args.edits), args.edits)
    print_report(r)
    out = Path(args.out or Path(args.edits).with_name("inspect.json"))
    out.write_text(json.dumps(r, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
