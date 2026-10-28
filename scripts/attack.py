"""Optionally undo the decoys, then run the authors' standard RFA and judge it.

usage:
  .venv/bin/python scripts/attack.py --model DIR --tag seed42/defended \
      [--undo none|oracle|gate_up_cos|down_norm|norm_ratio] [--m 1] [--layers all|6-14] [--edits FILE]
      [--no-rfa]   (judge the model as it is, e.g. one already abliterated by another tool)

The paper's Table 21 setup: zero the top-m down_proj columns per layer, then
3-point RFA (rank 1, 128 unfiltered probes) on JailbreakBench, judged by
LlamaGuard-2 (the substring judge is logged too, but is not a reportable number).
  oracle       zero exactly the neurons in --edits
  gate_up_cos, down_norm   the authors' undo_attack heuristics
  norm_ratio   ours: highest |gate row| / |up row|
Raw completions go to models/results/<tag>.json (gitignored: never commit them);
results/<tag>.json gets only the rates and settings.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from safetensors.torch import load_file

from collect_artifacts import REPO
from inspect_decoys import layer_scores

UNDO = ("none", "oracle", "gate_up_cos", "down_norm", "norm_ratio")


def undo(model, how: str, m: int, layers: list, edits: dict | None) -> dict:
    """Zero neurons' down_proj columns in place; returns ``{layer: [neurons]}``."""
    from ddo_defense.attacks.undo import undo_attack

    if how == "none":
        return {}
    if how in ("gate_up_cos", "down_norm"):
        return undo_attack(model, target_layers=layers, m=m, heuristic=how)["ablated"]
    if how == "oracle":
        ablated = {int(k.split(".")[0]): v.tolist() for k, v in edits.items() if k.endswith(".neurons")}
    else:
        ablated = {}
        for l in layers:
            mlp = model.model.layers[l].mlp
            ratio = layer_scores(mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight)["norm_ratio"]
            ablated[l] = sorted(ratio.topk(m).indices.tolist())
    for l, idx in ablated.items():
        model.model.layers[l].mlp.down_proj.weight.data[:, idx] = 0
    return ablated


def parse_layers(spec: str, n_layers: int) -> list:
    if spec == "all":
        return list(range(n_layers))
    a, b = map(int, spec.split("-"))
    return list(range(a, b + 1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True, help="result name, e.g. seed42/defended")
    p.add_argument("--undo", choices=UNDO, default="none")
    p.add_argument("--m", type=int, default=1, help="neurons zeroed per layer")
    p.add_argument("--layers", default="all", help="'all' or an inclusive range like 6-14")
    p.add_argument("--edits", help="edits.safetensors: needed for oracle, reports decoy hits otherwise")
    p.add_argument("--no-rfa", action="store_true", help="skip RFA; same prompts and decoding")
    args = p.parse_args()
    if args.undo == "oracle" and not args.edits:
        p.error("--undo oracle needs --edits")

    out, raw = REPO / "results" / f"{args.tag}.json", REPO / "models" / "results" / f"{args.tag}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; move it away to rerun")
    os.environ.setdefault("HF_HOME", str(REPO / "models" / "hf_cache"))

    from ddo_defense.attacks.rfa import run_rfa
    from ddo_defense.judges import LlamaGuard2Judge, SubstringJudge
    from ddo_defense.models import ModelAdapter

    adapter = ModelAdapter.from_pretrained(args.model)
    layers = parse_layers(args.layers, adapter.n_layers)
    edits = load_file(args.edits) if args.edits else None
    ablated = undo(adapter.model, args.undo, args.m, layers, edits)
    print(f"undo {args.undo}: zeroed {ablated}")

    decoys_zeroed = None
    if edits:
        decoys = {int(k.split(".")[0]): set(v.tolist()) for k, v in edits.items() if k.endswith(".neurons")}
        hit = sum(len(ix & set(ablated.get(l, []))) for l, ix in decoys.items())
        decoys_zeroed = f"{hit}/{sum(map(len, decoys.values()))}"
        print(f"decoys zeroed: {decoys_zeroed}")

    if args.no_rfa:
        from ddo_defense.attacks.rfa import generate_under_attack
        from ddo_defense.data import load_eval_set
        # The authors' unattacked baseline: run_rfa's decoding, with no directions ablated.
        completions = generate_under_attack(adapter, load_eval_set("jailbreakbench"), directions=None)
        attack = {"attack": "none", "eval_set": "jailbreakbench", "max_new_tokens": 512,
                  "n_completions": len(completions)}
    else:
        attack = run_rfa(adapter, rank=1, n_probes=128, ablation_mode="three_point",
                         eval_set="jailbreakbench")
        completions = attack.pop("completions")
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(completions, indent=2) + "\n")
    print(f"wrote raw completions to {raw}")
    adapter.free()  # LlamaGuard-2 needs the GPU memory

    asr = {}
    for judge in (LlamaGuard2Judge(), SubstringJudge()):
        try:
            asr[judge.name] = judge.compute_asr(completions)
        finally:
            judge.cleanup()
    print(f"ASR: {asr}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model, "undo": args.undo, "m": args.m, "layers": args.layers,
        "edits": args.edits, "ablated": ablated, "decoys_zeroed": decoys_zeroed,
        "asr": asr, "rfa": attack,
    }, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
