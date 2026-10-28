"""Does a model still help with harmless requests? Refusal and degeneracy rates, no attack.

usage:
  .venv/bin/python scripts/benign.py --model DIR --tag benign/additive_seed42 [--n 100]

A defence that refuses everything also scores ASR 0, so every low ASR needs this.
Prompts: the first --n of the harmless test split (not used by DDO training or RFA).
Writes rates to results/<tag>.json and the completions to models/results/<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from collect_artifacts import REPO


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--n", type=int, default=100)
    args = p.parse_args()

    out, raw = REPO / "results" / f"{args.tag}.json", REPO / "models" / "results" / f"{args.tag}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; move it away to rerun")
    os.environ.setdefault("HF_HOME", str(REPO / "models" / "hf_cache"))

    from ddo_defense.coherence import find_loop
    from ddo_defense.data import load_dataset_split
    from ddo_defense.judges import is_refusal
    from ddo_defense.models import ModelAdapter

    prompts = load_dataset_split("harmless", "test", instructions_only=True)[:args.n]
    adapter = ModelAdapter.from_pretrained(args.model)
    responses = adapter.generate(prompts, max_new_tokens=128, batch_size=16)
    adapter.free()

    refused = [is_refusal(r) for r in responses]
    degenerate = [find_loop(r) is not None for r in responses]
    rates = {"refusal_rate": sum(refused) / len(responses),
             "degenerate_rate": sum(degenerate) / len(responses),
             "mean_words": sum(len(r.split()) for r in responses) / len(responses)}
    print(rates)

    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps([{"prompt": p_, "response": r} for p_, r in zip(prompts, responses)], indent=2) + "\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model, "n_prompts": len(prompts), "split": "harmless/test", "max_new_tokens": 128,
        **rates,
    }, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
