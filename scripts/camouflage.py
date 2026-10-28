"""Hide DDO's cos(gate, up) = 1 fingerprint after the fact, keeping each decoy's function.

usage (GPU):
  .venv/bin/python scripts/camouflage.py --name noise_seed42 [--base ddo_llama3_seed42] \
      [--gate-norm 1.3] [--up-norm 0.8] [--down-norm 0.8] [--split 1]

A DDO decoy is gate = beta*r, up = r, down = s*u (beta 1.30, s 0.463, |r| = |u| = 1).
Its output silu(gate.x) * (up.x) * down depends on r.x only, so we may:
  * pad gate and up with noise n that the MLP input almost never contains
    (the lowest-variance directions of that input, measured on prompts), and
  * move the scale between up and down: up's r part a, down's norm D, a*D = s.
Then cos(gate, up) = beta*a / (|gate|*|up|). With --split k the decoy is shared by
k neurons (the base one plus the next-smallest-down-norm neurons), each with
a*D = s/k, which divides the cosine by k again.

Writes models/<name> (gitignored) and artifacts/<name>/: edits.safetensors (same
format as a build's, so reapply/inspect/run_attacks work) and camouflage.json
(settings, per-layer fingerprints and ranks, noise leak, KL to the base build).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from collect_artifacts import REPO
from inspect_decoys import layer_scores



def mlp_input_moments(adapter, layers, prompts, batch=16) -> dict:
    """``{layer: E[x x^T]}`` (float32, uncentred) of each MLP input over all prompt tokens."""
    moments = {l: torch.zeros(adapter.d_model, adapter.d_model, device="cuda") for l in layers}
    count, mask = [0], [None]

    def hook(l):
        def fn(_module, _inp, out):
            x = out[mask[0]].float()  # real tokens only, [n, d]
            moments[l] += x.T @ x
        return fn

    handles = [adapter.model.model.layers[l].post_attention_layernorm.register_forward_hook(hook(l))
               for l in layers]
    try:
        for i in range(0, len(prompts), batch):
            enc = adapter.tokenize(prompts[i:i + batch]).to("cuda")
            mask[0] = enc.attention_mask.bool()
            count[0] += int(mask[0].sum())
            with torch.no_grad():
                adapter.model(**enc)
    finally:
        for h in handles:
            h.remove()
    return {l: m / count[0] for l, m in moments.items()}


def logprobs(adapter, prompts, batch=8) -> list:
    out = []
    for i in range(0, len(prompts), batch):
        enc = adapter.tokenize(prompts[i:i + batch]).to("cuda")
        with torch.no_grad():
            lp = adapter.model(**enc).logits.float().log_softmax(-1)
        out.append((lp[enc.attention_mask.bool()].cpu().to(torch.bfloat16)))
    return out


def unit_in(basis, gen, avoid=()):
    """Random unit vector in span(basis), orthogonal to the vectors in ``avoid``."""
    v = basis @ torch.randn(basis.shape[1], generator=gen, device=basis.device)
    for a in avoid:
        v -= (v @ a) * a
    return v / v.norm()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", required=True)
    p.add_argument("--base", default="ddo_llama3_seed42", help="replace-mode build to camouflage")
    p.add_argument("--gate-norm", type=float, default=1.3)
    p.add_argument("--up-norm", type=float, default=0.8)
    p.add_argument("--down-norm", type=float, default=0.8)
    p.add_argument("--split", type=int, default=1, help="neurons sharing each decoy")
    p.add_argument("--n-null", type=int, default=1024, help="size of the low-variance subspace for the noise")
    args = p.parse_args()

    base_edits = REPO / "artifacts" / args.base / "edits.safetensors"
    out_model, art = REPO / "models" / args.name, REPO / "artifacts" / args.name
    if art.exists() or out_model.exists():
        raise SystemExit(f"{art} or {out_model} exists; pick another --name")

    from ddo_defense.data import load_dataset_split
    from ddo_defense.models import ModelAdapter

    edits = load_file(base_edits)
    with safe_open(base_edits, "pt") as f:
        meta = f.metadata()
    layers = sorted(int(k.split(".")[0]) for k in edits if k.endswith(".neurons"))
    # Not the RFA probes (train split) nor the eval set (JailbreakBench). harmful/val has only 39.
    harmful = (load_dataset_split("harmful", "val", instructions_only=True)
               + load_dataset_split("harmful", "test", instructions_only=True))
    prompts = load_dataset_split("harmless", "val", instructions_only=True)[:128] + harmful[:128]
    kl_prompts = load_dataset_split("harmless", "val", instructions_only=True)[128:160]

    adapter = ModelAdapter.from_pretrained(str(REPO / "models" / args.base))
    print("measuring MLP inputs ...")
    moments = mlp_input_moments(adapter, layers, prompts)
    ref = logprobs(adapter, kl_prompts)

    gen = torch.Generator(device="cuda").manual_seed(0)
    new_edits, report = {}, {}
    for l in layers:
        mlp = adapter.model.model.layers[l].mlp
        gate_w, up_w, down_w = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
        base_n = int(edits[f"{l}.neurons"][0])
        g0 = edits[f"{l}.gate_proj"][0].float().cuda()
        u0 = edits[f"{l}.up_proj"][0].float().cuda()
        d0 = edits[f"{l}.down_proj"][:, 0].float().cuda()
        r = u0 / u0.norm()
        beta, s = (g0 @ r).item(), (u0.norm() * d0.norm()).item()
        a = s / (args.split * args.down_norm)
        if args.up_norm < a or args.gate_norm < beta - 0.01:  # beta is 1.30 up to bf16 rounding
            raise SystemExit(f"layer {l}: need --up-norm >= {a:.3f} and --gate-norm >= {beta:.3f}")

        evals, evecs = torch.linalg.eigh(moments[l])  # ascending
        basis = evecs[:, :args.n_null]
        basis = torch.linalg.qr(basis - torch.outer(r, r @ basis))[0]
        leak = (evals[:args.n_null].mean() / (r @ moments[l] @ r)).sqrt().item()

        # The base neuron plus the next-smallest down columns (DDO's own low_norm rule).
        norms = down_w.float().norm(dim=0)
        norms[base_n] = -1
        neurons = norms.argsort()[:args.split].tolist()

        rows = {"gate_proj": [], "up_proj": [], "down_proj": []}
        for n in neurons:
            ng = unit_in(basis, gen)
            nu = unit_in(basis, gen, avoid=[ng])
            gate = beta * r + max(args.gate_norm ** 2 - beta ** 2, 0) ** 0.5 * ng
            up = a * r + (args.up_norm ** 2 - a ** 2) ** 0.5 * nu
            down = d0 / d0.norm() * args.down_norm
            with torch.no_grad():
                gate_w[n], up_w[n], down_w[:, n] = gate, up, down
            rows["gate_proj"].append(gate_w[n].clone())
            rows["up_proj"].append(up_w[n].clone())
            rows["down_proj"].append(down_w[:, n].clone())

        order = sorted(range(len(neurons)), key=lambda i: neurons[i])
        new_edits[f"{l}.neurons"] = torch.tensor([neurons[i] for i in order])
        new_edits[f"{l}.gate_proj"] = torch.stack([rows["gate_proj"][i] for i in order]).cpu()
        new_edits[f"{l}.up_proj"] = torch.stack([rows["up_proj"][i] for i in order]).cpu()
        new_edits[f"{l}.down_proj"] = torch.stack([rows["down_proj"][i] for i in order], 1).cpu()

        scores = layer_scores(gate_w, up_w, down_w)
        # Activation-weighted cosine: what the rows look like to real inputs. The noise
        # lives where inputs are ~0, so this should expose the decoy again (the honest limit).
        gm, um = gate_w.float() @ moments[l], up_w.float() @ moments[l]
        scores["act_cos"] = ((gm * up_w.float()).sum(1) /
                             ((gm * gate_w.float()).sum(1) * (um * up_w.float()).sum(1)).sqrt()).abs()
        del gm, um
        natural = torch.ones(len(norms), dtype=torch.bool, device=norms.device)
        natural[neurons] = False
        report[l] = {"neurons": sorted(neurons), "noise_leak_rms_vs_r": leak, "decoys": [
            {"neuron": n, **{k: {"value": v[n].item(), "rank": 1 + int((v[natural] >= v[n]).sum())}
                             for k, v in scores.items()}} for n in sorted(neurons)]}
        c = report[l]["decoys"][0]["gate_up_cos"]
        print(f"  layer {l:2d}: neurons {sorted(neurons)}, cos {c['value']:.3f} (rank {c['rank']}), "
              f"noise leak {leak:.4f}")

    new = logprobs(adapter, kl_prompts)
    kl = torch.cat([F.kl_div(n.float(), r_.float(), log_target=True, reduction="none").sum(-1)
                    for r_, n in zip(ref, new)])
    print(f"KL(base || camouflaged) per token: mean {kl.mean():.2e}, max {kl.max():.2e}")

    adapter.model.save_pretrained(out_model)
    adapter.tokenizer.save_pretrained(out_model)
    art.mkdir(parents=True)
    save_file(new_edits, art / "edits.safetensors", metadata=meta)
    (art / "camouflage.json").write_text(json.dumps({
        "base": args.base, "gate_norm": args.gate_norm, "up_norm": args.up_norm,
        "down_norm": args.down_norm, "split": args.split, "n_null": args.n_null,
        "kl_mean": kl.mean().item(), "kl_max": kl.max().item(), "layers": report,
    }, indent=2) + "\n")
    print(f"wrote {out_model} and {art}")


if __name__ == "__main__":
    main()
