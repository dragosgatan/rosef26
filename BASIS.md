# Hiding in Plain Sight: Detecting and Camouflaging Decoy Defenses Against LLM Abliteration

RoSEF 2026, category III: Tehnologia informației. The competition runs 30-31 Oct 2026.

## The problem

Open-weight chat models are trained to refuse harmful requests. **Abliteration** (Refusal Feature Ablation, RFA) removes that safeguard easily:

1. Run the model on harmful and harmless prompts.
2. At each layer, subtract the two mean activations. The result is that layer's *refusal direction*.
3. While the model runs, remove each layer's direction from its activations. The model then stops refusing.

This is the DDO paper's "standard RFA" (one direction per layer, applied with hooks). We use their implementation (`ddo_defense.attacks.rfa`) so our numbers are comparable.

## The defense we study

**Decoy Direction Optimization (DDO)**, from Muhamed, Diab & Smith, arXiv:2609.16204, 2026, is a cheap defense that edits weights after training:

- In each target layer, DDO overwrites one low-impact MLP neuron so that it fires strongly (≈ βa²) on harmful prompts and writes a *decoy* direction orthogonal to r̂.
- The attacker's difference-of-means estimate is pulled toward the decoy, so abliteration deletes the decoy and leaves the real refusal mechanism intact.
- On Llama-3-8B-Instruct, standard RFA success drops from 85.1% to 1.8% (Table 2).

## The gap

1. **Default decoys have exact fingerprints.** In `replace` mode, `w_gate = β·r̂`, `w_up = r̂`, `w_down = s·u`. For Llama-3 (β = 1.30, s = 0.463) the gate and up rows are exactly parallel (cosine 1.0), with norms of exactly 1.30 and 1.0. The paper calls the norm ratio "β ≈ 3-6 ... an outlier among ~12K neurons" (Appendix F.16), but for Llama-3 the ratio is 1.30 and a layer has 14,336 neurons, so the ratio may not stand out. The cosine almost certainly does.
2. **The camouflage they propose was never tested.** The paper suggests two fixes: redistributing the gain (both rows set to `√β·r̂`) and adding a small perturbation `εv` to `w_up`. The authors state: *"All models evaluated in this paper use the default compile without norm redistribution or cosine camouflage."*
3. **Their undo result is unexplained.** Their gate/up cosine heuristic should zero exactly the decoy (cosine 1.0) in every DDO layer, yet Table 21 reports only 17% ASR afterwards, against 80% undefended. Either the decoy was not the one removed, or removing it does not restore the attack. They never tested the norm-ratio heuristic, and never combined undo with stronger attacks.

## Research questions

- **RQ0 (run first):** Zero exactly the decoys found by the weight diff, then run RFA. Does ASR return to about 80%? If not, hiding the decoy does not matter for protection, and the question becomes why.
- **RQ1:** Can default DDO decoys be found from the weights alone (gate/up cosine, norm ratio, `down` column norm)? Does removing them restore abliteration?
- **RQ2:** Do the authors' proposed camouflages hide the decoys from these detectors?
- **RQ3:** Do camouflaged decoys still protect the model? What is the trade-off between stealth and protection?
- **RQ4 (stretch):** Can camouflaged decoys still be caught by their behaviour, i.e. abnormal activations on harmful prompts?

## Hypotheses

- **H1:** The default decoy ranks #1 in every DDO layer under gate/up cosine, but not reliably under norm ratio (β = 1.30 is close to normal).
- **H2:** Norm redistribution plus a large enough cosine perturbation drops the decoy into the natural distribution.
- **H3:** Protection falls as ε grows, so stealth trades off against protection.
- **H4:** An activation-based detector still flags camouflaged decoys.

## Method

| Step | What we do |
|---|---|
| Model | Llama 3 8B Instruct, bf16 (no quantization) |
| Baseline | Standard RFA on the undefended model |
| Defense | DDO via the authors' code and published config (replace mode), with a memory patch so it fits a 24 GB GPU (see `patches/README.md`) |
| Ground truth | Diff of the defended weights against the original, which gives the exact edited neurons |
| Detectors | Gate/up cosine, gate/up norm ratio, `down` column norm, gate·d̂ projection, and an activation outlier score, each compared against all normal neurons |
| Undo attack | Zero the flagged neurons, then RFA |
| Camouflage | `√β` redistribution plus `w_up = r̂ + εv`, with ε large enough to reach the natural cosine range (below) |

**Choosing ε.** cos(gate, up) = 1/√(1+ε²):

| ε | 0.1 | 0.4 | 1 | 3 |
|---|---|---|---|---|
| cos | 0.995 | 0.928 | 0.71 | 0.32 |

The first plan (ε ≤ 0.4) would keep the cosine above 0.92, probably still an outlier. We first measure the cosine of normal neurons, then sweep ε until the decoy reaches that range.

## Where the code and the paper disagree

We run the authors' code as published, plus the memory patch.

| Topic | Paper | Code | We |
|---|---|---|---|
| Norm ratio | "β ≈ 3-6" (App. F.16) | β = 1.30, matching the paper's own Table 13 | use 1.30 |
| Probe prompts | validation split (Sec. 4.1) | train split | follow the code |
| Memory | overhead "beyond inference is minimal" (Sec. 5) | loads the model twice (~30 GiB) | patch: load once |
| Simulated attack in training | re-estimated on the defended model | only at layers 6-14, every 10 steps | follow the code |

## Expected main figure

Detectability (decoy rank or AUROC) plotted against protection (ASR under RFA), for each camouflage setting.

## Ethics

This is defensive red-teaming of a published safeguard:
- Raw harmful generations are never committed or shown. Only aggregate rates are reported.
- No abliterated or undone checkpoints are uploaded.
- Every finding is paired with what it means for building stronger defenses.

## References

- Muhamed, Diab, Smith (2026). *Decoy Direction Optimization: A Post-Hoc Defense Against LLM Abliteration.* arXiv:2609.16204. Code: github.com/aashiqmuhamed/defending-against-abliteration
- Arditi et al. (2024). *Refusal in Language Models Is Mediated by a Single Direction.*
