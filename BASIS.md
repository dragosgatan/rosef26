# Hiding in Plain Sight: Detecting and Camouflaging Decoy Defenses Against LLM Abliteration

RoSEF 2026, category III: Tehnologia informației. The competition runs 30-31 Oct 2026.

## The problem

Open-weight chat models are trained to refuse harmful requests. **Abliteration** (Refusal Feature Ablation, RFA) removes that safeguard in minutes, with no training:

1. Run the model on harmful and harmless prompts.
2. Subtract the two mean activations. The result is the *refusal direction* r̂.
3. Project r̂ out of every weight matrix that writes to the residual stream. The model then stops refusing.

## The defense we study

**Decoy Direction Optimization (DDO)**, from Muhamed, Diab & Smith, arXiv:2609.16204, 2026, is a cheap defense that edits weights after training:

- In each target layer, DDO overwrites one low-impact MLP neuron so that it fires strongly (≈ βa²) on harmful prompts and writes a *decoy* direction orthogonal to r̂.
- The attacker's difference-of-means estimate is pulled toward the decoy, so abliteration deletes the decoy and leaves the real refusal mechanism intact.
- Against standard RFA, attack success rate (ASR) falls from about 85% to under 10%.

## The gap

1. Default decoys are structurally abnormal. In the default `replace` mode, `w_gate = β·r̂` and `w_up = r̂`. The two rows are exactly parallel, with a norm ratio of β ≈ 3-6. The authors call this "an outlier among ~12K neurons" (Appendix F.16).
2. The camouflage they propose was never tested. The paper suggests two fixes: redistributing the gain (both rows set to `√β·r̂`) and adding a small perturbation `εv` to `w_up`. The authors state: *"All models evaluated in this paper use the default compile without norm redistribution or cosine camouflage."*
3. Their undo attacks work only in part. Zeroing neurons flagged by gate/up cosine and then re-running RFA brings ASR back only to 12-17%, against an undefended 80%. They never ran the norm-ratio heuristic as an attack, nor combined undo with stronger attacks.

## Research questions

- **RQ1:** Can default DDO decoys be found from the weights alone (norm ratio, gate/up cosine, alignment with r̂)? Does removing them restore abliteration?
- **RQ2:** Do the authors' proposed camouflages hide the decoys from these detectors?
- **RQ3:** Do camouflaged decoys still protect the model? What is the trade-off between stealth and protection?
- **RQ4 (stretch):** Can camouflaged decoys still be caught by their *behaviour*, i.e. abnormal activations on harmful prompts?

## Hypotheses

- **H1:** Under the norm-ratio score, the default decoy ranks #1 in every DDO layer.
- **H2:** Norm redistribution plus a cosine perturbation drops the decoy into the natural distribution.
- **H3:** Some protection is lost as ε grows, so stealth trades off against protection.
- **H4:** An activation-based detector still flags camouflaged decoys.

## Method

| Step | What we do |
|---|---|
| Model | Qwen3-1.7B-Instruct (same family as the paper's Qwen3-8B); Llama-3.2-1B/3B if time allows |
| Baseline | Standard RFA on the undefended model |
| Defense | DDO via the authors' code (`ddo-apply`, replace mode first) |
| Ground truth | Diff of the defended weights against the original, which gives the exact edited neurons |
| Detectors | Gate/up norm ratio, gate/up cosine, gate·d̂ projection, and an activation outlier score |
| Undo attack | Zero the flagged neurons, then RFA |
| Camouflage | `√β` redistribution, sweep of `ε ∈ {0, 0.05, 0.1, 0.2, 0.4}`, `n_readers > 1` |

**Metrics:**
- ASR on JailbreakBench (100 behaviours), judged by Llama-Guard-3-1B plus refusal-string matching, with 50 responses hand-labelled to check the judge.
- Decoy rank and detector AUROC.
- Over-refusal on XSTest.
- Coherence, and MMLU on a subset.
- At least 2 seeds for each headline result, reported with bootstrap confidence intervals.

**Compute:** free Kaggle or Colab T4 GPUs.

## Expected main figure

Detectability (decoy rank or AUROC) plotted against protection (ASR under RFA), for each camouflage setting.

## Fallback

If DDO does not lower ASR on small models, that is itself a finding: *DDO does not transfer to 1-3B models*. RQ1 and RQ2 are answered on the weights regardless, since they do not need a working defense.

## Ethics

This is defensive red-teaming of a published safeguard:
- Raw harmful generations are never committed or shown. Only aggregate rates are reported.
- No abliterated or undone checkpoints are uploaded.
- Every finding is paired with what it means for building stronger defenses.

## References

- Muhamed, Diab, Smith (2026). *Decoy Direction Optimization: A Post-Hoc Defense Against LLM Abliteration.* arXiv:2609.16204. Code: github.com/aashiqmuhamed/defending-against-abliteration
- Arditi et al. (2024). *Refusal in Language Models Is Mediated by a Single Direction.*
