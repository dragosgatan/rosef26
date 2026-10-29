# Findings

DDO (Muhamed, Diab & Smith, 2026), built with the authors' code and published Llama-3-8B-Instruct config.

**Attack used throughout:** 3-point RFA (rank 1, 128 unfiltered probes) on 100 JailbreakBench prompts. ASR is judged by LlamaGuard-2.

**Noise:** with 100 prompts, each ASR is about ±10 points. Raw records are in `results/` and the full log is in `RUNLOG.md`.

## 1. The authors' own detector undoes DDO completely

Seed 42:

| | decoys zeroed | ASR |
|---|---|---|
| undefended | – | 0.82 |
| defended | – | 0.39 |
| oracle (zero the 9 known decoys) | 9/9 | 0.80 |
| gate_up_cos, m=1, layers 6-14 | **9/9** | **0.80** |
| gate_up_cos, m=1, all 32 layers | 9/9 | 0.83 |
| down_norm, m=1, layers 6-14 | 0/9 | 0.45 |
| norm_ratio (ours), m=1, layers 6-14 | 0/9 | 0.45 |

- **Why it works:** every decoy has gate row = 1.30 × up row, so cos(gate, up) = 1. No natural neuron does; the natural max is 0.73-0.79 per layer.
- **No prior knowledge needed:** one neuron per layer, from the weights alone, without knowing which layers DDO used.
- **Only the cosine fingerprint works:** the down-norm and norm-ratio fingerprints don't single out the decoys.
- **Paper comparison:** the paper's Table 21 reports 0.17 for gate_up_cos at m=1. We can't reproduce their exact setup; see `RUNLOG.md`.

## 2. DDO's protection depends heavily on the random seed

Defended ASR over five builds. Each build has the same config, the same 9 neurons and the same gate/up rows; only the trained decoy vector differs.

| seed | 42 | 43 | 44 | 45 | 46 |
|---|---|---|---|---|---|
| defended ASR | 0.39 | 0.83 | 0.80 | 0.18 | 0.24 |
| final refusal log-odds (training) | 8.16 | 2.75 | 2.51 | 11.32 | 5.53 |

- **Two of five builds give no protection at all**, yet all five pass the authors' coherence gate.
- **The final refusal log-odds printed during training predict the outcome:** a low value means no protection.
- **The confusion-pool size predicts it even better** (Spearman −0.9 vs −0.8 for log-odds). Before training, DDO keeps only the harmful prompts that the model with untrained decoys still refuses. In replace mode only 13-38 survive, depending on the seed, and the two failed builds kept the fewest (13, 14). In additive mode fewer than 8 survive, so the code falls back to filtering on the clean model and keeps 126 every time (`build.log`: "too few prompts survived").

  | seed | 44 | 43 | 42 | 46 | 45 |
  |---|---|---|---|---|---|
  | harmful prompts in pool | 13 | 14 | 17 | 31 | 38 |
  | defended ASR | 0.80 | 0.83 | 0.39 | 0.24 | 0.18 |

## 3. Camouflage

**Additive mode** (the authors' `--compile_mode additive`) adds the decoy onto the host neuron instead of overwriting it.

- **Training:** healthy, final refusal log-odds 8.49.
- **Fingerprint:** each decoy's cosine drops to 0.30-0.67. Its rank in its layer goes from #1 to between #2 and #117 of about 14,300, and no detector ranks any decoy first.
- **Protection:** defended ASR **0.04**, against 0.39 for the same seed in replace mode.
- **Attacks:**

  | | decoys zeroed | ASR |
  |---|---|---|
  | gate_up_cos, m=1, layers 6-14 | 0/9 | 0.03 |
  | gate_up_cos, m=1, all 32 layers | 0/9 | 0.08 |
  | down_norm / norm_ratio, m=1 | 0/9 | 0.04 / 0.05 |
  | oracle (zero the 9 hosts) | 9/9 | 0.80 |

- **Still helpful:** refuses 2% of 100 harmless prompts (original 0%, replace mode 3%), with no degenerate output.
- **Who proposed it:** additive mode is the authors' own option; their code offers it as a fallback when replace mode fails the coherence gate. What we add is that it removes the fingerprint and, here, protects much better.
- **Likely reason:** its training kept 126 harmful prompts for the confusion loss (the filter's clean-model fallback), against 13-38 in replace mode; see section 2.
- **Across seeds** (same seeds as section 2):

  | seed | 42 | 43 | 44 |
  |---|---|---|---|
  | replace mode, defended ASR | 0.39 | 0.83 | 0.80 |
  | additive, defended ASR | **0.04** | **0.00** | **0.03** |
  | additive, gate_up_cos m=1 | 0/9 → 0.03 | 0/9 → 0.00 | 0/9 → 0.01 |
  | additive, harmless refusals | 2% | 8% | 2% |
  | additive, final refusal log-odds | 8.49 | 9.61 | 8.78 |

  Additive training ends with high log-odds every time, while replace mode does so only sometimes.

**Noise / split** (`scripts/camouflage.py`): applied after the fact to the seed-42 replace build, with no retraining. Noise is added only in directions the model's MLP input almost never uses. Both variants leave the model unchanged (KL about 1e-3 per token; harmless refusals 3%).

| | cos (decoy rank in layer) | defended | gate_up_cos m=1 | gate_up_cos m=10 | oracle |
|---|---|---|---|---|---|
| replace (no camouflage) | 1.00 (#1) | 0.39 | 9/9 → 0.80 | – | 0.80 |
| noise | 0.72 (#1-#13) | 0.44 | 1/9 → 0.29 | 8/9 → 0.49 | 0.80 |
| noise + split over 2 neurons | 0.36 (#64-#232) | 0.44 | 0/18 → 0.39 | **0/18 → 0.31** | 0.83 |

- **Noise alone:** beats the cheap m=1 attacker, but an attacker zeroing 10 neurons per layer finds 8 of 9 decoys. This matches the maths: inside natural weight sizes, one neuron's cosine can't go below about 0.7.
- **Split:** hides the decoys even from m=10.
- **Protection:** neither variant improves it beyond what replace mode gives (0.39-0.44).

## 4. Against OBLITERATUS, a popular abliteration tool

OBLITERATUS (commit `43b5d6d`, bf16) makes a permanent weight edit. We then judge the result with LlamaGuard-2 on the same 100 prompts, with no extra RFA.

| method | original | replace DDO (seed 42) | additive DDO (seed 42) |
|---|---|---|---|
| `advanced` (default: 4 directions) | 0.15 | 0.03 | 0.02 |
| `aggressive` (8 directions, 3 passes) | **0.73** | **0.55** | **0.10** |

- **`advanced` is too weak to judge a defence:** it reaches only 0.15 even on the undefended model.
- **Under `aggressive`, replace-mode DDO mostly falls** (0.55), while additive holds (0.10).
- **No model was damaged:** harmless refusals stayed at 0-2% and no output was degenerate.

**Limit:** the gate row still reads the refusal direction. So a detector that first estimates that direction, or looks at activations instead of weights, can still find the decoys. Camouflage raises the attacker's cost; it does not make the decoys invisible.

## 5. What the paper says (arXiv:2609.16204)

- **Additive mode on Llama-3:** they tested it and found it *worse* than replace: 16.5% vs 1.8% ASR in the ablation (Table 5, with the same hyperparameters as replace, the way we ran it) and 7% vs 0% with its own tuning (Table 10). Both numbers average 4 RFA variants and 3 judges. They never ran the undo detectors on additive. We find the opposite: additive 0.00-0.04 vs replace 0.18-0.83, and the detectors find nothing.
- **Table 21:** their setup matches ours: Llama-3 DDO, LlamaGuard-2, 3-point RFA, JailbreakBench, top-m `down_proj` columns zeroed per DDO layer, base 80% (ours 0.82). They report gate/up cosine at 12-17% for every m; we get 0.80 at m=1. Since cos = 1 holds for every replace build whatever the seed, the gap is still unexplained.
- **The cos = 1 fingerprint:** Appendix F.16 calls it "an artifact of a particular compile choice". It suggests adding "a small orthogonal perturbation" to the up row, but says that every model in the paper was evaluated without it. Our noise variant is that test: it beats the m=1 detector, but the m=10 detector still finds 8/9 decoys, because within natural weight sizes the cosine can't drop below about 0.7.
- **Norm ratio:** F.16 says the gate/up norm ratio is β ≈ 3-6, which would make the decoys stand out. Llama-3's β is 1.30, which is why norm_ratio found 0/9.
- **Seeds:** the paper reports seed effects only for probe sets (±1%) and Heretic's Optuna search (±3%). It says nothing about the DDO training seed, which in our runs moved defended ASR by up to 65 points.
