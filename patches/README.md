# Patch: running DDO on a 24 GB GPU

`ddo-24gb-vram.patch` is the only change we make to the authors' code
(`third_party/ddo`, commit `a26f23e`). The submodule itself is never edited:
the setup script installs from a patched copy.

## Why it is needed

The authors ran on an A100 GPU. Their code, unchanged, needs well over 24 GB
of GPU memory for Llama-3-8B, so it cannot run on the RTX 3090 we use:

1. **It loads the model twice.** Once to measure the refusal direction, and a
   second, separate copy for training (through the nnsight library). Each copy
   is about 15 GiB, so this alone is about 30 GiB.
2. **It processes 64 prompts at once while training.** Training needs to
   remember the intermediate results of every prompt in the batch, and with 64
   prompts that adds an estimated 15 GB or more.

## What the patch changes

1. **One copy of the model instead of two.** The training library now uses the
   copy that is already loaded. Both copies held identical weights and the
   training never changes them, so nothing about the result changes. This also
   matches the paper, which says the base weights stay frozen and memory
   overhead "beyond inference is minimal". (`ddo_defense/tuning.py`)
2. **Prompts are processed one at a time in two training steps** (the
   "confusion" loss and the "refusal-score" loss) instead of 16 at once. The
   loss is an average over the prompts, so each prompt's part is computed
   separately, divided by the number of prompts, and the parts are added up.
   That is exactly the same average and the same direction of improvement;
   only the last decimal places of the arithmetic can differ.
   (`ddo_defense/defense/optimizer.py`)

## What stays the same

- Every hyperparameter from the paper (Table 13) and the README: layers 6-14,
  β = 1.30, scale = 0.463, λ = 1.12, learning rate 0.0026, 2 epochs, replace
  mode, seed 42.
- The method, the losses, the data, the number of training steps and which
  neurons get edited.
- bf16 weights. No quantization anywhere.

## Evidence that the results are the same

Checked on CPU with small, randomly initialised Llama models, running the
original and the patched code side by side with the same seed:

| Test | Result |
|---|---|
| The authors' own test suite, with our pinned versions, before and after the patch | 345 passed, 1 skipped both times |
| Training only (float32): learned decoy directions, original vs. patched | differ by at most 1.3 × 10⁻⁷ (they moved 0.7-0.9 during training) |
| Training only: loss values at every step | identical to 4 significant digits |
| Full `apply` run (bf16): chosen layers, chosen neurons, saved record | identical |
| Full `apply` run: saved weights | identical tensor names; largest value difference 6 × 10⁻⁵, one bf16 rounding step |

For scale: two runs of the *unpatched* code on the same GPU can also differ in
the last decimal places, because some GPU operations add numbers in a slightly
different order each time. The patch introduces differences of that same kind.

## Our claim

The defended model is built with the authors' method and their exact published
configuration. Their code was modified only so it fits in less memory, and we
verified that the modification does not change the computation beyond rounding.

Patch sha256: `e2d5042a1f52f8c8a70af54a42f4cacd18a8808815ffb95bd485fd0f4f7f4200`
(also recorded in each build's `manifest.json`).
