# Camouflage ideas

Decoy fingerprint: gate = 1.30·r̂, up = r̂, so cos(gate, up) = 1. gate_up_cos m=1 finds all 9.

1. **Additive mode** (`--compile_mode additive`): add the trigger to the host neuron instead of overwriting it. Tested: decoys rank #2-#117 by cos instead of #1; defended ASR 0.00-0.04 on 3 seeds (FINDINGS §3).
2. **Perpendicular noise**: gate = 1.30·r̂ + n_g, up = a·r̂ + n_u, down rescaled so that a·|down| = 0.463, with n taken from directions the MLP input barely uses. Within natural norms, cos can only go down to about 0.7. Risk: the neuron also reacts to the input along n, so it may fire at the wrong times. That's why n is taken from low-variance directions, and why the script reports KL to the base build. `scripts/camouflage.py`. Tested: cos 0.72, found by gate_up_cos at m=10 (FINDINGS §3).
3. **Split across k neurons**: each neuron carries 1/k of the decoy, so cos ÷ k. Alone, the authors' `--n_decoys k` hides nothing (each decoy still has cos = 1, and the attacker zeroes the top k); it only helps combined with 2. Same script with `--split k`. Tested with k=2: cos 0.36, hidden even at m=10, but no better protection (FINDINGS §3).

Skipped: chaff neurons, rescaling or reordering neurons, `--n_readers`.

Limit: the gate row still reads r̂, so `gate_proj` or an activation-based detector can still find the decoys.
