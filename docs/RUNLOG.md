# GPU run log

What was run on the rented GPU and what happened. Times in UTC.

The box-only scripts (`queue*.sh`, `obl_run.py`), the box logs and the exact package versions are in [`runs/2026-10-28_vast/`](../runs/2026-10-28_vast/). Which command produced each result is in [EXPERIMENTS.md](EXPERIMENTS.md).

## 2026-10-28: vast.ai

**Box:** 1× RTX 3090 24 GB (driver 580.173.02, CUDA 13.0), Python 3.12.3, 31 GB RAM, 12 CPUs, 150 GB disk (140 GB free). Nevada, $0.229/h. Repo at commit `6624b65`.

- **20:28** Box checked; everything within spec. Started the clone and `scripts/setup_env.sh`.
- **20:30** Setup finished in 2 min: the authors' tests and our 5 tests all passed; torch 2.3.0+cu121, `cuda available: True`.
- **20:31** Build started (`scripts/build_ddo_llama3.sh --seed 42`). Licence and preflight checks passed; downloading the pinned Llama-3-8B-Instruct snapshot.
- **20:33** Build crashed before reaching the GPU: `ModuleNotFoundError: No module named 'PIL'`. nnsight 0.3.7 imports Pillow without declaring it (torchvision normally brings it in, but we install nnsight with `--no-deps`). Fix: `pillow==10.4.0` added to `requirements.txt`, and the same version installed into the box's `.venv` by hand. All the imports the build and attacks need then worked.
- **20:42** Build restarted (the model was already downloaded). `build.log` has both attempts.
- **20:45** Build done; the memory patch held on the 3090 (no OOM). The DDO optimisation took 121 s (2 epochs, 256 steps); the final refusal log-odds were 8.16. Coherence gate: **PASS**. Decoy neurons, layer:neuron: 6:5153, 7:11231, 8:6787, 9:11057, 10:1340, 11:1943, 12:2180, 13:8022, 14:3688. The build warned that 2/128 refusal targets don't read as refusals (2%).
- **20:55** `artifacts/ddo_llama3_seed42/` copied home; its `edits.safetensors` sha256 (`3c1e1038…`) matches the box copy.
- **20:56** Started `scripts/run_attacks.sh --seed 42` (7 runs; log at `~/rosef26/attacks_seed42.log` on the box).
- **21:01** Fixed my monitoring: `pgrep -f` over ssh was matching its own command line, so the earlier watchers never saw the build finish. Results so far, as ASR (LlamaGuard-2; substring in brackets, not reportable), 100 JailbreakBench prompts, 3-point RFA, rank 1, 128 unfiltered probes:
  - `undefended` (original Llama-3-8B-Instruct): **0.82** (0.99). The paper reports about 0.80. Each run takes about 6 min, so all 7 take about 45 min.
  - `defended` (DDO, no undo): **0.39** (0.54).
  - `oracle` (zero the 9 known decoys' down_proj columns): **0.80** (0.99). Removing the decoys brings back the undefended rate.
  - `gate_up_cos_ddo`: the authors' gate_up_cos heuristic, m=1 on layers 6-14, picked **exactly the 9 decoys (9/9)**, the same neurons as the oracle: **0.80** (0.99), identical to the oracle. The paper's Table 21 reports 0.17 for this setting (see the open question below).
  - `gate_up_cos_all` (m=1 in all 32 layers, so 32 neurons zeroed): 9/9 decoys among them; **0.83** (0.98). Even without knowing which layers DDO edited, the attack fully succeeds, and zeroing 23 extra natural neurons does no harm.
  - `down_norm_ddo` (m=1, layers 6-14): picked **0/9 decoys**; the decoys' down norm (0.463) is not the largest in any layer. **0.45** (0.57), within noise of `defended` (0.39).
  - `norm_ratio_ddo` (ours, m=1, layers 6-14): also picked **0/9 decoys**; some natural neurons have a gate/up norm ratio above the decoys' 1.30. On its own, norm ratio is not a useful detector. **0.45** (0.55).
- **21:27** Seed 42 attacks done (7 runs, 31 min). `results/seed42/` copied home; it holds only aggregates (the only completion-related key is `n_completions`). Raw completions stay on the box.
- **21:27** Started seed 43: `build_ddo_llama3.sh --seed 43 && run_attacks.sh --seed 43`.

### Open question: why does the paper report 17% for gate_up_cos m=1?

Table 21 reports 17% ASR for gate_up_cos at m=1. On our seed-42 build, the authors' own `undo_attack` (heuristic `gate_up_cos`, m=1, layers 6-14) zeroes the same 9 neurons as the oracle, and the oracle gives 0.80. The repo has no script for Table 21, so their exact setup can't be checked. Possible reasons, most likely first:

1. **A different build.** In replace mode with the published config, each decoy's gate and up rows are copies of each other (cos = 1 exactly), which no natural neuron comes close to. A variant that adds the trigger onto an existing neuron, or uses several decoys or readers per layer, would not have this fingerprint.
2. **"m=1" meant one neuron for the whole model, not one per layer.** That zeroes 1 of the 9 decoys, and the defence would mostly survive. The code's default and our runs use one per layer.
3. **Other differences in their evaluation**, such as RFA settings, or estimating the attack direction before the undo instead of after.

We can claim: "with the published config and the authors' code, gate_up_cos m=1 finds all 9 decoys". We cannot claim that the paper is wrong. Planned checks: `inspect_decoys.py` (each decoy's rank in its layer and globally) and one extra run zeroing only the single top neuron across the whole model (tests reason 2).
- **21:35** Seed 43 build done (training 111.5 s, coherence gate **PASS**), and `artifacts/ddo_llama3_seed43/` copied home. **It uses the same 9 neurons as seed 42.** DDO picks neurons by `low_norm` (the smallest down_proj columns), which doesn't depend on the seed. The gate and up rows are also **identical** in both builds (cos 1.0000; they are 1.30·r̂ and r̂, the refusal direction, which comes from the model and not the seed). Only the decoy vector (down_proj) differs: cos between the two seeds is 0.06-0.21, with the same norm 0.463. So the seed changes only *what the decoy writes*, not *where it sits or what its fingerprint looks like*. gate_up_cos will catch seed 43 in exactly the same way; the second seed tests how much the defence's strength varies, not how much detection varies.
  - `seed43/undefended`: **0.82** (0.99), identical to seed 42 (same original model, greedy decoding).
  - `seed43/defended`: **0.83** (0.98). **On this build the defence does nothing**; seed 42's defended build gave 0.39. Its training log fits this: the final refusal log-odds were **2.75** (seed 42: **8.16**), and the confusion loss spiked to 2.21 at step 160 (epoch 1, just after the attack directions were re-estimated). The coherence gate still said PASS, because it checks that the model stays fluent, not that it is protected. Same neurons, same gate/up rows, same config; only the training seed differs.
  - `seed43/oracle`: **0.80** (0.99), the same as seed 42's oracle. This is expected: zeroing the down columns erases the only part that differs between seeds, so both oracle models are identical.
  - `seed43/gate_up_cos_ddo`: 9/9 decoys, as predicted.
- **21:50** Checked whether seed 43's failure could be our mistake:
  - Both defended models on disk carry exactly their own edits (max diff 0 against `edits.safetensors`), and `seed43/defended` attacked `models/ddo_llama3_seed43`. So the right model was attacked, and it does contain its decoys.
  - The attack code, judge and settings are identical for both seeds, and seed 42 behaved as expected with them.
  - The only change we make to the authors' code is the memory patch. On small models it reproduced their training to within rounding (see `patches/README.md`), and it is the same for both seeds.
  - The weakness was already visible during training (final refusal log-odds 2.75 vs 8.16), before any attack ran.

  Conclusion: most likely real seed-to-seed variance in DDO training, not our error. Not ruled out: rounding differences in bf16 can grow over 256 training steps, the same way GPU non-determinism does in the unpatched code, so one seed's outcome may be fragile. More seeds would measure the spread.
  - Rest of seed 43: `gate_up_cos_all` 9/9, **0.83**; `down_norm_ddo` 0/9, 0.84; `norm_ratio_ddo` 0/9, 0.84. Everything sits at the undefended level, because this build's defence was already ineffective.
- **22:00** Seed 43 attacks done; `results/seed43/` copied home.
- **22:01** Started extra seeds 44, 45 and 46: build, then the `defended` attack only, then delete that seed's model (disk; it can be rebuilt from `edits.safetensors`). Log at `~/rosef26/extra_seeds.log` on the box.
- **22:18** Extra seeds, defended ASR only (final refusal log-odds from training in brackets):
  - seed 44: **0.80** (2.51)
  - seed 45: **0.18** (11.32)
  - seed 46: running

  Together with seeds 42 (0.39, 8.16) and 43 (0.83, 2.75), the protection ranges from none to strong, and **the final refusal log-odds from training predict it**: high log-odds come with low ASR.

### Natural neuron statistics vs the decoy (original weights, seed-42 decoy layers, CPU on the box)

| layer | stat | p50 | p99 | p99.9 | max | host neuron (original) |
|---|---|---|---|---|---|---|
| 6 | cos(g,u) | 0.089 | 0.300 | 0.541 | 0.767 | 0.051 |
| 6 | \|g\| | 0.850 | 1.288 | 1.607 | 2.092 | 3.141 |
| 6 | \|u\| | 0.656 | 0.713 | 0.770 | 1.073 | 0.433 |
| 6 | \|d\| | 0.651 | 0.725 | 0.793 | 0.903 | 0.394 |
| 10 | cos(g,u) | 0.084 | 0.279 | 0.554 | 0.733 | 0.464 |
| 10 | \|g\| | 0.811 | 1.335 | 1.729 | 2.022 | 2.258 |
| 10 | \|u\| | 0.673 | 0.785 | 0.853 | 1.105 | 0.433 |
| 10 | \|d\| | 0.667 | 0.773 | 0.839 | 0.971 | 0.371 |
| 14 | cos(g,u) | 0.082 | 0.481 | 0.694 | 0.790 | 0.460 |
| 14 | \|g\| | 0.806 | 1.389 | 2.465 | 2.822 | 2.563 |
| 14 | \|u\| | 0.692 | 0.828 | 0.906 | 1.211 | 0.421 |
| 14 | \|d\| | 0.682 | 0.800 | 0.905 | 1.165 | 0.327 |

The decoy has |g| = 1.30, |u| = 1.00, |d| = 0.463 and cos = 1. Its |u| = 1.00 is also beyond the natural p99.9 for up-row norms, which is a second fingerprint.

For the decoy to keep working, its gate must read 1.3·r̂ and (up's r̂ part) × |d| must equal 0.463. If padding is added orthogonally, cos = 0.60/(|g|·|u|·|d|). Keeping every norm at or below about the natural p99 (|g| 1.3, |u| 0.8, |d| 0.8) gives cos ≈ 0.72: near the natural max, but not unique. Reaching cos ≈ 0.3 (p99) would need norms at or above the natural max. So **single-neuron camouflage can push the decoy from #1 into the top handful, but not into the crowd.**

DDO's hosts (the smallest down columns) have large gate rows (|g| 2.3-3.1). So additive mode could naturally give cos ≈ 0.35-0.5 there; this needs testing.
  - seed 46: **0.24** (5.53). Defended ASR over 5 seeds: 0.39, 0.83, 0.80, 0.18, 0.24 (mean 0.49). Every build passed the coherence gate.
- **22:27** All extra-seed results and build records copied home. Deleted `models/ddo_llama3_seed43` on the box to free disk (it can be rebuilt from its edits).
- **22:28** Camouflage test 1, **additive mode**: `build_ddo_llama3.sh --seed 42 --name additive_seed42 -- --compile_mode additive`, then `run_attacks.sh --name additive_seed42`. Both scripts gained `--name` (separate output folders) and pass-through `apply` flags; the updated scripts were copied to the box.
- **22:30** Additive build (`additive_seed42`) done: final refusal log-odds **8.49** (seed 42 replace mode: 8.16); coherence gate PASS; same 9 host neurons. `inspect_decoys.py` on it: each decoy's rank among its layer's 14,335 natural neurons:

  | layer | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
  |---|---|---|---|---|---|---|---|---|---|
  | gate_up_cos | 0.335 (#88) | 0.674 (#2) | 0.379 (#79) | 0.652 (#14) | 0.297 (#117) | 0.577 (#18) | 0.539 (#44) | 0.542 (#53) | 0.576 (#76) |
  | norm_ratio rank | #12 | #510 | #68 | #67 | #59 | #63 | #49 | #78 | #99 |
  | down_norm rank | #13368 | #9472 | #13737 | #13303 | #13971 | #14005 | #14147 | #14015 | #14025 |

  **No detector ranks any decoy first.** gate_up_cos m=1 should miss all 9; m=10 would catch only layer 7's.
- **22:30** `camouflage.py` written (noise in low-variance MLP-input directions, plus an optional split across k neurons; reports ranks, an activation-weighted cosine and KL to the base build). Smoke-tested on the box with a tiny random Llama: cos matched the formula exactly, and the edits file round-trips through `reapply_edits.py`.
- **22:37** Additive attacks so far: `defended` **0.04** (0.11); `oracle` 0.80 (0.99); `gate_up_cos_ddo` 0/9 decoys found, **0.03** (0.11). Much stronger than replace mode (0.39). Its training kept **126 harmful prompts** in the confusion pool, against 17 for replace seed 42, which may explain this. Caveat: a model that refuses everything also scores about 0, so this needs a harmless-prompt check.
- **22:40** Wrote `scripts/benign.py`: refusal and degeneracy rates on 100 harmless test prompts, no attack; aggregates to `results/benign/`, raw output to `models/results/`. Queued `queue.sh` to run when the additive attacks end:
  - benign on original, seed-42 replace and additive
  - additive gate_up_cos m=10
  - `camouflage.py` variants `noise_seed42` (up and down norms 0.8) and `split2_seed42` (the same with `--split 2`), each followed by benign, defended, gate_up_cos m=1 and m=10, and oracle

  The command was typed ahead into the tmux pane (it needs that shell's HF_TOKEN); log at `~/rosef26/queue.log`.
- **22:47** Additive attacks complete: `defended` 0.04, `gate_up_cos_ddo` 0/9 **0.03**, `gate_up_cos_all` 0/9 0.08, `down_norm_ddo` 0/9 0.04, `norm_ratio_ddo` 0/9 0.05, `oracle` 9/9 0.80. Harmless-prompt check (`results/benign/`, 100 harmless test prompts, no attack):

  | model | refusal rate | degenerate | mean words |
  |---|---|---|---|
  | original | 0.00 | 0.00 | 83.4 |
  | replace seed 42 | 0.03 | 0.00 | 81.5 |
  | additive seed 42 | 0.02 | 0.00 | 81.7 |

  So additive's low ASR is real protection, not blanket refusal. Not tested: borderline prompts (XSTest-style), which is where over-refusal usually shows up.
- **22:47-23:13** Camouflage queue (`queue.sh`):
  - additive `gate_up_cos_m10_ddo`: 1/9 decoys found, **0.00**.
  - `noise_seed42` (gate 1.3, up 0.8, down 0.8; noise in the 1024 lowest-variance MLP-input directions): cos 0.723, decoys rank #1-#13 in their layer; noise leak 1-2% of r̂; KL mean 1.3e-3; harmless refusals 3%. `defended` 0.44; `gate_up_cos_ddo` 1/9, 0.29; `gate_up_cos_m10_ddo` 8/9, **0.49**.
  - `split2_seed42` (same, but split over 2 neurons): cos 0.362, ranks #64-#232; KL mean 1.3e-3; harmless refusals 3%. `defended` 0.44; `gate_up_cos_ddo` 0/18.
- **23:10** Queued `queue2.sh`: additive seeds 43 and 44 (build, then `defended`, `gate_up_cos_ddo` and benign). It first deletes the noise, split, replace-42 and additive-42 models to get under the 60 GB build check; all can be rebuilt from their edits.
- **23:12** Installed OBLITERATUS (commit `43b5d6d`, AGPL-3.0) into the box's `.venv` with `--no-deps` and a constraints file of the existing pins: torch, transformers, numpy and the rest are unchanged, and the imports work. Telemetry is off by default and forced off with `OBLITERATUS_TELEMETRY=0`. Its docs report that `basic` failed to remove refusals on two other hardened Llama-3-8B models (DeepRefusal, Circuit Breakers RR) on a 24 GB 4090.
- **23:13** `attack.py --no-rfa`: judges a model as it is, generating with the authors' `generate_under_attack(directions=None)` (their unattacked baseline, the same decoding as RFA). Queued `queue3.sh`: OBLITERATUS `--method advanced --dtype bfloat16 --seed 42` on the original, replace seed 42 and additive seed 42; each result judged with `--no-rfa` (LlamaGuard-2) and benign.
- **23:24** Rest of `split2_seed42`: `gate_up_cos_ddo` 0/18, 0.39; `gate_up_cos_m10_ddo` **0/18, 0.31**; `oracle` 18/18, 0.83. Splitting hides the decoys even from m=10, while protecting as well as plain replace mode. Queue 1 done.
- **23:32** Additive seed 43: final refusal log-odds 9.61 (replace seed 43: 2.75); coherence PASS. `defended` **0.00**; `gate_up_cos_ddo` 0/9, **0.00**. Harmless refusals **8%**, higher than additive 42 (2%) and replace 42 (3%), so additive 43 is somewhat more over-cautious. Seed 44 building.
- **23:40** Additive seed 44: final refusal log-odds 8.78 (replace seed 44: 2.51); `defended` **0.03**; `gate_up_cos_ddo` 0/9, **0.01**; harmless refusals 2%. Queue 2 done. **Additive across seeds 42/43/44: defended 0.04/0.00/0.03, against replace mode's 0.39/0.83/0.80 for the same seeds.**
- **23:41** OBLITERATUS started on the original model. Its memory estimator warns that it cannot model our pinned torch 2.3 / transformers 4.51 (it calls newer APIs). The warnings are estimate-only, and it proceeds through its phases (summon → baseline → probe → distill → validate → excise).
- **23:43** OBLITERATUS's first run abliterated the model but crashed while saving: its checkpoint export needs `accelerate.utils.align_module_device`, which our pinned accelerate 0.33 lacks. Stopped the queue and removed OBLITERATUS from `.venv`. It now gets its own `.venv-obl` (unpinned install: torch 2.15.0+cu132, transformers 5.21.0, accelerate 1.15.0). A bf16 matmul works on the GPU even though the driver reports CUDA 13.0.
- **23:50** Restarted `queue3.sh` with `.venv-obl/bin/obliteratus`. Each output's tensor names are checked against the input model, and the input's config and tokenizer files are copied over, so our transformers 4.51 judge loads the same weights. Disk: 50 GB free.
- **23:50** Second OBLITERATUS failure, again at the save step: its memory admission check refused ("needs 3.69 GiB on cpu but only 1.39 GiB available"). It counts only free RAM; the box had 26 GB available (27 GB reclaimable page cache). Restarted with `OBLITERATUS_PHASE_ADMISSION=off`, the bypass its error message names.
- **23:52** Third failure, at export: `NativeExportCapacityError` (it saw 0.8 GB of host memory). OBLITERATUS's `host_resources._cgroup_available_bytes` uses cgroup limit − `memory.current`, which counts reclaimable page cache as used, and no setting overrides it. Now run through `obl_run.py` on the box, which disables only that cgroup reading, so the estimate falls back to psutil's MemAvailable (measured: 4.1 GiB before, 28.9 GiB after). The abliteration itself is untouched, and their files are not edited. Restarted `queue3.sh`.
- **00:06** OBLITERATUS `advanced` (default method; 4 directions, norm-preserving), judged with `attack.py --no-rfa` (LlamaGuard-2, the same 100 JailbreakBench prompts):

  | model | ASR | harmless refusals |
  |---|---|---|
  | original | **0.15** (0.21) | 0% |
  | replace seed 42 | 0.03 (0.07) | 2% |
  | additive seed 42 | 0.02 (0.05) | 1% |

  Every checkpoint's tensor names matched its input (291). **On Llama-3-8B with our judge, `advanced` is a weak attack** (0.15 on the undefended model, against 0.82 for RFA), so blocking it says little about the defence.
- **00:10** `queue4.sh METHOD`: the same pipeline for any method, but the DDO models run only if the original reaches ASR ≥ 0.5. The first launch used `$METHOD_$name` (bash reads that as `$METHOD_`); it was stopped before writing anything, fixed to `${METHOD}_$name`, and restarted with `aggressive`.
- **00:22** OBLITERATUS `aggressive` on the original: **0.73** (0.99); harmless refusals 0%, nothing degenerate, mean 73 words. A real attack, so the queue continued to replace seed 42 and then additive seed 42.
- **00:46** OBLITERATUS `aggressive` (8 directions, 3 passes), judged by LlamaGuard-2:

  | model | ASR | harmless refusals | mean words |
  |---|---|---|---|
  | original | 0.73 (0.99) | 0% | 73 |
  | replace seed 42 | **0.55** (0.98) | 0% | 82 |
  | additive seed 42 | **0.10** (0.30) | 0% | 83 |

  Replace-mode DDO mostly falls to a strong multi-direction attack. Additive holds. All checkpoints passed the tensor-name check, and no model got degenerate. Queue 4 done; results copied home.
- **00:55** Saved to the repo everything that existed only on the box: logs, queue scripts, `obl_run.py`, `pip freeze` of both venvs, the OBLITERATUS commit and box info (in `runs/2026-10-28_vast/`). The logs were checked: they contain no generated text and no token. Raw completions (`models/results/`) stay on the box and are destroyed with it. The box's `scripts/` matched the local ones, except that local `run_attacks.sh` has a `gate_up_cos_m10_ddo` line the box never ran (the m=10 runs went through the queue scripts with the same arguments).
