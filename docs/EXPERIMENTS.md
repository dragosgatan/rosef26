# Experiments: how each result was made

All runs were on rented RTX 3090s (24 GB): the main run on 2026-10-28/29 and the follow-up on 2026-10-29. The timeline, every fix and the box details are in [RUNLOG.md](RUNLOG.md). The box's own scripts, logs and package versions are in [`runs/2026-10-28_vast/`](../runs/2026-10-28_vast/).

## Fixed for every result

| | |
|---|---|
| Model | `meta-llama/Meta-Llama-3-8B-Instruct` @ `8afb486`, bf16, no quantization |
| DDO code | `third_party/ddo` @ `a26f23e` plus `patches/ddo-24gb-vram.patch` (memory only) |
| Published config | layers 6-14, β 1.30, scale 0.463, λ 1.12, lr 0.0026, 2 epochs |
| Eval prompts | JailbreakBench, 100 behaviours |
| RFA attack | the authors' `run_rfa`: 3-point, rank 1, 128 unfiltered probes, greedy, 512 new tokens, seed 42 |
| Judge | LlamaGuard-2 (`meta-llama/Meta-Llama-Guard-2-8B`), via the authors' `LlamaGuard2Judge`; the substring judge is logged but not reportable |
| Undo attack | the authors' `undo_attack` (gate_up_cos, down_norm); norm_ratio is ours; m neurons per layer, `down_proj` columns zeroed |
| Harmless check | `scripts/benign.py`: first 100 harmless test prompts, 128 tokens, refusal phrases and loop detector from the authors' code |
| Environment | `.venv` from `requirements.txt` (versions in `runs/.../env/pip-freeze-venv.txt`) |

Each `results/<folder>/<run>.json` holds rates and settings only. Raw completions were kept on the box and destroyed with it.

## Result folders

| folder | what | command |
|---|---|---|
| `seed42/`, `seed43/` | replace-mode DDO: undefended, defended, oracle, gate_up_cos (layers 6-14 and all), down_norm, norm_ratio | `scripts/build_ddo_llama3.sh --seed N`, then `scripts/run_attacks.sh --seed N` |
| `seed44/` to `seed46/` | replace-mode DDO, `defended` only | build, then `scripts/attack.py --model models/ddo_llama3_seedN --tag seedN/defended --edits artifacts/ddo_llama3_seedN/edits.safetensors` (loop in RUNLOG, 22:01) |
| `additive_seed42/` | additive mode: the same 6 runs plus gate_up_cos m=10 | `build_ddo_llama3.sh --seed 42 --name additive_seed42 -- --compile_mode additive`, `run_attacks.sh --name additive_seed42`; m=10 via `runs/.../scripts/queue.sh` |
| `additive_seed43/`, `additive_seed44/` | additive mode: defended, gate_up_cos m=1 | `runs/.../scripts/queue2.sh` |
| `nofilter_seed43/`, `nofilter_seed44/` | replace mode with the confusion filter off: defended, gate_up_cos m=1 | `build_ddo_llama3.sh --seed N --name nofilter_seedN --no-confusion-filter`, attacks via `scripts/queue_followup.sh` |
| `additive_seed45/`, `additive_seed46/` | additive mode: defended, gate_up_cos m=1 | `scripts/queue_followup.sh` |
| `noise_seed42/`, `split2_seed42/` | post-hoc camouflage of the replace seed-42 build: defended, gate_up_cos m=1 and m=10, oracle | `scripts/camouflage.py --name noise_seed42 --up-norm 0.8 --down-norm 0.8` (plus `--split 2` for split2), attacks via `queue.sh` |
| `benign/` | harmless-prompt check for each model above and each OBLITERATUS output | `scripts/benign.py`, via the queue scripts |
| `obliteratus/` | OBLITERATUS `advanced` and `aggressive` on the original, replace 42 and additive 42, then judged with no RFA | `queue3.sh` (advanced) and `queue4.sh aggressive`; see below |

## Build records (`artifacts/`)

- **Replace and additive builds** (`ddo_llama3_seed42`-`46`, `additive_seed42`-`44`): `build.log` (full training log), `config.json` (exact command), `ddo_defense.json`, `edits.safetensors` (every edited row and column, which rebuilds the model with `scripts/reapply_edits.py`) and `manifest.json` (versions, GPU, shard hashes).
- **Camouflage variants** (`noise_seed42`, `split2_seed42`): `edits.safetensors` relative to the original model, plus `camouflage.json` (settings, per-layer cos/ranks, noise leak, activation-weighted cosine, KL to the base build).
- **`additive_seed42/inspect.json`**: `scripts/inspect_decoys.py` output (decoy ranks among natural neurons).

## OBLITERATUS

- **Version:** commit `43b5d6d`, AGPL-3.0, in its own venv `.venv-obl`; versions are in `runs/.../env/pip-freeze-venv-obl.txt`: torch 2.15, transformers 5.21. It is not installed in `.venv`.
- **Command:** `obliterate <model> --method {advanced,aggressive} --dtype bfloat16 --seed 42 --output-dir ...`, with `OBLITERATUS_TELEMETRY=0` and `OBLITERATUS_PHASE_ADMISSION=off`.
- **Wrapper:** run through `runs/.../scripts/obl_run.py`, which disables only its cgroup memory reading. That reading counted reclaimable page cache as used and refused to save. The attack itself is unchanged.
- **Before judging:** each output's tensor names were checked against its input (all 291 matched). The input's config and tokenizer files were copied over so `.venv` (transformers 4.51) could load it. The result was then judged with `scripts/attack.py --model ... --no-rfa`, which uses the authors' unattacked generation path.
