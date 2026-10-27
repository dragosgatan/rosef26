# GPU box: step by step

## Before renting

- Read token from https://huggingface.co/settings/tokens, kept on your PC (never in a file in the repo).
- Everything committed and pushed: the box clones from GitHub.

## What to rent

| | Aim for |
|---|---|
| GPU | 1× RTX 3090, 24 GB |
| Disk | 100 GB (three 16 GB models + venv; the build refuses under 60 GB free) |
| RAM | 32 GB (24 GB minimum) |
| CPU | 8+ cores |
| CUDA driver | 12.1 or newer |
| Image | Ubuntu 22.04 or 24.04 (Python 3.10-3.12) |
| Download speed | 500+ Mbps (about 50 GB of models) |
| Host | high reliability / verified; runs take hours |

## On the box

1. SSH into the box, then start tmux there. Everything after this runs inside tmux, so a dropped SSH connection doesn't kill the run. To get back in: SSH again, then `tmux attach -t ddo`.
   ```bash
   apt-get update && apt-get install -y python3-venv git tmux   # skip if already there
   tmux new -s ddo
   ```
2. Check the box:
   ```bash
   nvidia-smi          # RTX 3090, 24576 MiB, CUDA 12.x
   python3 --version   # 3.10-3.12
   df -h ~             # 100 GB free
   ```
3. Clone and set up (~10-15 min, ends with the tests passing and `cuda available: True`):
   ```bash
   git clone --recurse-submodules https://github.com/dragosgatan/rosef26.git && cd rosef26
   scripts/setup_env.sh
   ```
4. Token:
   ```bash
   export HF_TOKEN=hf_...
   ```
5. Build (~30-45 min). Watch the first 10 min; on an out-of-memory error, stop and keep `artifacts/ddo_llama3_seed42/build.log`.
   ```bash
   scripts/build_ddo_llama3.sh --seed 42
   ```
6. Copy the record home right away, from a terminal **on your PC** (host and port from the rental's SSH command):
   ```bash
   scp -P PORT -r root@HOST:rosef26/artifacts ~/documents/projectsvsc/rosef26/
   ```
7. Attacks (~2-3 h, a guess; time the first run). Safe to rerun after a crash; finished runs are skipped.
   ```bash
   scripts/run_attacks.sh --seed 42
   ```
8. Copy the results home, from your PC:
   ```bash
   scp -P PORT -r root@HOST:rosef26/results ~/documents/projectsvsc/rosef26/
   ```
   Don't copy `models/results/` (raw completions) into the repo.
9. Destroy the box (not just stop it), so the models and your token are gone.

Optional second seed: repeat steps 5-8 with `--seed 43`.

Back home: commit `artifacts/` and `results/`, then run `scripts/inspect_decoys.py` (see README).
