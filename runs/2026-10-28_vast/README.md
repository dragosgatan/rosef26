# GPU run, 2026-10-28/29 (vast.ai, RTX 3090)

Saved from the rented box before it was destroyed. See [docs/RUNLOG.md](../../docs/RUNLOG.md) and [docs/EXPERIMENTS.md](../../docs/EXPERIMENTS.md).

- **`scripts/`**: the queue scripts run in the box's tmux session, plus `obl_run.py` (the OBLITERATUS wrapper). They expect to run from the repo root on the box.
- **`logs/`**: their output, plus the setup and attack logs. The logs hold progress output only; generated text was never logged.
- **`env/`**:
  - `pip freeze` of `.venv` and `.venv-obl`
  - the OBLITERATUS commit
  - `box-info.txt`: GPU, driver, OS, disk and RAM
  - the box's repo commit and working-tree status (tonight's scripts were copied in by hand and match the committed ones)
