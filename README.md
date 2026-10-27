# RoSEF 2026 - Camouflaging DDO

Relevant: [DDO](https://arxiv.org/html/2609.16204v1), [DDO Github](https://github.com/aashiqmuhamed/defending-against-abliteration), [Original Abliteration Paper](https://arxiv.org/html/2406.11717v3), [Comparison of methods](https://arxiv.org/html/2512.13655v2)

What we aim to achieve: Breaking DDO by detection, propose/test a Camouflage of DDO, among other things related to it, like how well it works on smaller models (1.5b-3b)

## Building the DDO-defended Llama-3-8B

Authors' code, their published Llama-3 config, bf16. Needs a 24 GB GPU (RTX 3090), ~1-1.5 h for two seeds. Full box walkthrough: [INSTRUCTIONS.md](INSTRUCTIONS.md).

First: accept the Meta Llama 3 licence at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct (it also covers Llama Guard 2) and create a Read token at https://huggingface.co/settings/tokens.

```bash
git clone --recurse-submodules <your repo URL> && cd rosef26
scripts/setup_env.sh
export HF_TOKEN=hf_...
scripts/build_ddo_llama3.sh --seed 42
scripts/build_ddo_llama3.sh --seed 43    # optional
```

The authors' code is installed with a memory patch so it fits in 24 GB; see [`patches/README.md`](patches/README.md).

**Rebuild the exact defended model later** (any machine, CPU is enough):

```bash
.venv/bin/python scripts/reapply_edits.py \
    --edits artifacts/ddo_llama3_seed42/edits.safetensors \
    --out models/ddo_llama3_seed42
```

## Inspect and attack

```bash
.venv/bin/python scripts/inspect_decoys.py --edits artifacts/ddo_llama3_seed42/edits.safetensors   # CPU
scripts/run_attacks.sh --seed 42                                                                  # GPU
```

`inspect_decoys.py` ranks each decoy among all normal neurons by gate/up cosine, norm ratio and down norm (`inspect.json` next to the edits). `run_attacks.sh` runs RFA after each undo method and writes ASR to `results/`. Raw completions stay in `models/results/` and are never committed.

## Credits

The DDO method and code are by Aashiq Muhamed, Mona T. Diab and Virginia Smith:

> Muhamed, A., Diab, M. T., & Smith, V. (2026). *Decoy Direction Optimization: A Post-Hoc Defense Against LLM Abliteration.* arXiv:2609.16204.

Their code ([github.com/aashiqmuhamed/defending-against-abliteration](https://github.com/aashiqmuhamed/defending-against-abliteration)) is released under the Apache-2.0 licence and is included unmodified as a git submodule in `third_party/ddo`, pinned to commit `a26f23e`. Our only change to it is the memory patch in `patches/`.
