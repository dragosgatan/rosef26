"""Decoy ranking on the tiny fake checkpoint from test_edits (CPU, no downloads)."""

import sys
from pathlib import Path

import pytest
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from collect_artifacts import extract_edits  # noqa: E402
from inspect_decoys import inspect  # noqa: E402
from test_edits import NEURONS, _defend, _model, _write  # noqa: E402


def test_planted_decoys_rank_first_by_cosine(tmp_path):
    orig = _write(tmp_path / "orig", _model())
    defended = _write(tmp_path / "defended", _defend(_model()))
    save_file(extract_edits(orig, defended, NEURONS), tmp_path / "edits.safetensors")

    r = inspect(orig, tmp_path / "edits.safetensors")

    assert r["layers"]["0"]["decoys"] == []
    for layer, ix in NEURONS.items():
        decoys = r["layers"][layer]["decoys"]
        assert [d["neuron"] for d in decoys] == ix
        for d in decoys:
            assert d["gate_up_cos"]["value"] == pytest.approx(1.0, abs=1e-3)
            assert d["norm_ratio"]["value"] == pytest.approx(1.3, abs=1e-2)
            assert d["gate_up_cos"]["rank"] == d["gate_up_cos"]["global_rank"] == 1
    assert r["n_normal_neurons"] == 3 * 16 - 3
