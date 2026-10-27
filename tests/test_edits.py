"""Edit extraction and reapplication on a tiny fake checkpoint (CPU, no downloads)."""

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from collect_artifacts import differing_tensors, extract_edits  # noqa: E402
from reapply_edits import reapply  # noqa: E402

D, I = 8, 16
NEURONS = {"1": [3], "2": [5, 11]}


def _model():
    g = torch.Generator().manual_seed(0)
    t = {"model.embed_tokens.weight": torch.randn(32, D, generator=g)}
    for l in range(3):
        t[f"model.layers.{l}.mlp.gate_proj.weight"] = torch.randn(I, D, generator=g)
        t[f"model.layers.{l}.mlp.up_proj.weight"] = torch.randn(I, D, generator=g)
        t[f"model.layers.{l}.mlp.down_proj.weight"] = torch.randn(D, I, generator=g)
    return {k: v.to(torch.bfloat16) for k, v in t.items()}


def _defend(t):
    """What replace-mode DDO writes: up = r, gate = 1.3 r, down column = 0.463 u."""
    t = {k: v.clone() for k, v in t.items()}
    g = torch.Generator().manual_seed(1)
    for l, ix in NEURONS.items():
        r = torch.randn(D, generator=g).to(torch.bfloat16)
        for i in ix:
            t[f"model.layers.{l}.mlp.up_proj.weight"][i] = r
            t[f"model.layers.{l}.mlp.gate_proj.weight"][i] = 1.3 * r
            t[f"model.layers.{l}.mlp.down_proj.weight"][:, i] = 0.463 * torch.randn(D, generator=g)
    return t


def _write(path, tensors):
    """Two shards plus an index, like a real sharded checkpoint."""
    path.mkdir()
    names = sorted(tensors)
    shards = {"model-00001-of-00002.safetensors": names[::2],
              "model-00002-of-00002.safetensors": names[1::2]}
    for shard, ns in shards.items():
        save_file({n: tensors[n] for n in ns}, path / shard)
    weight_map = {n: shard for shard, ns in shards.items() for n in ns}
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (path / "config.json").write_text("{}")
    return path


def test_round_trip_rebuilds_the_defended_model_exactly(tmp_path):
    orig = _write(tmp_path / "orig", _model())
    defended = _write(tmp_path / "defended", _defend(_model()))

    edits = extract_edits(orig, defended, NEURONS)
    assert edits["2.neurons"].tolist() == [5, 11]
    assert edits["2.down_proj"].shape == (D, 2)
    save_file(edits, tmp_path / "edits.safetensors")

    reapply(tmp_path / "edits.safetensors", orig, tmp_path / "out")
    assert list(differing_tensors(tmp_path / "out", defended)) == []
    assert (tmp_path / "out" / "config.json").exists()


def test_a_change_outside_the_mlp_is_refused(tmp_path):
    bad = _defend(_model())
    bad["model.embed_tokens.weight"][0, 0] += 1
    orig, defended = _write(tmp_path / "orig", _model()), _write(tmp_path / "defended", bad)
    with pytest.raises(SystemExit, match="embed_tokens"):
        extract_edits(orig, defended, NEURONS)


def test_a_mismatch_with_the_record_is_refused(tmp_path):
    orig = _write(tmp_path / "orig", _model())
    defended = _write(tmp_path / "defended", _defend(_model()))
    with pytest.raises(SystemExit, match="layer 1"):
        extract_edits(orig, defended, {"1": [4], "2": [5, 11]})


def test_reapply_refuses_a_non_empty_output(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "x").write_text("keep me")
    with pytest.raises(SystemExit, match="not empty"):
        reapply(tmp_path / "unused", tmp_path / "unused", tmp_path / "out")
