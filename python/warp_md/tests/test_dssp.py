import importlib

import numpy as np
import pytest

dssp_mod = importlib.import_module("warp_md.analysis.dssp")


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self):
        self._atoms = {
            "resid": [1, 1, 1, 2, 2, 2],
        }

    def atom_table(self):
        return self._atoms

    def select(self, _mask):
        return _DummySelection([0, 1, 2, 3, 4, 5])


class _DummyTraj:
    pass


def test_dssp_uses_rust_plan_wrapper(monkeypatch):
    class _DummyPlan:
        def __init__(self, sel):
            assert len(sel.indices) == 6

        def run(self, _traj, _system, chunk_frames=None, device="auto"):
            assert chunk_frames == 64
            assert device == "auto"
            labels = ["ALA:1", "GLY:2"]
            codes = np.array([[0, 2], [1, 0]], dtype=np.uint8)
            return labels, codes

    monkeypatch.setattr(dssp_mod, "_DsspPlan", _DummyPlan, raising=True)
    labels, ss, avg = dssp_mod.dssp(
        _DummyTraj(),
        _DummySystem(),
        mask="protein",
        chunk_frames=64,
    )
    assert labels.shape == (2,)
    assert labels[0] == "ALA:1"
    assert ss.shape == (2, 2)
    assert ss[0, 0] == "C"
    assert ss[0, 1] == "E"
    assert ss[1, 0] == "H"
    assert ss[1, 1] == "C"
    assert avg["C"] == 2.0
    assert avg["E"] == 1.0
    assert avg["H"] == 1.0


def test_dssp_raises_when_binding_missing(monkeypatch):
    monkeypatch.setattr(dssp_mod, "_DsspPlan", None, raising=True)
    with pytest.raises(RuntimeError, match="PyDsspPlan binding unavailable"):
        dssp_mod.dssp(_DummyTraj(), _DummySystem(), mask="protein")
