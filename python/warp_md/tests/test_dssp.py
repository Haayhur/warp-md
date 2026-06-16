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

        def run(
            self,
            _traj,
            _system,
            chunk_frames=None,
            device="auto",
            frame_indices=None,
            simplified=False,
        ):
            assert chunk_frames == 64
            assert device == "auto"
            assert frame_indices is None
            assert simplified is False
            labels = ["ALA:1", "GLY:2"]
            codes = np.array([[0, 1], [3, 7]], dtype=np.uint8)
            symbols = ["0", "b", "G", "S"]
            avg = {
                "none_avg": np.array([0.5, 0.0], dtype=np.float32),
                "extended_avg": np.array([0.0, 0.5], dtype=np.float32),
                "3-10_avg": np.array([0.5, 0.0], dtype=np.float32),
                "bend_avg": np.array([0.0, 0.5], dtype=np.float32),
            }
            return labels, codes, symbols, 2, 2, avg

    monkeypatch.setattr(dssp_mod, "_DsspPlan", _DummyPlan, raising=True)
    labels, ss, avg = dssp_mod.dssp(
        _DummyTraj(),
        _DummySystem(),
        mask="protein",
        chunk_frames=64,
        simplified=False,
    )
    assert labels.shape == (2,)
    assert labels[0] == "ALA:1"
    assert ss.shape == (2, 2)
    assert ss[0, 0] == "0"
    assert ss[0, 1] == "b"
    assert ss[1, 0] == "G"
    assert ss[1, 1] == "S"
    np.testing.assert_allclose(avg["none_avg"], np.array([0.5, 0.0], dtype=np.float32))
    np.testing.assert_allclose(avg["extended_avg"], np.array([0.0, 0.5], dtype=np.float32))


def test_dssp_simplified_collapses_8_state_codes(monkeypatch):
    class _DummyPlan:
        def __init__(self, _sel):
            pass

        def run(
            self,
            _traj,
            _system,
            chunk_frames=None,
            device="auto",
            frame_indices=None,
            simplified=False,
        ):
            assert frame_indices is None
            assert simplified is True
            labels = ["ALA:1", "GLY:2", "SER:3"]
            codes = np.array([[3, 2, 6], [5, 1, 7]], dtype=np.uint8)
            symbols = ["H", "E", "C", "H", "E", "C"]
            avg = {
                "3-10_avg": np.array([0.5, 0.0, 0.0], dtype=np.float32),
                "bridge_avg": np.array([0.0, 0.5, 0.0], dtype=np.float32),
                "turn_avg": np.array([0.0, 0.0, 0.5], dtype=np.float32),
                "pi_avg": np.array([0.5, 0.0, 0.0], dtype=np.float32),
                "extended_avg": np.array([0.0, 0.5, 0.0], dtype=np.float32),
                "bend_avg": np.array([0.0, 0.0, 0.5], dtype=np.float32),
            }
            return labels, codes, symbols, 2, 3, avg

    monkeypatch.setattr(dssp_mod, "_DsspPlan", _DummyPlan, raising=True)
    labels, ss, avg = dssp_mod.dssp(
        _DummyTraj(),
        _DummySystem(),
        mask="protein",
        simplified=True,
    )
    assert labels.shape == (3,)
    assert ss.tolist() == [["H", "E", "C"], ["H", "E", "C"]]
    np.testing.assert_allclose(avg["extended_avg"], np.array([0.0, 0.5, 0.0], dtype=np.float32))


def test_dssp_defaults_to_simplified_output(monkeypatch):
    seen = {}

    class _DummyPlan:
        def __init__(self, _sel):
            pass

        def run(
            self,
            _traj,
            _system,
            chunk_frames=None,
            device="auto",
            frame_indices=None,
            simplified=False,
        ):
            seen["simplified"] = simplified
            labels = ["ALA:1"]
            codes = np.array([[0]], dtype=np.uint8)
            symbols = ["C"]
            avg = {"coil_avg": np.array([1.0], dtype=np.float32)}
            return labels, codes, symbols, 1, 1, avg

    monkeypatch.setattr(dssp_mod, "_DsspPlan", _DummyPlan, raising=True)
    labels, ss, avg = dssp_mod.dssp(_DummyTraj(), _DummySystem(), mask="protein")
    assert seen["simplified"] is True
    assert labels.tolist() == ["ALA:1"]
    assert ss.tolist() == [["C"]]
    np.testing.assert_allclose(avg["coil_avg"], np.array([1.0], dtype=np.float32))


def test_dssp_passes_frame_indices_to_rust_plan(monkeypatch):
    traj = _DummyTraj()
    seen = {}

    class _DummyPlan:
        def __init__(self, _sel):
            pass

        def run(
            self,
            got_traj,
            _system,
            chunk_frames=None,
            device="auto",
            frame_indices=None,
            simplified=False,
        ):
            seen["traj"] = got_traj
            seen["chunk_frames"] = chunk_frames
            seen["device"] = device
            seen["frame_indices"] = frame_indices
            seen["simplified"] = simplified
            labels = ["ALA:1"]
            codes = np.array([[1]], dtype=np.uint8)
            symbols = ["b"]
            avg = {"extended_avg": np.array([1.0], dtype=np.float32)}
            return labels, codes, symbols, 1, 1, avg

    monkeypatch.setattr(dssp_mod, "_DsspPlan", _DummyPlan, raising=True)
    labels, ss, avg = dssp_mod.dssp(
        traj,
        _DummySystem(),
        mask="protein",
        frame_indices=[2, 0],
        chunk_frames=8,
        simplified=False,
    )
    assert seen == {
        "traj": traj,
        "chunk_frames": 8,
        "device": "auto",
        "frame_indices": [2, 0],
        "simplified": False,
    }
    assert labels.tolist() == ["ALA:1"]
    assert ss.tolist() == [["b"]]
    np.testing.assert_allclose(avg["extended_avg"], np.array([1.0], dtype=np.float32))


def test_dssp_raises_when_binding_missing(monkeypatch):
    monkeypatch.setattr(dssp_mod, "_DsspPlan", None, raising=True)
    with pytest.raises(RuntimeError, match="PyDsspPlan binding unavailable"):
        dssp_mod.dssp(_DummyTraj(), _DummySystem(), mask="protein")
