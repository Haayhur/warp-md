import numpy as np
import pytest
import warp_md

from warp_md.analysis.xtalsymm import xtalsymm


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms):
        self._n_atoms = n_atoms
        self._atoms = {"resid": list(range(n_atoms))}

    def select(self, _mask):
        return _DummySelection(list(range(self._n_atoms)))

    def select_indices(self, indices):
        return _DummySelection(list(indices))

    def atom_table(self):
        return self._atoms

    def n_atoms(self):
        return self._n_atoms


class _DummyTraj:
    def __init__(self, coords):
        self._coords = coords
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        return {"coords": self._coords}


def test_xtalsymm_uses_plan_and_slices(monkeypatch):
    traj = _DummyTraj(np.zeros((2, 1, 3), dtype=np.float32))
    system = _DummySystem(1)
    called = {}
    raw = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    class _DummyPlan:
        def __init__(self, sel, repeats, symmetry_ops=None):
            called["n_sel"] = len(sel.indices)
            called["repeats"] = repeats
            called["symmetry_ops"] = symmetry_ops

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["frame_indices"] = frame_indices
            if frame_indices is None:
                return raw.reshape(-1)
            selected = raw[np.asarray(frame_indices, dtype=np.int64)]
            return selected.reshape(-1)

    monkeypatch.setattr(warp_md, "XtalSymmPlan", _DummyPlan, raising=False)
    out = xtalsymm(traj, system, mask="all", repeats=(2, 1, 1), frame_indices=[1])
    got = out.read_chunk()["coords"]
    assert called["n_sel"] == 1
    assert called["repeats"] == [2, 1, 1]
    assert called["symmetry_ops"] is None
    assert called["frame_indices"] == [1]
    assert got.shape == (1, 2, 3)
    np.testing.assert_allclose(got[0], raw[1], atol=1e-6)


def test_xtalsymm_passes_symmetry_ops_to_plan(monkeypatch):
    traj = _DummyTraj(np.zeros((1, 1, 3), dtype=np.float32))
    system = _DummySystem(1)
    called = {}

    class _DummyPlan:
        def __init__(self, _sel, _repeats, symmetry_ops=None):
            called["symmetry_ops"] = symmetry_ops

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["frame_indices"] = frame_indices
            return np.array([1.0, 2.0, 3.0], dtype=np.float32)

    monkeypatch.setattr(warp_md, "XtalSymmPlan", _DummyPlan, raising=False)
    out = xtalsymm(
        traj,
        system,
        mask="all",
        repeats=(1, 1, 1),
        symmetry_ops=[np.eye(3, dtype=np.float64)],
    )
    got = out.read_chunk()["coords"]
    assert got.shape == (1, 1, 3)
    assert called["symmetry_ops"] == [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
    assert called["frame_indices"] is None


def test_xtalsymm_rejects_invalid_symmetry_op_shape():
    traj = _DummyTraj(np.zeros((1, 1, 3), dtype=np.float32))
    system = _DummySystem(1)
    with pytest.raises(ValueError, match="each symmetry op must be shape"):
        xtalsymm(
            traj,
            system,
            mask="all",
            repeats=(1, 1, 1),
            symmetry_ops=[np.zeros((2, 2), dtype=np.float64)],
        )


def test_xtalsymm_affine_4x4_converts_to_3x4_payload(monkeypatch):
    traj = _DummyTraj(np.zeros((1, 1, 3), dtype=np.float32))
    system = _DummySystem(1)
    called = {}

    class _DummyPlan:
        def __init__(self, _sel, _repeats, symmetry_ops=None):
            called["symmetry_ops"] = symmetry_ops

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return np.array([1.0, 2.0, 3.0], dtype=np.float32)

    monkeypatch.setattr(warp_md, "XtalSymmPlan", _DummyPlan, raising=False)
    op = np.array(
        [
            [1.0, 0.0, 0.0, 1.5],
            [0.0, 1.0, 0.0, -2.0],
            [0.0, 0.0, 1.0, 0.25],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    out = xtalsymm(
        traj,
        system,
        mask="all",
        repeats=(1, 1, 1),
        symmetry_ops=[op],
    )
    got = out.read_chunk()["coords"]
    assert got.shape == (1, 1, 3)
    assert called["symmetry_ops"] == [[1.0, 0.0, 0.0, 1.5, 0.0, 1.0, 0.0, -2.0, 0.0, 0.0, 1.0, 0.25]]


def test_xtalsymm_rejects_non_affine_4x4():
    traj = _DummyTraj(np.zeros((1, 1, 3), dtype=np.float32))
    system = _DummySystem(1)
    bad = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.1, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="affine"):
        xtalsymm(
            traj,
            system,
            mask="all",
            repeats=(1, 1, 1),
            symmetry_ops=[bad],
        )


def test_xtalsymm_rejects_invalid_repeats():
    traj = _DummyTraj(np.zeros((1, 1, 3), dtype=np.float32))
    system = _DummySystem(1)
    with pytest.raises(ValueError, match="repeats must contain positive integers"):
        xtalsymm(traj, system, mask="all", repeats=(0, 1, 1))
    with pytest.raises(ValueError, match="repeats must be a 3-item tuple/list"):
        xtalsymm(traj, system, mask="all", repeats=(1, 1))  # type: ignore[arg-type]


def test_xtalsymm_rejects_nonfinite_symmetry_op_values():
    traj = _DummyTraj(np.zeros((1, 1, 3), dtype=np.float32))
    system = _DummySystem(1)
    op = np.eye(3, dtype=np.float64)
    op[0, 0] = np.inf
    with pytest.raises(ValueError, match="symmetry op values must be finite"):
        xtalsymm(
            traj,
            system,
            mask="all",
            repeats=(1, 1, 1),
            symmetry_ops=[op],
        )


def test_xtalsymm_rejects_legacy_constructor_signature(monkeypatch):
    traj = _DummyTraj(np.zeros((1, 1, 3), dtype=np.float32))
    system = _DummySystem(1)

    class _LegacyPlan:
        def __init__(self, _sel, _repeats):  # no symmetry_ops kw
            pass

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return np.array([1.0, 2.0, 3.0], dtype=np.float32)

    monkeypatch.setattr(warp_md, "XtalSymmPlan", _LegacyPlan, raising=False)
    with pytest.raises(RuntimeError, match="requires updated Rust bindings with symmetry_ops support"):
        xtalsymm(traj, system, mask="all", repeats=(1, 1, 1))
