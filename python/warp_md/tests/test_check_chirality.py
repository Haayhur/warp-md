import numpy as np
import pytest
import warp_md

from warp_md.analysis.check_chirality import check_chirality


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms, selection_map=None):
        self._n_atoms = n_atoms
        self._atoms = {"mass": [1.0] * n_atoms}
        self._selection_map = selection_map or {}

    def select(self, mask):
        if mask in self._selection_map:
            return _DummySelection(self._selection_map[mask])
        return _DummySelection(range(self._n_atoms))

    def select_indices(self, indices):
        return _DummySelection(indices)

    def atom_table(self):
        return self._atoms


class _DummyTraj:
    def __init__(self, coords):
        self._coords = coords
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        return {"coords": self._coords}


def _center(frame_coords, idx, masses, mass_weighted):
    if idx.size == 0:
        return np.zeros((3,), dtype=np.float64)
    sel = frame_coords[idx, :]
    if not mass_weighted:
        return sel.mean(axis=0)
    w = masses[idx]
    wsum = float(np.sum(w))
    if wsum <= 0.0:
        return sel.mean(axis=0)
    return np.sum(sel * w[:, None], axis=0) / wsum


def _normalize_frame_indices(frame_indices, n_frames):
    if frame_indices is None:
        return list(range(n_frames))
    out = []
    for raw in frame_indices:
        idx = int(raw)
        if idx < 0:
            idx += n_frames
        if 0 <= idx < n_frames:
            out.append(idx)
    return out


class _DummyChiralityPlan:
    def __init__(self, groups, mass_weighted=False):
        self._groups = groups
        self._mass_weighted = bool(mass_weighted)

    def run(self, traj, system, chunk_frames=None, device="auto", frame_indices=None):
        del chunk_frames, device
        coords = np.asarray(traj._coords, dtype=np.float64)
        rows = _normalize_frame_indices(frame_indices, coords.shape[0])
        coords = coords[np.asarray(rows, dtype=np.int64)]
        masses = np.asarray(system.atom_table().get("mass", []), dtype=np.float64)
        if masses.size != coords.shape[1]:
            masses = np.ones((coords.shape[1],), dtype=np.float64)
        out = np.zeros((coords.shape[0], len(self._groups)), dtype=np.float32)
        for f in range(coords.shape[0]):
            frame = coords[f]
            for g, (a, b, c, d) in enumerate(self._groups):
                ia = np.asarray(list(a.indices), dtype=np.int64)
                ib = np.asarray(list(b.indices), dtype=np.int64)
                ic = np.asarray(list(c.indices), dtype=np.int64)
                idd = np.asarray(list(d.indices), dtype=np.int64)
                pa = _center(frame, ia, masses, self._mass_weighted)
                pb = _center(frame, ib, masses, self._mass_weighted)
                pc = _center(frame, ic, masses, self._mass_weighted)
                pd = _center(frame, idd, masses, self._mass_weighted)
                vol = float(np.dot(np.cross(pb - pa, pc - pa), pd - pa))
                out[f, g] = np.float32(vol)
        return out


@pytest.fixture(autouse=True)
def _mock_rust_plan(monkeypatch):
    monkeypatch.setattr(warp_md, "CheckChiralityPlan", _DummyChiralityPlan, raising=False)


def test_check_chirality_signed_volume():
    coords = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    out = check_chirality(traj, system, groups=[(0, 1, 2, 3)])
    assert out.shape == (2, 1)
    assert out[0, 0] > 0.0
    assert out[1, 0] < 0.0


def test_check_chirality_labels():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1e-12]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    values, labels = check_chirality(
        traj,
        system,
        groups=[(0, 1, 2, 3)],
        return_labels=True,
        planar_tolerance=1e-6,
    )
    assert values.shape == (3, 1)
    np.testing.assert_array_equal(labels[:, 0], np.array([1, -1, 0], dtype=np.int8))


def test_check_chirality_string_groups_plan():
    coords = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(
        coords.shape[1],
        selection_map={"A": [0], "B": [1], "C": [2], "D": [3]},
    )

    out = check_chirality(traj, system, groups=[("A", "B", "C", "D")])
    assert out.shape == (2, 1)
    assert out[0, 0] > 0.0
    assert out[1, 0] < 0.0


def test_check_chirality_frame_indices_forwarded():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    out = check_chirality(traj, system, groups=[(0, 1, 2, 3)], frame_indices=[1])
    assert out.shape == (1, 1)
    assert out[0, 0] > 0.0


def test_check_chirality_frame_indices_negative_duplicates_and_oob():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    out = check_chirality(
        traj,
        system,
        groups=[(0, 1, 2, 3)],
        frame_indices=[-1, -1, 0, 99, -99],
    )
    assert out.shape == (3, 1)
    assert out[0, 0] > 0.0
    assert out[1, 0] > 0.0
    assert np.isclose(out[0, 0], out[1, 0])
    assert out[2, 0] < 0.0


def test_check_chirality_rejects_non_bool_flags():
    coords = np.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    with pytest.raises(ValueError, match="mass_weighted must be bool"):
        check_chirality(traj, system, groups=[(0, 1, 2, 3)], mass_weighted=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="return_labels must be bool"):
        check_chirality(traj, system, groups=[(0, 1, 2, 3)], return_labels=1)  # type: ignore[arg-type]


def test_check_chirality_rejects_invalid_planar_tolerance():
    coords = np.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    with pytest.raises(ValueError, match="planar_tolerance must be a finite value >= 0"):
        check_chirality(traj, system, groups=[(0, 1, 2, 3)], planar_tolerance=-1e-8)
    with pytest.raises(ValueError, match="planar_tolerance must be a finite value >= 0"):
        check_chirality(traj, system, groups=[(0, 1, 2, 3)], planar_tolerance=np.inf)


def test_check_chirality_rejects_invalid_group_shape():
    coords = np.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    with pytest.raises(ValueError, match="each chirality group must have 4 selections"):
        check_chirality(traj, system, groups=[(0, 1, 2)])
