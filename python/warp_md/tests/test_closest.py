import importlib

import numpy as np

import warp_md

from warp_md.analysis.closest import _closest_gather, closest, closest_atom


closest_mod = importlib.import_module("warp_md.analysis.closest")


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms):
        self._atoms = {
            "name": ["A"] * n_atoms,
            "resname": ["RES"] * n_atoms,
            "resid": list(range(1, n_atoms + 1)),
            "chain_id": [0] * n_atoms,
            "mass": [1.0] * n_atoms,
        }

    def atom_table(self):
        return self._atoms

    def select(self, _mask):
        return _DummySelection(range(len(self._atoms["name"])))


class _DummyTraj:
    def __init__(self, coords):
        self._coords = coords
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        return {"coords": self._coords}


def test_closest_atom():
    coords = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    system = _DummySystem(coords.shape[0])
    idx = closest_atom(system, coords, point=(1.0, 0.0, 0.0))
    assert idx == 0


def test_closest_basic():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    out = closest(traj, system, mask="all", n_solvents=2)
    chunk = out.read_chunk()
    assert chunk["coords"].shape == (1, 2, 3)


def test_closest_gather_native_route(monkeypatch):
    calls = []

    def fake_native(coords, keep_idx):
        calls.append((coords.copy(), keep_idx.copy()))
        return np.array(
            [
                [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[4.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )

    monkeypatch.setattr(warp_md, "closest_gather_array", fake_native, raising=False)
    fake_native.__name__ = "closest_gather_array"
    coords = np.arange(18, dtype=np.float64).reshape(2, 3, 3)
    keep_idx = np.array([[2, 0], [1, 2]], dtype=np.int64)

    out = _closest_gather(coords, keep_idx)

    assert len(calls) == 1
    assert calls[0][0].dtype == np.float32
    assert calls[0][1].dtype == np.int64
    assert out.dtype == np.float32
    assert out.shape == (2, 2, 3)


def test_closest_native_frame_indices_without_python_read(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    time = np.array([10.0, 11.0, 12.0], dtype=np.float32)
    box = np.array(
        [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
        dtype=np.float32,
    )
    atoms = {
        "name": ["CA", "OW", "HW"],
        "resname": ["ALA", "SOL", "SOL"],
        "resid": [1, 2, 3],
        "chain": ["A", "A", "A"],
        "element": ["C", "O", "H"],
        "mass": [1.0, 16.0, 1.0],
    }
    traj = warp_md.Trajectory.from_numpy(coords, box=box, time_ps=time)
    system = warp_md.System.from_arrays(atoms, positions0=coords[0])

    def _fail_read_all(*_args, **_kwargs):
        raise AssertionError("closest should not materialize native trajectory frames")

    monkeypatch.setattr(closest_mod, "read_all_frames", _fail_read_all)
    out = closest_mod.closest(
        traj,
        system,
        mask="resid 1",
        solvent_mask="resid 2:3",
        n_solvents=1,
        frame_indices=[0, 2],
    )
    chunk = out.read_chunk()
    np.testing.assert_allclose(
        chunk["coords"],
        np.array([[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]], dtype=np.float32),
    )
    np.testing.assert_allclose(chunk["box"], box[[0, 2]])
    np.testing.assert_allclose(chunk["time"], np.array([10.0, 12.0], dtype=np.float32))
