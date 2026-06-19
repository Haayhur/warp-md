import importlib

import numpy as np

import warp_md
from warp_md.analysis.native_contacts import native_contacts

native_contacts_mod = importlib.import_module("warp_md.analysis.native_contacts")


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
    def __init__(self, coords, box=None):
        self._coords = coords
        self._box = box
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        chunk = {"coords": self._coords}
        if self._box is not None:
            chunk["box"] = self._box
        return chunk


def test_native_contacts_basic():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    out = native_contacts(traj, system, mask="all", ref=0, distance=1.5)
    np.testing.assert_allclose(out, np.array([1.0, 0.0], dtype=np.float32), rtol=1e-6)


def test_native_contacts_native_frame_indices_without_python_read(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(_DummySystem(coords.shape[1]).atom_table(), positions0=coords[0])
    traj = warp_md.Trajectory.from_numpy(coords)

    def _fail_read_all(*_args, **_kwargs):
        raise AssertionError("native path should not materialize all coordinates in Python")

    monkeypatch.setattr(native_contacts_mod, "read_all_frames", _fail_read_all)

    out = native_contacts(traj, system, mask="all", ref=0, distance=1.5, frame_indices=[0, 2])

    np.testing.assert_allclose(out, np.array([1.0, 1.0], dtype=np.float32), rtol=1e-6)


def test_native_contacts_mask2_native_without_python_read(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [4.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    atoms = {
        "name": ["CA", "CB", "CG"],
        "resname": ["ALA", "ALA", "ALA"],
        "resid": [1, 2, 3],
        "chain_id": [0, 0, 0],
        "mass": [1.0, 1.0, 1.0],
    }
    system = warp_md.System.from_arrays(atoms, positions0=coords[0])
    traj = warp_md.Trajectory.from_numpy(coords)

    def _fail_read_all(*_args, **_kwargs):
        raise AssertionError("native mask2 path should not materialize all coordinates in Python")

    monkeypatch.setattr(native_contacts_mod, "read_all_frames", _fail_read_all)

    out = native_contacts(
        traj,
        system,
        mask="resid 1",
        mask2="resid 2:3",
        ref=0,
        distance=1.5,
        frame_indices=[0, 2],
    )

    np.testing.assert_allclose(out, np.array([1.0, 1.0], dtype=np.float32), rtol=1e-6)


def test_native_contacts_mindist_native_without_python_read(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    atoms = {
        "name": ["CA", "CB"],
        "resname": ["ALA", "ALA"],
        "resid": [1, 2],
        "chain_id": [0, 0],
        "mass": [1.0, 1.0],
    }
    system = warp_md.System.from_arrays(atoms, positions0=coords[0])
    traj = warp_md.Trajectory.from_numpy(coords)

    def _fail_read_all(*_args, **_kwargs):
        raise AssertionError("native mindist path should not materialize all coordinates in Python")

    monkeypatch.setattr(native_contacts_mod, "read_all_frames", _fail_read_all)

    out = native_contacts(
        traj,
        system,
        mask="resid 1",
        mask2="resid 2",
        ref=0,
        distance=1.5,
        mindist=0.5,
        frame_indices=[0, 1, 2],
    )

    np.testing.assert_allclose(out, np.array([1.0, 0.0, 1.0], dtype=np.float32), rtol=1e-6)


def test_native_contacts_topology_reference_native_without_python_read(monkeypatch):
    topology_coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(
        _DummySystem(coords.shape[1]).atom_table(),
        positions0=topology_coords,
    )
    traj = warp_md.Trajectory.from_numpy(coords)

    def _fail_read_all(*_args, **_kwargs):
        raise AssertionError("native topology reference path should not materialize all coordinates in Python")

    monkeypatch.setattr(native_contacts_mod, "read_all_frames", _fail_read_all)

    out = native_contacts(
        traj,
        system,
        mask="all",
        ref="topology",
        distance=1.5,
        image=False,
        frame_indices=[0, 2],
    )

    np.testing.assert_allclose(out, np.array([0.0, 1.0], dtype=np.float32), rtol=1e-6)


def test_native_contacts_nonzero_reference_native_without_python_read(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(_DummySystem(coords.shape[1]).atom_table(), positions0=coords[0])
    traj = warp_md.Trajectory.from_numpy(coords)

    def _fail_read_all(*_args, **_kwargs):
        raise AssertionError("native nonzero reference path should not materialize all coordinates in Python")

    monkeypatch.setattr(native_contacts_mod, "read_all_frames", _fail_read_all)

    out = native_contacts(
        traj,
        system,
        mask="all",
        ref=1,
        distance=1.5,
        image=False,
        chunk_frames=1,
    )

    np.testing.assert_allclose(out, np.array([0.0, 1.0, 1.0], dtype=np.float32), rtol=1e-6)
