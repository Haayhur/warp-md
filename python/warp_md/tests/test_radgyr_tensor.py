import numpy as np

import warp_md.analysis.structure as structure_mod
from warp_md.analysis.structure import radgyr, radgyr_tensor
from warp_md.analysis.trajectory import ArrayTrajectory


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms):
        self._atoms = {
            "name": ["CA"] * n_atoms,
            "resname": ["ALA"] * n_atoms,
            "resid": [1] * n_atoms,
            "chain_id": [0] * n_atoms,
            "mass": [1.0] * n_atoms,
        }

    def atom_table(self):
        return self._atoms

    def select(self, _mask):
        return _DummySelection(range(len(self._atoms["name"])))


def test_radgyr_tensor_basic():
    coords = np.array(
        [
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = ArrayTrajectory(coords)
    system = _DummySystem(coords.shape[1])

    rg, tensor = radgyr_tensor(traj, system, mask="all")
    np.testing.assert_allclose(rg, np.array([1.0, 2.0], dtype=np.float32), rtol=1e-6)
    expected = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(tensor, expected, rtol=1e-6)

    traj.reset()
    out = radgyr_tensor(traj, system, mask="all", frame_indices=[1], dtype="dict")
    assert set(out.keys()) == {"rg", "tensor"}
    np.testing.assert_allclose(out["rg"], np.array([2.0], dtype=np.float32), rtol=1e-6)


def test_radgyr_uses_live_native_path(monkeypatch):
    traj = object()
    system = _DummySystem(2)
    seen = {}

    class _DummyPlan:
        def __init__(
            self,
            selection,
            mass_weighted=True,
            include_max=False,
            include_tensor=False,
        ):
            seen["selection"] = list(selection.indices)
            seen["mass_weighted"] = mass_weighted
            seen["include_max"] = include_max
            seen["include_tensor"] = include_tensor

        def run(self, got_traj, got_system, chunk_frames=None, device="auto", frame_indices=None):
            seen["traj"] = got_traj
            seen["system"] = got_system
            seen["chunk_frames"] = chunk_frames
            seen["device"] = device
            seen["frame_indices"] = frame_indices
            return np.array([[1.0, 2.0]], dtype=np.float32)

    monkeypatch.setattr(structure_mod, "is_native_traj", lambda got: got is traj)
    monkeypatch.setattr(structure_mod, "coerce_native_system", lambda got: got)
    monkeypatch.setattr(structure_mod, "load_native_symbol", lambda _name: _DummyPlan)
    monkeypatch.setattr(
        structure_mod,
        "read_all_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected frame read")),
    )

    out = radgyr(
        traj,
        system,
        mask="all",
        mass=False,
        nomax=False,
        frame_indices=[0, -1],
        chunk_frames=5,
    )
    np.testing.assert_allclose(out, np.array([[1.0, 2.0]], dtype=np.float32))
    assert seen == {
        "selection": [0, 1],
        "mass_weighted": False,
        "include_max": True,
        "include_tensor": False,
        "traj": traj,
        "system": system,
        "chunk_frames": 5,
        "device": "auto",
        "frame_indices": [0, -1],
    }


def test_radgyr_tensor_uses_live_native_path(monkeypatch):
    traj = object()
    system = _DummySystem(2)
    seen = {}

    class _DummyPlan:
        def __init__(self, selection, mass_weighted=False):
            seen["selection"] = list(selection.indices)
            seen["mass_weighted"] = mass_weighted

        def run(self, got_traj, got_system, chunk_frames=None, device="auto", frame_indices=None):
            seen["traj"] = got_traj
            seen["system"] = got_system
            seen["frame_indices"] = frame_indices
            return np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(structure_mod, "is_native_traj", lambda got: got is traj)
    monkeypatch.setattr(structure_mod, "coerce_native_system", lambda got: got)
    monkeypatch.setattr(structure_mod, "load_native_symbol", lambda _name: _DummyPlan)
    monkeypatch.setattr(
        structure_mod,
        "read_all_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected frame read")),
    )

    out = radgyr_tensor(traj, system, mask="all", mass=True, frame_indices=[1], dtype="dict")
    np.testing.assert_allclose(out["rg"], np.array([1.0], dtype=np.float32))
    np.testing.assert_allclose(out["tensor"], np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    assert seen == {
        "selection": [0, 1],
        "mass_weighted": True,
        "traj": traj,
        "system": system,
        "frame_indices": [1],
    }
