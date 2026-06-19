import numpy as np
import pytest
import warp_md

import warp_md.analysis.structure as structure_mod
from warp_md.analysis.structure import gyrate, radgyr, radgyr_tensor
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
    system = warp_md.System.from_arrays(
        _DummySystem(coords.shape[1]).atom_table(),
        positions0=coords[0],
    )

    rg, tensor = radgyr_tensor(warp_md.Trajectory.from_numpy(coords), system, mask="all")
    np.testing.assert_allclose(rg, np.array([1.0, 2.0], dtype=np.float32), rtol=1e-6)
    expected = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(tensor, expected, rtol=1e-6)

    out = radgyr_tensor(
        warp_md.Trajectory.from_numpy(coords),
        system,
        mask="all",
        frame_indices=[1],
        dtype="dict",
    )
    assert set(out.keys()) == {"rg", "tensor"}
    np.testing.assert_allclose(out["rg"], np.array([2.0], dtype=np.float32), rtol=1e-6)


def test_radgyr_rejects_non_native_without_reading(monkeypatch):
    coords = np.array(
        [
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    system = _DummySystem(coords.shape[1])
    monkeypatch.setattr(
        structure_mod,
        "read_all_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected frame read")
        ),
    )

    with pytest.raises(RuntimeError, match="Rust-backed trajectory"):
        radgyr(ArrayTrajectory(coords), system, mask="all")
    with pytest.raises(RuntimeError, match="Rust-backed trajectory"):
        radgyr_tensor(ArrayTrajectory(coords), system, mask="all")


def test_radgyr_rejects_length_scale_on_native_path():
    coords = np.array(
        [
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(
        _DummySystem(coords.shape[1]).atom_table(),
        positions0=coords[0],
    )

    with pytest.raises(ValueError, match="length_scale must be 1.0"):
        radgyr(
            warp_md.Trajectory.from_numpy(coords),
            system,
            mask="all",
            length_scale=10.0,
        )
    with pytest.raises(ValueError, match="length_scale must be 1.0"):
        radgyr_tensor(
            warp_md.Trajectory.from_numpy(coords),
            system,
            mask="all",
            length_scale=10.0,
        )


def test_radgyr_accepts_atom_index_mask():
    coords = np.array(
        [
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
            [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(
        _DummySystem(coords.shape[1]).atom_table(),
        positions0=coords[0],
    )

    out = radgyr(
        warp_md.Trajectory.from_numpy(coords),
        system,
        mask="@1 @2",
        mass=False,
    )
    np.testing.assert_allclose(out, np.array([1.0, 2.0], dtype=np.float32), rtol=1e-6)


def test_radgyr_reports_axis_radii():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(
        _DummySystem(coords.shape[1]).atom_table(),
        positions0=coords[0],
    )

    out = radgyr(
        warp_md.Trajectory.from_numpy(coords),
        system,
        mask="all",
        mass=False,
        axes=True,
        dtype="dict",
    )

    np.testing.assert_allclose(out["rg"], np.array([1.0, 2.0], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(
        out["axes"],
        np.array(
            [
                [0.0, 1.0, 1.0],
                [2.0, 0.0, 2.0],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_gyrate_defaults_to_mass_weighted_axis_output():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
        ],
        dtype=np.float32,
    )
    atoms = _DummySystem(coords.shape[1]).atom_table()
    atoms["mass"] = [1.0, 3.0]
    system = warp_md.System.from_arrays(atoms, positions0=coords[0])

    out = gyrate(
        warp_md.Trajectory.from_numpy(coords),
        system,
        mask="all",
        dtype="dict",
    )

    assert set(out) == {"rg", "axes"}
    np.testing.assert_allclose(
        out["rg"],
        np.array([np.sqrt(0.75), np.sqrt(3.0)], dtype=np.float32),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        out["axes"],
        np.array(
            [
                [0.0, np.sqrt(0.75), np.sqrt(0.75)],
                [np.sqrt(3.0), 0.0, np.sqrt(3.0)],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_gyrate_uses_live_native_path(monkeypatch):
    traj = object()
    system = _DummySystem(2)
    seen = {}

    class _DummyPlan:
        def __init__(
            self,
            selection,
            mass_weighted=True,
            include_max=False,
            include_axes=False,
            include_tensor=False,
        ):
            seen["selection"] = list(selection.indices)
            seen["mass_weighted"] = mass_weighted
            seen["include_max"] = include_max
            seen["include_axes"] = include_axes
            seen["include_tensor"] = include_tensor

        def run(self, got_traj, got_system, chunk_frames=None, device="auto", frame_indices=None):
            seen["traj"] = got_traj
            seen["system"] = got_system
            seen["chunk_frames"] = chunk_frames
            seen["device"] = device
            seen["frame_indices"] = frame_indices
            return np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

    monkeypatch.setattr(structure_mod, "is_native_traj", lambda got: got is traj)
    monkeypatch.setattr(structure_mod, "coerce_native_system", lambda got: got)
    monkeypatch.setattr(structure_mod, "load_native_symbol", lambda _name: _DummyPlan)
    monkeypatch.setattr(
        structure_mod,
        "read_all_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected frame read")
        ),
    )

    out = gyrate(
        traj,
        system,
        mask="all",
        frame_indices=[0, -1],
        chunk_frames=5,
        dtype="dict",
        device="cpu",
    )

    np.testing.assert_allclose(out["rg"], np.array([1.0], dtype=np.float32))
    np.testing.assert_allclose(out["axes"], np.array([[2.0, 3.0, 4.0]], dtype=np.float32))
    assert seen == {
        "selection": [0, 1],
        "mass_weighted": True,
        "include_max": False,
        "include_axes": True,
        "include_tensor": False,
        "traj": traj,
        "system": system,
        "chunk_frames": 5,
        "device": "cpu",
        "frame_indices": [0, -1],
    }


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
            include_axes=False,
            include_tensor=False,
        ):
            seen["selection"] = list(selection.indices)
            seen["mass_weighted"] = mass_weighted
            seen["include_max"] = include_max
            seen["include_axes"] = include_axes
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
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected frame read")
        ),
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
        "include_axes": False,
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
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected frame read")
        ),
    )

    out = radgyr_tensor(traj, system, mask="all", mass=True, frame_indices=[1], dtype="dict")
    np.testing.assert_allclose(out["rg"], np.array([1.0], dtype=np.float32))
    np.testing.assert_allclose(
        out["tensor"],
        np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    assert seen == {
        "selection": [0, 1],
        "mass_weighted": True,
        "traj": traj,
        "system": system,
        "frame_indices": [1],
    }
