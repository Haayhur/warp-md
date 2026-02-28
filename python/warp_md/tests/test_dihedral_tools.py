import numpy as np
import pytest
import warp_md

from warp_md.analysis.dihedral_tools import rotate_dihedral, set_dihedral


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
        return _DummySelection(list(range(len(self._atoms["name"]))))

    def select_indices(self, indices):
        return _DummySelection(list(indices))

    def n_atoms(self):
        return len(self._atoms["name"])


class _DummyTraj:
    def __init__(self, coords):
        self._coords = coords
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        return {"coords": self._coords}


def test_rotate_dihedral_uses_plan(monkeypatch):
    coords = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
            [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 1.0]],
        ],
        dtype=np.float32,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(self, _a, _b, _c, _d, _rot, angle, mass, degrees):
            called["angle"] = angle
            called["mass"] = mass
            called["degrees"] = degrees

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["frame_indices"] = frame_indices
            if frame_indices is None:
                return coords.reshape(-1)
            selected = coords[np.asarray(frame_indices, dtype=np.int64)]
            return selected.reshape(-1)

    monkeypatch.setattr(warp_md, "RotateDihedralPlan", _DummyPlan, raising=False)
    out = rotate_dihedral(
        traj,
        system,
        atoms=[0, 1, 2, 3],
        rotate_mask=[3],
        angle=90.0,
        frame_indices=[1],
    )
    got = out.read_chunk()["coords"]
    assert called["angle"] == 90.0
    assert called["mass"] is False
    assert called["degrees"] is True
    assert called["frame_indices"] == [1]
    assert got.shape == (1, 4, 3)
    np.testing.assert_allclose(got[0], coords[1], atol=1e-6)


def test_set_dihedral_uses_plan(monkeypatch):
    coords = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(self, _a, _b, _c, _d, _rot, target, mass, pbc, degrees, range360):
            called["target"] = target
            called["pbc"] = pbc
            called["degrees"] = degrees
            called["range360"] = range360
            called["mass"] = mass

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["frame_indices"] = frame_indices
            return coords.reshape(-1)

    monkeypatch.setattr(warp_md, "SetDihedralPlan", _DummyPlan, raising=False)
    out = set_dihedral(
        traj,
        system,
        atoms=[0, 1, 2, 3],
        rotate_mask=[3],
        target=0.0,
        pbc="none",
        degrees=True,
        range360=False,
    )
    got = out.read_chunk()["coords"]
    assert called["target"] == 0.0
    assert called["pbc"] == "none"
    assert called["degrees"] is True
    assert called["range360"] is False
    assert called["mass"] is False
    assert called["frame_indices"] is None
    assert got.shape == (1, 4, 3)
    np.testing.assert_allclose(got[0], coords[0], atol=1e-6)


def test_rotate_dihedral_no_python_fallback_when_plan_missing(monkeypatch):
    traj = _DummyTraj(np.zeros((1, 4, 3), dtype=np.float32))
    system = _DummySystem(4)

    class _MissingPlan:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("RotateDihedralPlan binding unavailable in this build.")

    monkeypatch.setattr(warp_md, "RotateDihedralPlan", _MissingPlan, raising=False)
    with pytest.raises(RuntimeError, match="RotateDihedralPlan binding unavailable"):
        rotate_dihedral(traj, system, atoms=[0, 1, 2, 3], rotate_mask=[3], angle=30.0)


def test_rotate_dihedral_rejects_non_bool_flags(monkeypatch):
    coords = np.zeros((1, 4, 3), dtype=np.float32)
    traj = _DummyTraj(coords)
    system = _DummySystem(4)

    class _DummyPlan:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return coords.reshape(-1)

    monkeypatch.setattr(warp_md, "RotateDihedralPlan", _DummyPlan, raising=False)
    with pytest.raises(ValueError, match="mass must be bool"):
        rotate_dihedral(traj, system, atoms=[0, 1, 2, 3], mass=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="degrees must be bool"):
        rotate_dihedral(traj, system, atoms=[0, 1, 2, 3], degrees="yes")  # type: ignore[arg-type]


def test_set_dihedral_rejects_invalid_contract_values(monkeypatch):
    coords = np.zeros((1, 4, 3), dtype=np.float32)
    traj = _DummyTraj(coords)
    system = _DummySystem(4)

    class _DummyPlan:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return coords.reshape(-1)

    monkeypatch.setattr(warp_md, "SetDihedralPlan", _DummyPlan, raising=False)
    with pytest.raises(ValueError, match="pbc must be 'none' or 'orthorhombic'"):
        set_dihedral(traj, system, atoms=[0, 1, 2, 3], pbc="triclinic")
    with pytest.raises(ValueError, match="range360 must be bool"):
        set_dihedral(traj, system, atoms=[0, 1, 2, 3], range360=1)  # type: ignore[arg-type]
