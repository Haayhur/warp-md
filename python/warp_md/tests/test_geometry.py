import numpy as np

import warp_md.analysis.geometry as geometry_mod
from warp_md.analysis.geometry import angle, dihedral, distance


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

    def select_indices(self, indices):
        return _DummySelection(indices)


class _DummyTraj:
    def __init__(self, coords):
        self._coords = coords
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        return {"coords": self._coords}


def test_angle_basic():
    coords = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    out = angle(traj, system, "@1 @2 @3")
    np.testing.assert_allclose(out, np.array([90.0, 90.0], dtype=np.float32), rtol=1e-6)

    traj2 = _DummyTraj(coords)
    out2 = angle(traj2, system, np.array([[0, 1, 2]], dtype=np.int64))
    np.testing.assert_allclose(out2[0], np.array([90.0, 90.0], dtype=np.float32), rtol=1e-6)


def test_dihedral_basic():
    coords = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    out = dihedral(traj, system, "@1 @2 @3 @4")
    np.testing.assert_allclose(out, np.array([180.0], dtype=np.float32), rtol=1e-6, atol=1e-6)

    traj2 = _DummyTraj(coords)
    out2 = dihedral(traj2, system, np.array([[0, 1, 2, 3]], dtype=np.int64), range360=True)
    np.testing.assert_allclose(out2[0], np.array([180.0], dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_distance_pair_list_uses_native_streaming_path(monkeypatch):
    traj = object()
    system = _DummySystem(3)
    seen = {}

    class _DummyPlan:
        def __init__(self, pairs, pbc="none"):
            seen["pairs"] = np.asarray(pairs, dtype=np.int64)
            seen["pbc"] = pbc

        def run(self, got_traj, got_system, chunk_frames=None, device="auto", frame_indices=None):
            seen["traj"] = got_traj
            seen["system"] = got_system
            seen["chunk_frames"] = chunk_frames
            seen["device"] = device
            seen["frame_indices"] = frame_indices
            return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    monkeypatch.setattr(geometry_mod, "is_native_traj", lambda got: got is traj)
    monkeypatch.setattr(geometry_mod, "coerce_native_system", lambda got: got)
    monkeypatch.setattr(geometry_mod, "load_native_symbol", lambda name: _DummyPlan)
    monkeypatch.setattr(
        geometry_mod,
        "read_all_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected frame read")),
    )

    out = distance(
        traj,
        system,
        np.array([[0, 1], [1, 2]], dtype=np.int64),
        frame_indices=[0, -1],
        chunk_frames=7,
    )
    np.testing.assert_allclose(
        out,
        np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(seen["pairs"], np.array([[0, 1], [1, 2]]))
    assert seen["pbc"] == "none"
    assert seen["traj"] is traj
    assert seen["system"] is system
    assert seen["chunk_frames"] == 7
    assert seen["device"] == "auto"
    assert seen["frame_indices"] == [0, -1]


def test_distance_mask_uses_native_streaming_path(monkeypatch):
    traj = object()
    system = _DummySystem(3)
    seen = {}

    class _DummyPlan:
        def __init__(self, sel_a, sel_b, mass_weighted=True, pbc="none"):
            seen["sel_a"] = list(sel_a.indices)
            seen["sel_b"] = list(sel_b.indices)
            seen["mass_weighted"] = mass_weighted
            seen["pbc"] = pbc

        def run(self, got_traj, got_system, chunk_frames=None, device="auto", frame_indices=None):
            seen["traj"] = got_traj
            seen["system"] = got_system
            seen["frame_indices"] = frame_indices
            return np.array([5.0, 6.0], dtype=np.float32)

    monkeypatch.setattr(geometry_mod, "is_native_traj", lambda got: got is traj)
    monkeypatch.setattr(geometry_mod, "coerce_native_system", lambda got: got)
    monkeypatch.setattr(geometry_mod, "load_native_symbol", lambda name: _DummyPlan)
    monkeypatch.setattr(
        geometry_mod,
        "read_all_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected frame read")),
    )

    out = distance(
        traj,
        system,
        "@1 @2",
        image=True,
        mass=False,
        frame_indices=[1],
    )
    np.testing.assert_allclose(out, np.array([5.0, 6.0], dtype=np.float32))
    assert seen["sel_a"] == [0]
    assert seen["sel_b"] == [1]
    assert seen["mass_weighted"] is False
    assert seen["pbc"] == "orthorhombic"
    assert seen["traj"] is traj
    assert seen["system"] is system
    assert seen["frame_indices"] == [1]


def test_distance_command_list_uses_native_multi_plan(monkeypatch):
    traj = object()
    system = _DummySystem(3)
    seen = {}

    class _DummyPlan:
        def __init__(self, sel_a, sel_b, mass_weighted=True, pbc="none"):
            seen["sel_a"] = sel_a
            seen["sel_b"] = sel_b
            seen["mass_weighted"] = mass_weighted
            seen["pbc"] = pbc

        def run(self, got_traj, got_system, chunk_frames=None, device="auto", frame_indices=None):
            seen["traj"] = got_traj
            seen["system"] = got_system
            seen["chunk_frames"] = chunk_frames
            seen["device"] = device
            seen["frame_indices"] = frame_indices
            return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    def _load(name):
        seen["plan_name"] = name
        return _DummyPlan

    monkeypatch.setattr(geometry_mod, "is_native_traj", lambda got: got is traj)
    monkeypatch.setattr(geometry_mod, "coerce_native_system", lambda got: got)
    monkeypatch.setattr(geometry_mod, "load_native_symbol", _load)
    monkeypatch.setattr(
        geometry_mod,
        "read_all_frames",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected frame read")),
    )

    out = distance(
        traj,
        system,
        ["@1 @2", "@2 @3"],
        image=True,
        mass=False,
        frame_indices=[0, 2],
        chunk_frames=11,
    )

    np.testing.assert_allclose(
        out,
        np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float32),
    )
    assert seen["plan_name"] == "MultiDistancePlan"
    assert seen["sel_a"] == [[0], [1]]
    assert seen["sel_b"] == [[1], [2]]
    assert seen["mass_weighted"] is False
    assert seen["pbc"] == "orthorhombic"
    assert seen["traj"] is traj
    assert seen["system"] is system
    assert seen["chunk_frames"] == 11
    assert seen["device"] == "auto"
    assert seen["frame_indices"] == [0, 2]
