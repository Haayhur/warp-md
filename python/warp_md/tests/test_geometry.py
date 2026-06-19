import numpy as np
import pytest

import warp_md
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
    traj = warp_md.Trajectory.from_numpy(coords)
    system = warp_md.System.from_arrays(_DummySystem(coords.shape[1]).atom_table())
    out = angle(traj, system, "@1 @2 @3")
    np.testing.assert_allclose(out, np.array([90.0, 90.0], dtype=np.float32), rtol=1e-6)

    traj_subset = warp_md.Trajectory.from_numpy(coords)
    out_subset = angle(traj_subset, system, "@1 @2 @3", frame_indices=[1])
    np.testing.assert_allclose(out_subset, np.array([90.0], dtype=np.float32), rtol=1e-6)

    traj2 = warp_md.Trajectory.from_numpy(coords)
    out2 = angle(traj2, system, np.array([[0, 1, 2]], dtype=np.int64))
    np.testing.assert_allclose(out2[0], np.array([90.0, 90.0], dtype=np.float32), rtol=1e-6)


def test_angle_index_array_uses_native_multi_plan(monkeypatch):
    traj = object()
    system = _DummySystem(4)
    seen = {}

    class _DummyPlan:
        def __init__(self, sel_a, sel_b, sel_c, mass_weighted=False, pbc="none", degrees=True):
            seen["sel_a"] = sel_a
            seen["sel_b"] = sel_b
            seen["sel_c"] = sel_c
            seen["mass_weighted"] = mass_weighted
            seen["pbc"] = pbc
            seen["degrees"] = degrees

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

    out = angle(
        traj,
        system,
        np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        mass=True,
        frame_indices=[0, 2],
        chunk_frames=5,
    )

    np.testing.assert_allclose(
        out,
        np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float32),
    )
    assert seen["plan_name"] == "MultiAnglePlan"
    assert seen["sel_a"] == [[0], [1]]
    assert seen["sel_b"] == [[1], [2]]
    assert seen["sel_c"] == [[2], [3]]
    assert seen["mass_weighted"] is True
    assert seen["pbc"] == "none"
    assert seen["degrees"] is True
    assert seen["traj"] is traj
    assert seen["system"] is system
    assert seen["chunk_frames"] == 5
    assert seen["device"] == "auto"
    assert seen["frame_indices"] == [0, 2]


def test_dihedral_basic():
    coords = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = warp_md.Trajectory.from_numpy(coords)
    system = warp_md.System.from_arrays(_DummySystem(coords.shape[1]).atom_table())
    out = dihedral(traj, system, "@1 @2 @3 @4")
    np.testing.assert_allclose(out, np.array([180.0], dtype=np.float32), rtol=1e-6, atol=1e-6)

    traj_subset = warp_md.Trajectory.from_numpy(coords)
    out_subset = dihedral(traj_subset, system, "@1 @2 @3 @4", frame_indices=[0])
    np.testing.assert_allclose(out_subset, np.array([180.0], dtype=np.float32), rtol=1e-6, atol=1e-6)

    traj2 = warp_md.Trajectory.from_numpy(coords)
    out2 = dihedral(traj2, system, np.array([[0, 1, 2, 3]], dtype=np.int64), range360=True)
    np.testing.assert_allclose(out2[0], np.array([180.0], dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_dihedral_index_array_uses_native_multi_plan(monkeypatch):
    traj = object()
    system = _DummySystem(5)
    seen = {}

    class _DummyPlan:
        def __init__(
            self,
            groups,
            mass_weighted=False,
            pbc="none",
            degrees=True,
            range360=False,
        ):
            seen["groups"] = [
                tuple(list(selection.indices) for selection in group) for group in groups
            ]
            seen["mass_weighted"] = mass_weighted
            seen["pbc"] = pbc
            seen["degrees"] = degrees
            seen["range360"] = range360

        def run(self, got_traj, got_system, chunk_frames=None, device="auto", frame_indices=None):
            seen["traj"] = got_traj
            seen["system"] = got_system
            seen["chunk_frames"] = chunk_frames
            seen["device"] = device
            seen["frame_indices"] = frame_indices
            return np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

    def _load(name):
        seen["plan_name"] = name
        return _DummyPlan

    monkeypatch.setattr(geometry_mod, "is_native_traj", lambda got: got is traj)
    monkeypatch.setattr(geometry_mod, "coerce_native_system", lambda got: got)
    monkeypatch.setattr(geometry_mod, "load_native_symbol", _load)

    out = dihedral(
        traj,
        system,
        np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64),
        mass=True,
        range360=True,
        frame_indices=[0, 1],
        chunk_frames=6,
    )

    np.testing.assert_allclose(
        out,
        np.array([[10.0, 30.0], [20.0, 40.0]], dtype=np.float32),
    )
    assert seen["plan_name"] == "MultiDihedralPlan"
    assert seen["groups"] == [
        ([0], [1], [2], [3]),
        ([1], [2], [3], [4]),
    ]
    assert seen["mass_weighted"] is True
    assert seen["pbc"] == "none"
    assert seen["degrees"] is True
    assert seen["range360"] is True
    assert seen["traj"] is traj
    assert seen["system"] is system
    assert seen["chunk_frames"] == 6
    assert seen["device"] == "auto"
    assert seen["frame_indices"] == [0, 1]


def test_angle_and_dihedral_coerce_read_chunk_trajectory():
    angle_coords = np.array(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        dtype=np.float32,
    )
    angle_out = angle(_DummyTraj(angle_coords), _DummySystem(3), "@1 @2 @3")
    np.testing.assert_allclose(angle_out, np.array([90.0], dtype=np.float32), atol=1e-6)

    dihedral_coords = np.array(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]],
        dtype=np.float32,
    )
    dihedral_out = dihedral(_DummyTraj(dihedral_coords), _DummySystem(4), "@1 @2 @3 @4")
    np.testing.assert_allclose(
        dihedral_out,
        np.array([180.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


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


def test_distance_pair_list_stats_dtype():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(_DummySystem(coords.shape[1]).atom_table())
    out = distance(
        warp_md.Trajectory.from_numpy(coords),
        system,
        np.array([[0, 1], [1, 2]], dtype=np.int64),
        dtype="stats",
    )

    np.testing.assert_allclose(out["mean"], np.array([1.5, 3.0], dtype=np.float32))
    np.testing.assert_allclose(out["std"], np.array([0.5, 1.0], dtype=np.float32))
    np.testing.assert_allclose(out["min"], np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(out["max"], np.array([2.0, 4.0], dtype=np.float32))
    assert out["n_frames"] == 2


def test_distance_adjacent_pair_mode_runs_through_pair_list_plan(monkeypatch):
    traj = object()
    system = _DummySystem(4)
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
            return np.array([[1.0, 4.0], [2.0, 5.0]], dtype=np.float32)

    def _load(name):
        seen["plan_name"] = name
        return _DummyPlan

    monkeypatch.setattr(geometry_mod, "is_native_traj", lambda got: got is traj)
    monkeypatch.setattr(geometry_mod, "coerce_native_system", lambda got: got)
    monkeypatch.setattr(geometry_mod, "load_native_symbol", _load)

    out = distance(
        traj,
        system,
        "@1 @2 @3 @4",
        pair_mode="adjacent",
        image=True,
        frame_indices=[1],
        chunk_frames=3,
    )

    np.testing.assert_allclose(
        out,
        np.array([[1.0, 2.0], [4.0, 5.0]], dtype=np.float32),
    )
    assert seen["plan_name"] == "PairListDistancePlan"
    np.testing.assert_array_equal(seen["pairs"], np.array([[0, 1], [2, 3]]))
    assert seen["pbc"] == "orthorhombic"
    assert seen["traj"] is traj
    assert seen["system"] is system
    assert seen["chunk_frames"] == 3
    assert seen["device"] == "auto"
    assert seen["frame_indices"] == [1]


def test_distance_adjacent_pair_mode_numeric():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(_DummySystem(coords.shape[1]).atom_table())
    out = distance(
        warp_md.Trajectory.from_numpy(coords),
        system,
        "@1 @2 @3 @4",
        pair_mode="adjacent",
    )
    expected = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_distance_adjacent_pair_mode_rejects_odd_selection():
    coords = np.zeros((1, 3, 3), dtype=np.float32)
    system = warp_md.System.from_arrays(_DummySystem(coords.shape[1]).atom_table())
    with pytest.raises(ValueError, match="even number"):
        distance(
            warp_md.Trajectory.from_numpy(coords),
            system,
            "@1 @2 @3",
            pair_mode="adjacent",
        )


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


def test_distance_components_use_native_vector_plan(monkeypatch):
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
            seen["chunk_frames"] = chunk_frames
            seen["device"] = device
            seen["frame_indices"] = frame_indices
            return np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    def _load(name):
        seen["plan_name"] = name
        return _DummyPlan

    monkeypatch.setattr(geometry_mod, "is_native_traj", lambda got: got is traj)
    monkeypatch.setattr(geometry_mod, "coerce_native_system", lambda got: got)
    monkeypatch.setattr(geometry_mod, "load_native_symbol", _load)

    out = distance(
        traj,
        system,
        "@1 @2",
        image=True,
        mass=False,
        components=True,
        frame_indices=[0],
        chunk_frames=5,
        dtype="dict",
    )

    assert seen["plan_name"] == "DistanceVectorPlan"
    assert seen["sel_a"] == [0]
    assert seen["sel_b"] == [1]
    assert seen["mass_weighted"] is False
    assert seen["pbc"] == "orthorhombic"
    assert seen["traj"] is traj
    assert seen["system"] is system
    assert seen["chunk_frames"] == 5
    assert seen["device"] == "auto"
    assert seen["frame_indices"] == [0]
    np.testing.assert_allclose(
        out["components"],
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
    )


def test_distance_components_numeric_with_image():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    box = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
    system = warp_md.System.from_arrays(_DummySystem(coords.shape[1]).atom_table())

    no_image = distance(
        warp_md.Trajectory.from_numpy(coords, box=box),
        system,
        "@1 @2",
        mass=False,
        image=False,
        components=True,
    )
    with_image = distance(
        warp_md.Trajectory.from_numpy(coords, box=box),
        system,
        "@1 @2",
        mass=False,
        image=True,
        components=True,
    )

    np.testing.assert_allclose(no_image, np.array([[9.0, 0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(with_image, np.array([[-1.0, 0.0, 0.0]], dtype=np.float32))


def test_distance_components_stats_dtype():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
            [[0.0, 0.0, 0.0], [3.0, 4.0, 5.0]],
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(_DummySystem(coords.shape[1]).atom_table())
    out = distance(
        warp_md.Trajectory.from_numpy(coords),
        system,
        "@1 @2",
        mass=False,
        components=True,
        dtype="stats",
    )

    np.testing.assert_allclose(out["mean"], np.array([2.0, 3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(out["std"], np.array([1.0, 1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(out["min"], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(out["max"], np.array([3.0, 4.0, 5.0], dtype=np.float32))
    assert out["n_frames"] == 2


def test_distance_components_reject_unsupported_inputs():
    coords = np.zeros((1, 2, 3), dtype=np.float32)
    system = warp_md.System.from_arrays(_DummySystem(coords.shape[1]).atom_table())
    with pytest.raises(ValueError, match="only supported"):
        distance(
            warp_md.Trajectory.from_numpy(coords),
            system,
            np.array([[0, 1]], dtype=np.int64),
            components=True,
        )


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


def test_distance_rejects_non_native_trajectory_without_frame_read():
    class _NoReadTraj:
        def read_chunk(self, *_args, **_kwargs):
            raise AssertionError("unexpected frame read")

    with pytest.raises(RuntimeError, match="Rust-backed trajectory"):
        distance(_NoReadTraj(), _DummySystem(2), np.array([[0, 1]], dtype=np.int64))


def test_distance_reference_frame_array_still_uses_rust_point_plan(monkeypatch):
    traj = object()
    system = _DummySystem(3)
    seen = {}

    class _DummyPlan:
        def __init__(self, selection, point, mass_weighted=True, pbc="none"):
            seen["selection"] = list(selection.indices)
            seen["point"] = tuple(point)
            seen["mass_weighted"] = mass_weighted
            seen["pbc"] = pbc

        def run(self, got_traj, got_system, chunk_frames=None, device="auto", frame_indices=None):
            seen["traj"] = got_traj
            seen["system"] = got_system
            seen["frame_indices"] = frame_indices
            return np.array([7.0], dtype=np.float32)

    def _load(name):
        seen["plan_name"] = name
        return _DummyPlan

    monkeypatch.setattr(geometry_mod, "is_native_traj", lambda got: got is traj)
    monkeypatch.setattr(geometry_mod, "coerce_native_system", lambda got: got)
    monkeypatch.setattr(geometry_mod, "load_native_symbol", _load)

    ref = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [99.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    out = distance(
        traj,
        system,
        "@1 @2",
        ref=ref,
        mass=False,
        frame_indices=[0],
    )

    np.testing.assert_allclose(out, np.array([7.0], dtype=np.float32))
    assert seen["plan_name"] == "DistanceCenterToPointPlan"
    assert seen["selection"] == [0, 1]
    assert seen["point"] == (1.0, 0.0, 0.0)
    assert seen["mass_weighted"] is False
    assert seen["pbc"] == "none"
    assert seen["traj"] is traj
    assert seen["system"] is system
    assert seen["frame_indices"] == [0]
