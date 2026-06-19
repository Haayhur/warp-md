import importlib

import numpy as np
import pytest

import warp_md
from warp_md.analysis.surf import molsurf, surf

surf_mod = importlib.import_module("warp_md.analysis.surf")


@pytest.fixture(autouse=True)
def _native_surface_inputs(monkeypatch):
    monkeypatch.setattr(surf_mod, "is_native_traj", lambda _traj: True)
    monkeypatch.setattr(surf_mod, "coerce_native_system", lambda system: system)


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms):
        self._n_atoms = n_atoms

    def select(self, _mask):
        return _DummySelection(range(self._n_atoms))

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


def test_surf_bbox_area(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    class _DummyPlan:
        def __init__(
            self,
            _sel,
            algorithm="bbox",
            probe_radius=1.4,
            n_sphere_points=64,
            radii=None,
            offset=0.0,
            nbrcut=2.5,
            solute_selection=None,
            radii_mode="gb",
        ):
            assert algorithm == "bbox"

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            assert frame_indices is None
            return np.array([6.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(warp_md, "SurfPlan", _DummyPlan, raising=False)
    out = surf(traj, system, mask=[0, 1], algorithm="bbox")
    np.testing.assert_allclose(out, np.array([6.0, 0.0], dtype=np.float32), atol=1e-5)


def test_molsurf_alias(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    class _DummyPlan:
        def __init__(
            self,
            _sel,
            algorithm="bbox",
            probe_radius=1.4,
            n_sphere_points=64,
            radii=None,
            offset=0.0,
            nbrcut=2.5,
            solute_selection=None,
            radii_mode="gb",
        ):
            assert algorithm == "bbox"
            assert radii_mode == "gb"

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            assert frame_indices is None
            dx, dy, dz = 1.0, 2.0, 3.0
            expected = 2.0 * (dx * dy + dy * dz + dx * dz)
            return np.array([expected], dtype=np.float32)

    monkeypatch.setattr(warp_md, "MolSurfPlan", _DummyPlan, raising=False)
    out = molsurf(traj, system, mask=[0, 1], algorithm="bbox")
    dx, dy, dz = 1.0, 2.0, 3.0
    expected = 2.0 * (dx * dy + dy * dz + dx * dz)
    np.testing.assert_allclose(out, np.array([expected], dtype=np.float32), atol=1e-5)


def test_surf_sasa_positive(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    class _DummyPlan:
        def __init__(
            self,
            _sel,
            algorithm="bbox",
            probe_radius=1.4,
            n_sphere_points=64,
            radii=None,
            offset=0.0,
            nbrcut=2.5,
            solute_selection=None,
            radii_mode="gb",
        ):
            assert algorithm == "sasa"
            assert probe_radius == 0.0
            assert n_sphere_points == 32
            assert radii == [1.0]

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            assert frame_indices is None
            expected = 4.0 * np.pi * 1.0 * 1.0
            return np.array([expected], dtype=np.float32)

    monkeypatch.setattr(warp_md, "SurfPlan", _DummyPlan, raising=False)
    out = surf(traj, system, mask=[0], algorithm="sasa", probe_radius=0.0, n_sphere_points=32, radii=[1.0])
    expected = 4.0 * np.pi * 1.0 * 1.0
    np.testing.assert_allclose(out, np.array([expected], dtype=np.float32), rtol=0.15)


def test_surf_sasa_uses_plan(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            _sel,
            algorithm="bbox",
            probe_radius=1.4,
            n_sphere_points=64,
            radii=None,
            offset=0.0,
            nbrcut=2.5,
            solute_selection=None,
            radii_mode="gb",
        ):
            called["algorithm"] = algorithm
            called["probe_radius"] = probe_radius
            called["n_sphere_points"] = n_sphere_points
            called["radii"] = radii
            called["radii_mode"] = radii_mode

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["frame_indices"] = frame_indices
            values = np.array([10.0, 20.0], dtype=np.float32)
            if frame_indices is None:
                return values
            return values[np.asarray(frame_indices, dtype=np.int64)]

    monkeypatch.setattr(warp_md, "SurfPlan", _DummyPlan, raising=False)
    out = surf(
        traj,
        system,
        mask=[0],
        algorithm="sasa",
        probe_radius=0.0,
        n_sphere_points=16,
        radii=[1.0],
        frame_indices=[1],
    )
    np.testing.assert_allclose(out, np.array([20.0], dtype=np.float32))
    assert called["algorithm"] == "sasa"
    assert called["probe_radius"] == 0.0
    assert called["n_sphere_points"] == 16
    assert called["radii_mode"] == "gb"
    assert called["frame_indices"] == [1]


def test_surf_atom_area_requests_native_detail(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            _sel,
            algorithm="bbox",
            probe_radius=1.4,
            n_sphere_points=64,
            radii=None,
            offset=0.0,
            nbrcut=2.5,
            solute_selection=None,
            radii_mode="gb",
            atom_area=False,
        ):
            called["algorithm"] = algorithm
            called["atom_area"] = atom_area

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["frame_indices"] = frame_indices
            total = np.array([3.0, 4.0], dtype=np.float32)
            atom_area = np.array([[1.0, 2.0], [1.5, 2.5]], dtype=np.float32)
            return total, atom_area, 2, 2

    monkeypatch.setattr(warp_md, "SurfPlan", _DummyPlan, raising=False)
    out = surf(traj, system, mask=[0, 1], algorithm="sasa", atom_area=True)
    assert called == {"algorithm": "sasa", "atom_area": True, "frame_indices": None}
    np.testing.assert_allclose(out["surf"], np.array([3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(
        out["atom_area"],
        np.array([[1.0, 2.0], [1.5, 2.5]], dtype=np.float32),
    )
    assert out["frames"] == 2
    assert out["atoms"] == 2


def test_surf_volume_requests_native_detail(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            _sel,
            algorithm="bbox",
            probe_radius=1.4,
            n_sphere_points=64,
            radii=None,
            offset=0.0,
            nbrcut=2.5,
            solute_selection=None,
            radii_mode="gb",
            atom_area=False,
            volume=False,
        ):
            called["algorithm"] = algorithm
            called["atom_area"] = atom_area
            called["volume"] = volume

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            total = np.array([12.0], dtype=np.float32)
            volume = np.array([4.0], dtype=np.float32)
            return total, None, volume, 1, 1

    monkeypatch.setattr(warp_md, "SurfPlan", _DummyPlan, raising=False)
    out = surf(traj, system, mask=[0], algorithm="sasa", volume=True)
    assert called == {"algorithm": "sasa", "atom_area": False, "volume": True}
    np.testing.assert_allclose(out["surf"], np.array([12.0], dtype=np.float32))
    np.testing.assert_allclose(out["volume"], np.array([4.0], dtype=np.float32))
    assert "atom_area" not in out
    assert out["frames"] == 1
    assert out["atoms"] == 1


def test_surf_residue_area_requests_native_detail(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            _sel,
            algorithm="bbox",
            probe_radius=1.4,
            n_sphere_points=64,
            radii=None,
            offset=0.0,
            nbrcut=2.5,
            solute_selection=None,
            radii_mode="gb",
            atom_area=False,
            volume=False,
            residue_area=False,
        ):
            called["algorithm"] = algorithm
            called["atom_area"] = atom_area
            called["volume"] = volume
            called["residue_area"] = residue_area

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            total = np.array([3.0, 4.0], dtype=np.float32)
            residue_area = np.array([[1.0, 2.0], [1.5, 2.5]], dtype=np.float32)
            residue_ids = np.array([1, 2], dtype=np.int32)
            return total, None, None, residue_area, residue_ids, 2, 2

    monkeypatch.setattr(warp_md, "SurfPlan", _DummyPlan, raising=False)
    out = surf(traj, system, mask=[0, 1], algorithm="sasa", residue_area=True)
    assert called == {
        "algorithm": "sasa",
        "atom_area": False,
        "volume": False,
        "residue_area": True,
    }
    np.testing.assert_allclose(out["surf"], np.array([3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(
        out["residue_area"],
        np.array([[1.0, 2.0], [1.5, 2.5]], dtype=np.float32),
    )
    np.testing.assert_array_equal(out["residue_ids"], np.array([1, 2], dtype=np.int32))
    assert out["frames"] == 2
    assert out["atoms"] == 2


def test_molsurf_uses_molsurf_plan_and_radii_mode(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            _sel,
            algorithm="sasa",
            probe_radius=1.4,
            n_sphere_points=64,
            radii=None,
            offset=0.0,
            radii_mode="gb",
        ):
            called["algorithm"] = algorithm
            called["probe_radius"] = probe_radius
            called["radii"] = radii
            called["offset"] = offset
            called["radii_mode"] = radii_mode

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["frame_indices"] = frame_indices
            return np.array([3.0], dtype=np.float32)

    monkeypatch.setattr(warp_md, "MolSurfPlan", _DummyPlan, raising=False)
    out = molsurf(
        traj,
        system,
        mask=[0],
        probe=1.2,
        offset=0.3,
        radii="parse",
        frame_indices=[0],
    )
    np.testing.assert_allclose(out, np.array([3.0], dtype=np.float32))
    assert called == {
        "algorithm": "sasa",
        "probe_radius": 1.2,
        "radii": None,
        "offset": 0.3,
        "radii_mode": "parse",
        "frame_indices": [0],
    }


def test_molsurf_solutemask_passes_report_and_occluder_selections(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            sel,
            algorithm="sasa",
            probe_radius=1.4,
            n_sphere_points=64,
            radii=None,
            offset=0.0,
            radii_mode="gb",
            solute_selection=None,
        ):
            called["selection"] = list(sel.indices)
            called["solute_selection"] = list(solute_selection.indices)

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr(warp_md, "MolSurfPlan", _DummyPlan, raising=False)
    out = molsurf(traj, system, mask=[0], solutemask=[0, 1])

    np.testing.assert_allclose(out, np.array([1.0], dtype=np.float32))
    assert called == {"selection": [0], "solute_selection": [0, 1]}


def test_molsurf_atom_area_uses_native_detail(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            _sel,
            algorithm="sasa",
            probe_radius=1.4,
            n_sphere_points=64,
            radii=None,
            offset=0.0,
            radii_mode="gb",
            atom_area=False,
        ):
            called["algorithm"] = algorithm
            called["atom_area"] = atom_area

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return (
                np.array([12.0], dtype=np.float32),
                np.array([[12.0]], dtype=np.float32),
                1,
                1,
            )

    monkeypatch.setattr(warp_md, "MolSurfPlan", _DummyPlan, raising=False)
    out = molsurf(traj, system, mask=[0], atom_area=True)
    assert called == {"algorithm": "sasa", "atom_area": True}
    np.testing.assert_allclose(out["molsurf"], np.array([12.0], dtype=np.float32))
    np.testing.assert_allclose(out["atom_area"], np.array([[12.0]], dtype=np.float32))


def test_molsurf_volume_uses_native_detail(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            _sel,
            algorithm="sasa",
            probe_radius=1.4,
            n_sphere_points=64,
            radii=None,
            offset=0.0,
            radii_mode="gb",
            atom_area=False,
            volume=False,
        ):
            called["algorithm"] = algorithm
            called["volume"] = volume

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return (
                np.array([12.0], dtype=np.float32),
                None,
                np.array([4.0], dtype=np.float32),
                1,
                1,
            )

    monkeypatch.setattr(warp_md, "MolSurfPlan", _DummyPlan, raising=False)
    out = molsurf(traj, system, mask=[0], volume=True)
    assert called == {"algorithm": "sasa", "volume": True}
    np.testing.assert_allclose(out["molsurf"], np.array([12.0], dtype=np.float32))
    np.testing.assert_allclose(out["volume"], np.array([4.0], dtype=np.float32))


def test_surf_no_python_fallback_when_plan_missing(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    class _MissingPlan:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("SurfPlan binding unavailable in this build.")

    monkeypatch.setattr(warp_md, "SurfPlan", _MissingPlan, raising=False)
    try:
        surf(traj, system, mask=[0], algorithm="sasa")
    except RuntimeError as exc:
        assert "SurfPlan binding unavailable" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for missing SurfPlan binding")


def test_surf_rejects_non_native_trajectory(monkeypatch):
    monkeypatch.setattr(surf_mod, "is_native_traj", lambda _traj: False)
    with pytest.raises(RuntimeError, match="Rust-backed trajectory"):
        surf(_DummyTraj(np.zeros((1, 1, 3), dtype=np.float32)), _DummySystem(1), mask=[0])


def test_molsurf_rejects_non_native_trajectory(monkeypatch):
    monkeypatch.setattr(surf_mod, "is_native_traj", lambda _traj: False)
    with pytest.raises(RuntimeError, match="Rust-backed trajectory"):
        molsurf(_DummyTraj(np.zeros((1, 1, 3), dtype=np.float32)), _DummySystem(1), mask=[0])


def test_surf_algorithm_case_normalized(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            _sel,
            algorithm="bbox",
            probe_radius=1.4,
            n_sphere_points=64,
            radii=None,
            offset=0.0,
            nbrcut=2.5,
            solute_selection=None,
            radii_mode="gb",
        ):
            called["algorithm"] = algorithm

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr(warp_md, "SurfPlan", _DummyPlan, raising=False)
    out = surf(traj, system, mask=[0], algorithm="SASA")
    np.testing.assert_allclose(out, np.array([1.0], dtype=np.float32))
    assert called["algorithm"] == "sasa"


def test_surf_defaults_to_lcpo_and_passes_lcpo_options(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            _sel,
            algorithm="bbox",
            probe_radius=1.4,
            n_sphere_points=64,
            radii=None,
            offset=0.0,
            nbrcut=2.5,
            solute_selection=None,
            radii_mode="gb",
        ):
            called["algorithm"] = algorithm
            called["offset"] = offset
            called["nbrcut"] = nbrcut
            called["solute_selection"] = solute_selection

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return np.array([2.0], dtype=np.float32)

    monkeypatch.setattr(warp_md, "SurfPlan", _DummyPlan, raising=False)
    out = surf(traj, system, mask=[0], offset=1.2, nbrcut=2.1, solutemask=[0])
    np.testing.assert_allclose(out, np.array([2.0], dtype=np.float32))
    assert called["algorithm"] == "lcpo"
    assert called["offset"] == 1.2
    assert called["nbrcut"] == 2.1
    assert called["solute_selection"].indices == [0]


def test_sasa_alias_keeps_zero_radius_offset(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            _sel,
            algorithm="bbox",
            probe_radius=1.4,
            n_sphere_points=64,
            radii=None,
            offset=0.0,
            nbrcut=2.5,
            solute_selection=None,
            radii_mode="gb",
        ):
            called["algorithm"] = algorithm
            called["offset"] = offset

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr(warp_md, "SurfPlan", _DummyPlan, raising=False)
    from warp_md.analysis.surf import sasa

    sasa(traj, system, mask=[0])
    assert called["algorithm"] == "sasa"
    assert called["offset"] == 0.0


def test_surface_wrappers_accept_atom_index_masks():
    coords = np.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]], dtype=np.float32)
    system = warp_md.System.from_arrays(
        {
            "name": ["A", "B"],
            "resname": ["RES", "RES"],
            "resid": [1, 2],
            "chain_id": [0, 0],
            "mass": [1.0, 1.0],
        },
        positions0=coords[0],
    )
    expected = np.array([4.0 * np.pi], dtype=np.float32)

    surf_out = surf(
        warp_md.Trajectory.from_numpy(coords),
        system,
        mask="@1",
        algorithm="sasa",
        probe_radius=0.0,
        radii=[1.0],
        n_sphere_points=32,
    )
    molsurf_out = molsurf(
        warp_md.Trajectory.from_numpy(coords),
        system,
        mask="@1",
        probe=0.0,
        radii=[1.0],
        n_sphere_points=32,
    )

    np.testing.assert_allclose(surf_out, expected, rtol=1e-6)
    np.testing.assert_allclose(molsurf_out, expected, rtol=1e-6)


def test_surfplan_binding_defaults_to_lcpo():
    coords = np.array([[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]], dtype=np.float32)
    system = warp_md.System.from_arrays(
        {
            "name": ["C", "C"],
            "resname": ["RES", "RES"],
            "resid": [1, 2],
            "chain_id": [0, 0],
            "mass": [12.0, 12.0],
        },
        positions0=coords[0],
    )
    sel = system.select("name C")

    default = warp_md.SurfPlan(sel).run(
        warp_md.Trajectory.from_numpy(coords),
        system,
        device="cpu",
    )
    explicit = warp_md.SurfPlan(sel, algorithm="lcpo").run(
        warp_md.Trajectory.from_numpy(coords),
        system,
        device="cpu",
    )

    np.testing.assert_allclose(default, explicit, rtol=1e-6)


def test_surf_rejects_invalid_numeric_contract(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    class _DummyPlan:
        def __init__(
            self,
            _sel,
            algorithm="bbox",
            probe_radius=1.4,
            n_sphere_points=64,
            radii=None,
            offset=0.0,
            nbrcut=2.5,
            solute_selection=None,
            radii_mode="gb",
        ):
            pass

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return np.array([1.0], dtype=np.float32)

    monkeypatch.setattr(warp_md, "SurfPlan", _DummyPlan, raising=False)
    with pytest.raises(ValueError, match="probe_radius must be a finite value >= 0"):
        surf(traj, system, mask=[0], probe_radius=-0.1)
    with pytest.raises(ValueError, match="n_sphere_points must be a positive integer"):
        surf(traj, system, mask=[0], n_sphere_points=0)
    with pytest.raises(ValueError, match="radii values must be finite and > 0"):
        surf(traj, system, mask=[0], radii=[1.0, np.inf])
    with pytest.raises(ValueError, match="offset must be a finite value >= 0"):
        surf(traj, system, mask=[0], offset=-0.1)
    with pytest.raises(ValueError, match="nbrcut must be a finite value > 0"):
        surf(traj, system, mask=[0], nbrcut=0.0)
    with pytest.raises(ValueError, match="radii_mode must be 'gb', 'parse', or 'vdw'"):
        surf(traj, system, mask=[0], radii_mode="bad")
    with pytest.raises(ValueError, match="surface detail output requires algorithm='sasa'"):
        surf(traj, system, mask=[0], algorithm="bbox", atom_area=True)
    with pytest.raises(ValueError, match="surface detail output requires algorithm='sasa'"):
        surf(traj, system, mask=[0], algorithm="bbox", volume=True)
