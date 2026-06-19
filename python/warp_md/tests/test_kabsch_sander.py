import numpy as np
import pytest
import warp_md

from warp_md.analysis.kabsch_sander import kabsch_sander
from warp_md.analysis.trajectory import ArrayTrajectory


def _backbone_fixture():
    names = []
    elements = []
    resnames = []
    resids = []
    chains = []
    masses = []
    coords = []
    residue_points = [
        {
            "N": (-1.0, 0.0, 0.0),
            "H": (-1.2, 0.0, 0.0),
            "CA": (0.0, 1.0, 0.0),
            "C": (0.0, 0.0, 0.0),
            "O": (0.0, 0.0, 1.0),
        },
        {
            "N": (4.0, 0.0, 0.0),
            "H": (4.0, 0.0, 1.0),
            "CA": (4.0, 1.0, 0.0),
            "C": (5.0, 1.0, 0.0),
            "O": (5.0, 1.0, 1.0),
        },
        {
            "N": (2.0, 0.0, 0.0),
            "H": (0.0, 0.0, 1.5),
            "CA": (2.0, 1.0, 0.0),
            "C": (3.0, 1.0, 0.0),
            "O": (3.0, 1.0, 1.0),
        },
    ]
    mass_by_name = {"N": 14.0, "H": 1.0, "CA": 12.0, "C": 12.0, "O": 16.0}
    element_by_name = {"N": "N", "H": "H", "CA": "C", "C": "C", "O": "O"}
    for resid, points in enumerate(residue_points, start=1):
        for name in ("N", "H", "CA", "C", "O"):
            names.append(name)
            elements.append(element_by_name[name])
            resnames.append("ALA")
            resids.append(resid)
            chains.append(0)
            masses.append(mass_by_name[name])
            coords.append(points[name])
    coords = np.asarray([coords], dtype=np.float32)
    system = warp_md.System.from_arrays(
        {
            "name": names,
            "element": elements,
            "resname": resnames,
            "resid": resids,
            "chain_id": chains,
            "mass": masses,
        },
        positions0=coords[0],
    )
    return system, coords


def test_kabsch_sander_reports_energy_and_hbond_matrix():
    system, coords = _backbone_fixture()
    out = kabsch_sander(
        warp_md.Trajectory.from_numpy(coords),
        system,
        selection="all",
        device="cpu",
    )

    assert out["residues"].tolist() == ["ALA:1", "ALA:2", "ALA:3"]
    assert out["energy"].shape == (1, 3, 3)
    assert out["hbonds"].shape == (1, 3, 3)
    assert out["hbonds"].dtype == np.bool_
    assert out["energy_cutoff"] == -0.5
    assert out["energy"][0, 0, 2] < -0.5
    assert out["hbonds"][0, 0, 2]
    assert np.isnan(out["energy"][0, 0, 0])


def test_kabsch_sander_dtype_routes():
    system, coords = _backbone_fixture()

    energy = kabsch_sander(
        warp_md.Trajectory.from_numpy(coords),
        system,
        selection="all",
        dtype="energy",
    )
    hbonds = kabsch_sander(
        warp_md.Trajectory.from_numpy(coords),
        system,
        selection="all",
        dtype="hbonds",
    )
    labels, tuple_energy, tuple_hbonds = kabsch_sander(
        warp_md.Trajectory.from_numpy(coords),
        system,
        selection="all",
        dtype="tuple",
    )

    assert energy.shape == (1, 3, 3)
    assert hbonds.shape == (1, 3, 3)
    assert labels.tolist() == ["ALA:1", "ALA:2", "ALA:3"]
    np.testing.assert_allclose(tuple_energy, energy, equal_nan=True)
    np.testing.assert_array_equal(tuple_hbonds, hbonds)


def test_kabsch_sander_requires_native_trajectory():
    system, coords = _backbone_fixture()
    with pytest.raises(RuntimeError, match="Rust-backed trajectory"):
        kabsch_sander(ArrayTrajectory(coords), system, selection="all")
