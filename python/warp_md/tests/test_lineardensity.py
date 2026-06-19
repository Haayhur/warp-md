import numpy as np
import pytest
import warp_md

from warp_md.analysis.lineardensity import linear_density, lineardensity
from warp_md.analysis.trajectory import ArrayTrajectory


def _atom_table(n_atoms):
    return {
        "name": ["C"] * n_atoms,
        "resname": ["MOL"] * n_atoms,
        "resid": list(range(1, n_atoms + 1)),
        "chain_id": [0] * n_atoms,
        "mass": [1.0] * n_atoms,
    }


def _coords():
    return np.array(
        [
            [
                [0.0, 0.0, 0.25],
                [0.0, 0.0, 0.75],
                [0.0, 0.0, 1.25],
                [0.0, 0.0, 1.75],
            ],
            [
                [0.0, 0.0, 0.25],
                [0.0, 0.0, 1.25],
                [0.0, 0.0, 1.25],
                [0.0, 0.0, 1.75],
            ],
        ],
        dtype=np.float32,
    )


def test_lineardensity_number_density_profile():
    coords = _coords()
    system = warp_md.System.from_arrays(_atom_table(coords.shape[1]), positions0=coords[0])

    out = lineardensity(
        warp_md.Trajectory.from_numpy(coords),
        system,
        selection="all",
        axis="z",
        bin=1.0,
        range=(0.0, 2.0),
        norm="density",
        cross_section_area=10.0,
        device="cpu",
    )

    np.testing.assert_allclose(out["axis"], np.array([0.5, 1.5], dtype=np.float32))
    np.testing.assert_allclose(out["profile"], np.array([0.15, 0.25], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(out["mean_weight"], np.array([1.5, 2.5], dtype=np.float32))
    assert out["axis_name"] == "z"
    assert out["norm"] == "density"


def test_linear_density_charge_weight_and_profile_dtype():
    coords = _coords()
    system = warp_md.System.from_arrays(_atom_table(coords.shape[1]), positions0=coords[0])

    profile = linear_density(
        warp_md.Trajectory.from_numpy(coords),
        system,
        selection="all",
        axis="z",
        bin=1.0,
        range=(0.0, 2.0),
        weight="charge",
        norm="count",
        charges=[1.0, -1.0, 2.0, -2.0],
        dtype="profile",
    )

    np.testing.assert_allclose(profile, np.array([0.5, -0.5], dtype=np.float32), atol=1e-6)


def test_lineardensity_requires_native_trajectory():
    coords = _coords()[:1]
    system = warp_md.System.from_arrays(_atom_table(coords.shape[1]), positions0=coords[0])

    with pytest.raises(RuntimeError, match="Rust-backed trajectory"):
        lineardensity(ArrayTrajectory(coords), system, selection="all")
