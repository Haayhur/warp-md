import numpy as np
import pytest
import warp_md

from warp_md.analysis.shape import (
    acylindricity,
    asphericity,
    principal_moments,
    relative_shape_anisotropy,
    shape_descriptors,
)
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


def _native_system(coords):
    return warp_md.System.from_arrays(
        _DummySystem(coords.shape[1]).atom_table(),
        positions0=coords[0],
    )


def test_shape_descriptors_basic():
    coords = np.array(
        [
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, -2.0, 0.0], [0.0, 2.0, 0.0]],
        ],
        dtype=np.float32,
    )
    system = _native_system(coords)

    out = shape_descriptors(
        warp_md.Trajectory.from_numpy(coords),
        system,
        mask="all",
        device="cpu",
    )

    np.testing.assert_allclose(out["rg"], np.array([1.0, 2.0], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(
        out["principal_moments"],
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 4.0]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        out["asphericity"], np.array([1.0, 4.0], dtype=np.float32), rtol=1e-6
    )
    np.testing.assert_allclose(
        out["acylindricity"], np.array([0.0, 0.0], dtype=np.float32), atol=1e-6
    )
    np.testing.assert_allclose(
        out["relative_shape_anisotropy"],
        np.array([1.0, 1.0], dtype=np.float32),
        rtol=1e-6,
    )


def test_shape_descriptor_helpers_and_frame_subset():
    coords = np.array(
        [
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, -2.0, 0.0], [0.0, 2.0, 0.0]],
        ],
        dtype=np.float32,
    )
    system = _native_system(coords)

    np.testing.assert_allclose(
        principal_moments(warp_md.Trajectory.from_numpy(coords), system, frame_indices=[1]),
        np.array([[0.0, 0.0, 4.0]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        asphericity(warp_md.Trajectory.from_numpy(coords), system, frame_indices=[1]),
        np.array([4.0], dtype=np.float32),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        acylindricity(warp_md.Trajectory.from_numpy(coords), system, frame_indices=[1]),
        np.array([0.0], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        relative_shape_anisotropy(
            warp_md.Trajectory.from_numpy(coords), system, frame_indices=[1]
        ),
        np.array([1.0], dtype=np.float32),
        rtol=1e-6,
    )


def test_shape_descriptors_requires_native_trajectory():
    coords = np.array([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32)
    system = _DummySystem(coords.shape[1])

    with pytest.raises(RuntimeError, match="Rust-backed trajectory"):
        shape_descriptors(ArrayTrajectory(coords), system, mask="all")
