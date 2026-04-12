import numpy as np

import warp_md
from warp_md.analysis.align import align, align_principal_axis, superpose
from warp_md.analysis.autoimage import autoimage
from warp_md.analysis.density import density
from warp_md.analysis.fluct import atomicfluct, rmsf
from warp_md.analysis.closest import closest
from warp_md.analysis.matrix import correl, covar, dist, mwcovar
from warp_md.analysis.multidihedral import multidihedral
from warp_md.analysis.pca import pca, projection
from warp_md.analysis.rmsd import distance_rmsd, pairwise_rmsd
from warp_md.analysis.rotation import rotation_matrix
from warp_md.analysis.structure import make_structure, mean_structure, radgyr_tensor
from warp_md.analysis.transform import center, rotate, scale, transform, translate
from warp_md.analysis.atomiccorr import atomiccorr
from warp_md.analysis.velocity import get_velocity
from warp_md.analysis.voxel import count_in_voxel
from warp_md.analysis.volmap import volmap
from warp_md.analysis.watershell import watershell


def _atom_table(n_atoms: int, *, resid=None):
    return {
        "name": ["CA"] * n_atoms,
        "resname": ["ALA"] * n_atoms,
        "resid": list(range(1, n_atoms + 1)) if resid is None else list(resid),
        "chain": ["A"] * n_atoms,
        "element": ["C"] * n_atoms,
        "mass": [1.0] * n_atoms,
    }


def test_system_from_arrays_supports_selection():
    coords0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(_atom_table(coords0.shape[0]), positions0=coords0)
    sel = system.select("protein and name CA")
    assert list(sel.indices) == [0, 1, 2]
    np.testing.assert_allclose(system.positions0(), coords0, rtol=1e-6)


def test_trajectory_from_numpy_exposes_chunk_box_and_time():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    box = np.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
    time_ps = np.array([0.5, 1.5], dtype=np.float32)

    traj = warp_md.Trajectory.from_numpy(coords, box=box, time_ps=time_ps)
    assert traj.count_frames() == 2
    chunk = traj.read_chunk()
    np.testing.assert_allclose(chunk["coords"], coords, rtol=1e-6)
    np.testing.assert_allclose(chunk["box"], box, rtol=1e-6)
    np.testing.assert_allclose(chunk["time_ps"], time_ps, rtol=1e-6)


def test_distance_rmsd_runs_on_in_memory_native_objects():
    ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    frame1 = ref + np.array([2.0, -1.0, 3.0], dtype=np.float32)
    coords = np.stack([ref, frame1], axis=0)

    system = warp_md.System.from_arrays(_atom_table(coords.shape[1]), positions0=ref)
    traj = warp_md.Trajectory.from_numpy(coords)
    vals = distance_rmsd(traj, system, mask="protein", ref=0)
    np.testing.assert_allclose(vals, np.zeros(2, dtype=np.float32), atol=1e-6)


def test_align_supports_nonzero_reference_and_ref_mask():
    coords = np.array(
        [
            [[5.0, 5.0, 0.0], [6.0, 5.0, 0.0], [7.0, 5.0, 0.0]],
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    expected = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    system = warp_md.System.from_arrays(_atom_table(3, resid=[1, 2, 3]))
    traj = warp_md.Trajectory.from_numpy(coords)
    aligned, transforms = align(
        traj,
        system,
        mask="resid 1:2",
        ref=-1,
        ref_mask="resid 2:3",
        return_transforms=True,
    )
    out = aligned.read_chunk()["coords"]
    np.testing.assert_allclose(out[0], expected, atol=1e-5)
    np.testing.assert_allclose(out[1], expected, atol=1e-5)
    assert transforms.shape == (2, 12)


def test_superpose_keeps_unselected_frames_when_subset_requested():
    coords = np.array(
        [
            [[5.0, 5.0, 0.0], [6.0, 5.0, 0.0], [7.0, 5.0, 0.0]],
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    expected = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    system = warp_md.System.from_arrays(_atom_table(3, resid=[1, 2, 3]))
    traj = warp_md.Trajectory.from_numpy(coords)
    aligned = superpose(
        traj,
        system,
        mask="resid 1:2",
        ref=-1,
        ref_mask="resid 2:3",
        frame_indices=[0],
    )
    out = aligned.read_chunk()["coords"]
    np.testing.assert_allclose(out[0], expected, atol=1e-5)
    np.testing.assert_allclose(out[1], coords[1], atol=1e-5)


def test_distance_rmsd_supports_negative_reference_with_box():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    box = np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]], dtype=np.float32)

    system = warp_md.System.from_arrays(_atom_table(2))
    traj = warp_md.Trajectory.from_numpy(coords, box=box)
    vals = distance_rmsd(traj, system, mask="all", ref=-1, pbc="orthorhombic")
    np.testing.assert_allclose(vals, np.array([1.0, 0.0], dtype=np.float32), atol=1e-6)


def test_pairwise_rmsd_supports_frame_subset_and_half_output():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    system = warp_md.System.from_arrays(_atom_table(2))
    traj = warp_md.Trajectory.from_numpy(coords)
    full = pairwise_rmsd(
        traj,
        system,
        mask="all",
        metric="nofit",
        mat_type="full",
        frame_indices=[0, 2, 3],
    )

    traj = warp_md.Trajectory.from_numpy(coords)
    half = pairwise_rmsd(
        traj,
        system,
        mask="all",
        metric="nofit",
        mat_type="half",
        frame_indices=[0, 2, 3],
    )
    np.testing.assert_allclose(half, full[np.triu_indices(3, k=1)], atol=1e-6)


def test_rotation_matrix_supports_nonzero_reference_and_subset():
    ref = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    rot = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    frame0 = ref @ rot.T + np.array([5.0, -3.0, 2.0], dtype=np.float32)
    coords = np.stack([frame0, ref], axis=0)

    system = warp_md.System.from_arrays(_atom_table(3), positions0=ref)
    traj = warp_md.Trajectory.from_numpy(coords)
    mats, rmsd = rotation_matrix(
        traj,
        system,
        mask="all",
        ref=-1,
        frame_indices=[0],
        with_rmsd=True,
    )
    np.testing.assert_allclose(mats[0], rot.T, atol=1e-5)
    np.testing.assert_allclose(mats[1], np.eye(3, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(rmsd, np.zeros(2, dtype=np.float32), atol=1e-6)


def test_transform_family_runs_on_in_memory_native_objects():
    coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float32)
    box = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    translated = translate(warp_md.Trajectory.from_numpy(coords, box=box), [1.0, 2.0, 3.0]).read_chunk()["coords"]
    np.testing.assert_allclose(translated[0, 0], [1.0, 2.0, 3.0], atol=1e-6)

    scaled = scale(warp_md.Trajectory.from_numpy(coords, box=box), [2.0, 3.0, 4.0]).read_chunk()["coords"]
    np.testing.assert_allclose(scaled[0, 1], [2.0, 0.0, 0.0], atol=1e-6)

    rotated = rotate(warp_md.Trajectory.from_numpy(coords, box=box), rot).read_chunk()["coords"]
    np.testing.assert_allclose(rotated[0, 1], [0.0, 1.0, 0.0], atol=1e-6)

    transformed = transform(
        warp_md.Trajectory.from_numpy(coords, box=box),
        rotation=rot,
        translation=[1.0, 0.0, 0.0],
    ).read_chunk()["coords"]
    np.testing.assert_allclose(transformed[0, 1], [1.0, 1.0, 0.0], atol=1e-6)

    system = warp_md.System.from_arrays(_atom_table(2), positions0=coords[0])
    centered = center(
        warp_md.Trajectory.from_numpy(coords, box=box),
        system,
        mask="all",
        mode="box",
    ).read_chunk()["coords"]
    np.testing.assert_allclose(centered.mean(axis=1), np.array([[5.0, 5.0, 5.0]], dtype=np.float32), atol=1e-6)


def test_autoimage_mean_structure_and_make_structure_compose_natively():
    ref = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    coords = np.array(
        [
            [[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
            [[11.0, 1.0, 0.0], [13.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    box = np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]], dtype=np.float32)

    system = warp_md.System.from_arrays(_atom_table(2), positions0=ref)
    imaged = autoimage(
        warp_md.Trajectory.from_numpy(coords, box=box),
        system,
        mask="all",
        frame_indices=[1],
    ).read_chunk()["coords"]
    np.testing.assert_allclose(imaged.shape, (1, 2, 3))
    np.testing.assert_allclose(imaged[0], np.array([[1.0, 1.0, 0.0], [3.0, 1.0, 0.0]], dtype=np.float32), atol=1e-6)

    mean = mean_structure(
        warp_md.Trajectory.from_numpy(coords, box=box),
        system,
        mask="all",
        autoimage=True,
        rmsfit=0,
    )
    made = make_structure(
        warp_md.Trajectory.from_numpy(coords, box=box),
        system,
        mask="all",
        autoimage=True,
        rmsfit=0,
    )
    np.testing.assert_allclose(mean, ref, atol=1e-5)
    np.testing.assert_allclose(made, ref, atol=1e-5)


def test_align_principal_axis_and_radgyr_tensor_run_natively():
    coords = np.array(
        [
            [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(_atom_table(3), positions0=coords[0])

    aligned, transforms = align_principal_axis(
        warp_md.Trajectory.from_numpy(coords),
        system,
        mask="all",
        return_transforms=True,
    )
    out = aligned.read_chunk()["coords"]
    assert transforms.shape == (2, 12)
    np.testing.assert_allclose(out.mean(axis=1), 0.0, atol=1e-6)

    rg = radgyr_tensor(
        warp_md.Trajectory.from_numpy(coords[:, :2, :]),
        warp_md.System.from_arrays(_atom_table(2), positions0=coords[0, :2, :]),
        mask="all",
        frame_indices=[1],
        dtype="dict",
    )
    np.testing.assert_allclose(rg["rg"], np.array([1.0], dtype=np.float32), atol=1e-6)


def test_closest_fluct_velocity_and_pca_run_natively():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    time = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    system = warp_md.System.from_arrays(_atom_table(3), positions0=coords[0])

    out = closest(
        warp_md.Trajectory.from_numpy(coords),
        system,
        mask="all",
        n_solvents=2,
    ).read_chunk()["coords"]
    assert out.shape == (3, 2, 3)

    fluct = rmsf(
        warp_md.Trajectory.from_numpy(coords),
        system,
        mask="all",
        bymask=True,
    )
    assert fluct.shape == (1, 2)
    assert float(fluct[0, 1]) > 0.0

    vel = get_velocity(
        warp_md.Trajectory.from_numpy(coords, time_ps=time),
        system,
        mask="all",
    )
    assert vel.shape == coords.shape
    np.testing.assert_allclose(vel[1, 1], np.array([1.0, 0.0, 0.0], dtype=np.float32), atol=1e-6)

    proj, (evals, evecs) = pca(
        warp_md.Trajectory.from_numpy(coords),
        system,
        mask="all",
        n_vecs=2,
        fit=False,
    )
    assert proj.shape == (2, 3)
    assert evals.shape == (2,)
    assert evecs.shape == (2, 9)

    proj2 = projection(
        warp_md.Trajectory.from_numpy(coords),
        system,
        mask="all",
        eigenvectors=evecs,
        eigenvalues=evals,
        scalar_type="mwcovar",
    )
    assert proj2.shape == (2, 3)

    adp = atomicfluct(
        warp_md.Trajectory.from_numpy(coords),
        system,
        mask="all",
        calcadp=True,
        frame_indices=[0, 2],
    )
    assert adp.shape == (3, 7)


def test_matrix_watershell_atomiccorr_and_multidihedral_run_natively():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [3.5, 0.0, 0.0], [4.5, 0.0, 0.0], [7.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [4.5, 0.0, 0.0], [5.5, 0.0, 0.0], [9.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(_atom_table(4), positions0=coords[0])
    traj = warp_md.Trajectory.from_numpy(coords)

    assert dist(warp_md.Trajectory.from_numpy(coords), system, mask="all").shape == (4, 4)
    assert covar(warp_md.Trajectory.from_numpy(coords), system, mask="all").shape == (12, 12)
    assert mwcovar(warp_md.Trajectory.from_numpy(coords), system, mask="all").shape == (12, 12)
    corr = correl(warp_md.Trajectory.from_numpy(coords), system, mask="all")
    assert corr.shape == (4, 4)
    np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)

    time, data = atomiccorr(
        warp_md.Trajectory.from_numpy(coords[:, :1, :]),
        warp_md.System.from_arrays(_atom_table(1), positions0=coords[0, :1, :]),
        mask="all",
        reference="frame0",
    )
    np.testing.assert_allclose(time, np.array([1.0, 2.0], dtype=np.float32), atol=1e-6)
    assert data.shape == (2,)

    shell_system = warp_md.System.from_arrays(
        {
            "name": ["S", "W1", "W2", "W3"],
            "resname": ["SOL", "WAT", "WAT", "WAT"],
            "resid": [1, 2, 2, 2],
            "chain_id": [0, 0, 0, 0],
            "mass": [32.0, 16.0, 16.0, 16.0],
        },
        positions0=coords[0],
    )
    counts = watershell(
        warp_md.Trajectory.from_numpy(coords),
        shell_system,
        solute_mask="resid 1:1",
        solvent_mask="resid 2:2",
        lower=3.0,
        upper=5.0,
        image=False,
    )
    np.testing.assert_allclose(counts, np.array([2.0, 2.0, 1.0], dtype=np.float32), atol=1e-6)

    dih_coords = np.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0], [3.0, 2.0, 0.0]]],
        dtype=np.float32,
    )
    dih_system = warp_md.System.from_arrays(
        {
            "name": ["N", "CA", "C", "N", "CA", "C"],
            "resname": ["ALA"] * 6,
            "resid": [1, 1, 1, 2, 2, 2],
            "chain_id": [0] * 6,
            "mass": [1.0] * 6,
        },
        positions0=dih_coords[0],
    )
    out = multidihedral(
        warp_md.Trajectory.from_numpy(dih_coords),
        dih_system,
        dihedral_types="phi psi",
        dtype="dict",
    )
    assert sorted(out) == ["phi:2", "psi:1"]
    assert out["phi:2"].shape == (1,)
    assert out["psi:1"].shape == (1,)


def test_grid_family_returns_native_grid_output_on_in_memory_objects():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [0.4, 0.4, 0.4], [1.1, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.8, 0.2, 0.2], [0.9, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(_atom_table(3), positions0=coords[0])

    for fn in (count_in_voxel, density, volmap):
        out = fn(
            warp_md.Trajectory.from_numpy(coords),
            system,
            "all",
            "resid 1:1",
            (1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
            frame_indices=[0, 1],
        )
        assert sorted(out) == ["dims", "first", "last", "max", "mean", "min", "std"]
        assert out["dims"] == (2, 2, 2)
        assert out["mean"].shape == (8,)
        np.testing.assert_allclose(out["mean"][0], 2.5, atol=1e-6)
