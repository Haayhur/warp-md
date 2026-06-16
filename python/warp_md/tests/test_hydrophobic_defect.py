import numpy as np

import warp_md


def test_hydrophobic_defects_native_wrapper_reports_frame_summary():
    coords = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [5.0, 5.0, 2.0],
            ]
        ],
        dtype=np.float32,
    )
    box = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
    system = warp_md.System.from_arrays(
        {
            "name": ["C1", "R1"],
            "resname": ["MEM", "MEM"],
            "resid": [1, 1],
            "chain_id": [0, 0],
            "element": ["C", "C"],
            "mass": [12.0, 12.0],
        },
        positions0=coords[0],
    )
    out = warp_md.hydrophobic_defects(
        warp_md.Trajectory.from_numpy(coords, box=box),
        system,
        lipid_selection="name C1",
        reference_selection="name R1",
        voxel_size=1.0,
        z_bounds=(0.0, 3.0),
        probe_radius=0.1,
        defect_radius=1.1,
    )

    assert out["kind"] == "hydrophobic_defects"
    assert out["dims"] == (10, 10, 3)
    assert out["frame_counts"].shape == (1,)
    assert out["frame_counts"][0] > 0
    np.testing.assert_allclose(out["frame_volume"][0], out["frame_counts"][0])
    assert out["frame_cluster_count"].shape == (1,)
    assert out["frame_largest_cluster"].shape == (1,)
    assert out["max_lifetime"].shape == out["last"].shape


def test_hydrophobic_defect_point_export(tmp_path):
    result = {
        "dims": (2, 2, 2),
        "voxel_size": 1.0,
        "z_bounds": (0.0, 2.0),
        "grid_mode": "lattice_nodes",
        "last": np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=np.uint32),
    }
    points = warp_md.hydrophobic_defect_points(result)
    np.testing.assert_allclose(points, [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    written = warp_md.write_hydrophobic_defect_points(result, tmp_path / "defects.xyz")
    assert written["points"] == 2
    assert (tmp_path / "defects.xyz").read_text().count("\n") == 2


def test_hydrophobic_defects_auto_leaflet_matches_manual_upper_selection():
    coords = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [5.0, 5.0, 2.0],
                [0.0, 0.0, -4.0],
                [5.0, 5.0, -2.0],
            ]
        ],
        dtype=np.float32,
    )
    box = np.array([[10.0, 10.0, 12.0]], dtype=np.float32)
    system = warp_md.System.from_arrays(
        {
            "name": ["CU", "RU", "CL", "RL"],
            "resname": ["LIPU", "REFU", "LIPL", "REFL"],
            "resid": [1, 1, 2, 2],
            "chain_id": [0, 0, 0, 0],
            "element": ["C", "C", "C", "C"],
            "vdw_radius": [1.7, 1.7, 1.7, 1.7],
            "mass": [12.0, 12.0, 12.0, 12.0],
        },
        positions0=coords[0],
    )
    traj = warp_md.Trajectory.from_numpy(coords, box=box)
    manual = warp_md.hydrophobic_defects(
        traj,
        system,
        lipid_selection="resname LIPU",
        reference_selection="resname REFU",
        voxel_size=1.0,
        z_bounds=(-5.0, 3.0),
        probe_radius=0.1,
        defect_radius=1.1,
    )
    auto = warp_md.hydrophobic_defects(
        warp_md.Trajectory.from_numpy(coords, box=box),
        system,
        lipid_selection="resname LIPU or resname LIPL",
        reference_selection="resname REFU or resname REFL",
        midplane_selection="resname LIPU or resname LIPL",
        leaflet="upper",
        leaflet_bins=1,
        voxel_size=1.0,
        z_bounds=(-5.0, 3.0),
        probe_radius=0.1,
        defect_radius=1.1,
    )

    np.testing.assert_array_equal(auto["frame_counts"], manual["frame_counts"])
    np.testing.assert_array_equal(auto["last"], manual["last"])

    local = warp_md.hydrophobic_defects(
        warp_md.Trajectory.from_numpy(coords, box=box),
        system,
        lipid_selection="resname LIPU or resname LIPL",
        reference_selection="resname REFU or resname REFL",
        midplane_selection="resname LIPU or resname LIPL",
        leaflet="upper",
        leaflet_bins=2,
        voxel_size=1.0,
        z_bounds=(-5.0, 3.0),
        probe_radius=0.1,
        defect_radius=1.1,
    )
    assert local["frame_counts"][0] > 0


def _read_reference_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        rows.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return rows


def _shift_rows(rows, box, center, *, z_cutoff=40.0):
    lx, ly, _lz = box
    cx, cy, _cz = center
    x_lo = cx - 0.5 * lx
    x_hi = cx + 0.5 * lx
    y_lo = cy - 0.5 * ly
    y_hi = cy + 0.5 * ly
    names = []
    coords = []
    elements = []
    for name, x, y, z in rows:
        if x < x_lo:
            x += lx
        elif x > x_hi:
            x -= lx
        if y < y_lo:
            y += ly
        elif y > y_hi:
            y -= ly
        names.append(name)
        elements.append(name[0])
        coords.append([x - x_lo, y - y_lo, z - z_cutoff])
    return names, elements, coords


def test_hydrophobic_defects_matches_3d_lipid_packing_reference_counts():
    from pathlib import Path

    ref_dir = Path(__file__).resolve().parents[3] / "ref" / "3DLipidPackingDefects"
    if not ref_dir.exists():
        import pytest

        pytest.skip("3DLipidPackingDefects reference repo is not checked out")

    boxes = np.loadtxt(ref_dir / "input_files" / "box_dimension.dat", dtype=np.float64)
    centers = np.loadtxt(ref_dir / "input_files" / "box_center.dat", dtype=np.float64)
    expected_counts = [
        sum(1 for line in (ref_dir / "output_files" / f"defects_{run}.xyz").read_text().splitlines() if line.split())
        for run in (1, 2, 3)
    ]
    observed_counts = []
    for run in (1, 2, 3):
        lipid_rows = _read_reference_rows(ref_dir / "input_files" / f"monolayer_atoms_{run}.dat")
        ref_rows = _read_reference_rows(ref_dir / "input_files" / f"ref_atoms_{run}.dat")
        lipid_names, lipid_elements, lipid_coords = _shift_rows(
            lipid_rows, boxes[run - 1], centers[run - 1]
        )
        ref_names, ref_elements, ref_coords = _shift_rows(ref_rows, boxes[run - 1], centers[run - 1])
        coords = np.asarray(lipid_coords + ref_coords, dtype=np.float32)
        n_lipids = len(lipid_coords)
        n_refs = len(ref_coords)
        names = lipid_names + ref_names
        vdw = {"C": 1.7, "H": 1.2, "O": 1.52, "P": 1.8, "N": 1.55}
        system = warp_md.System.from_arrays(
            {
                "name": names,
                "resname": ["LIP"] * n_lipids + ["REF"] * n_refs,
                "resid": list(range(1, n_lipids + n_refs + 1)),
                "chain_id": [0] * (n_lipids + n_refs),
                "element": lipid_elements + ref_elements,
                "vdw_radius": [vdw.get(element, 1.7) for element in lipid_elements + ref_elements],
                "mass": [12.0] * (n_lipids + n_refs),
            },
            positions0=coords,
        )
        traj = warp_md.Trajectory.from_numpy(
            coords[None, :, :],
            box=np.asarray([[boxes[run - 1, 0], boxes[run - 1, 1], boxes[run - 1, 2]]], dtype=np.float32),
        )
        out = warp_md.hydrophobic_defects(
            traj,
            system,
            lipid_selection="resname LIP",
            reference_selection="resname REF",
            voxel_size=1.0,
            z_bounds=(0.0, 50.0),
            probe_radius=1.4,
            defect_radius=7.3,
            grid_mode="lattice_nodes",
        )
        observed_counts.append(int(out["frame_counts"][0]))

    assert observed_counts == expected_counts
