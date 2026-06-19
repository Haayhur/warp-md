import numpy as np

import warp_md
from warp_md.analysis.lipid import _binned_statistic_2d, _nearest_fill_grid


def _bilayer_system():
    coords = np.array(
        [
            [
                [1.0, 1.0, 2.0],
                [1.0, 1.0, 1.0],
                [3.0, 1.0, 2.0],
                [3.0, 1.0, 1.0],
                [1.0, 3.0, -2.0],
                [1.0, 3.0, -1.0],
                [3.0, 3.0, -2.0],
                [3.0, 3.0, -1.0],
            ],
            [
                [1.0, 1.0, 2.2],
                [1.0, 1.0, 1.1],
                [3.0, 1.0, 2.1],
                [3.0, 1.0, 1.0],
                [1.0, 3.0, -2.1],
                [1.0, 3.0, -1.0],
                [3.0, 3.0, -2.2],
                [3.0, 3.0, -1.1],
            ],
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(
        {
            "name": ["H", "T"] * 4,
            "resname": ["LIP"] * 8,
            "resid": [1, 1, 2, 2, 3, 3, 4, 4],
            "chain_id": [0] * 8,
            "mass": [1.0] * 8,
        },
        positions0=coords[0],
    )
    box = np.array([[4.0, 4.0, 8.0], [4.0, 4.0, 8.0]], dtype=np.float32)
    return system, coords, box


def _traj(coords, box):
    return warp_md.Trajectory.from_numpy(coords, box=box)


def test_binned_statistic_2d_mean_and_median():
    x = np.array([0.0, 0.2, 0.9, 1.0], dtype=np.float64)
    y = np.array([0.0, 0.2, 0.9, 1.0], dtype=np.float64)
    values = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float64)
    bins = np.array([0.0, 0.5, 1.0], dtype=np.float64)

    mean, x_edges, y_edges, counts = _binned_statistic_2d(x, y, values, bins, "mean")
    np.testing.assert_allclose(mean, np.array([[2.0, np.nan], [np.nan, 6.0]], dtype=np.float32))
    np.testing.assert_array_equal(counts, np.array([[2, 0], [0, 2]], dtype=np.int64))
    np.testing.assert_allclose(x_edges, bins)
    np.testing.assert_allclose(y_edges, bins)

    median, _, _, _ = _binned_statistic_2d(x, y, values, bins, "median")
    np.testing.assert_allclose(median, np.array([[2.0, np.nan], [np.nan, 6.0]], dtype=np.float32))


def test_binned_statistic_2d_uses_native_array_kernel_when_available(monkeypatch):
    called = {}

    def fake_native(x, y, values, x_edges, y_edges, statistic):
        called["x_shape"] = x.shape
        called["edges"] = (x_edges.shape, y_edges.shape)
        called["statistic"] = statistic
        return np.array([[4.0]], dtype=np.float32), np.array([[2]], dtype=np.int64)

    monkeypatch.setattr(warp_md, "binned_statistic_2d_array", fake_native, raising=False)
    grid, x_edges, y_edges, counts = _binned_statistic_2d(
        [0.0, 0.1],
        [0.0, 0.1],
        [1.0, 2.0],
        np.array([0.0, 1.0], dtype=np.float64),
        "sum",
    )
    np.testing.assert_allclose(grid, np.array([[4.0]], dtype=np.float32))
    np.testing.assert_array_equal(counts, np.array([[2]], dtype=np.int64))
    np.testing.assert_allclose(x_edges, [0.0, 1.0])
    np.testing.assert_allclose(y_edges, [0.0, 1.0])
    assert called == {"x_shape": (2,), "edges": ((2,), (2,)), "statistic": "sum"}


def test_nearest_fill_grid_matches_expected_fill():
    grid = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float32)
    filled = _nearest_fill_grid(grid, tile=False)
    np.testing.assert_allclose(filled, np.array([[1.0, 1.0], [1.0, 4.0]], dtype=np.float32))


def test_nearest_fill_grid_uses_native_array_kernel_when_available(monkeypatch):
    called = {}

    def fake_native(grid, tile):
        called["shape"] = grid.shape
        called["dtype"] = grid.dtype
        called["tile"] = tile
        return np.array([[2.0]], dtype=np.float32)

    monkeypatch.setattr(warp_md, "nearest_fill_grid_array", fake_native, raising=False)
    out = _nearest_fill_grid(np.array([[np.nan]], dtype=np.float64), tile=True)
    np.testing.assert_allclose(out, np.array([[2.0]], dtype=np.float32))
    assert called == {"shape": (1, 1), "dtype": np.dtype("float32"), "tile": True}


def test_lipid_neighbour_composition_uses_native_array_kernel_when_available(monkeypatch):
    called = {}

    def fake_native(matrix, labels, label_values):
        called["matrix_shape"] = matrix.shape
        called["labels_shape"] = labels.shape
        called["label_values"] = label_values.tolist()
        return np.array([[[2, 1], [0, 3]]], dtype=np.int32)

    monkeypatch.setattr(warp_md, "lipid_neighbour_composition_array", fake_native, raising=False)
    out = warp_md.lipid_neighbour_composition(
        np.ones((1, 2, 2), dtype=np.int8),
        np.array([0, 1], dtype=np.int8),
        label_names={"zero": 0, "one": 1},
    )
    np.testing.assert_array_equal(out["counts"], np.array([[[2, 1], [0, 3]]], dtype=np.int32))
    np.testing.assert_array_equal(out["labels"], np.array(["zero", "one"], dtype=object))
    assert called == {"matrix_shape": (1, 2, 2), "labels_shape": (2, 1), "label_values": [0, 1]}


def test_lipid_leaflets_positions_thickness_and_angles():
    system, coords, box = _bilayer_system()

    leaflets = warp_md.lipid_leaflets(_traj(coords, box), system, "name H")
    np.testing.assert_array_equal(
        leaflets["values"],
        np.array([[1, 1], [1, 1], [-1, -1], [-1, -1]], dtype=np.float32),
    )
    np.testing.assert_array_equal(leaflets["residue_ids"], [1, 2, 3, 4])

    zpos = warp_md.lipid_z_positions(_traj(coords, box), system, "all", "name H")
    np.testing.assert_allclose(zpos["values"][:, 0], [2.0, 2.0, -2.0, -2.0])

    zthick = warp_md.lipid_z_thickness(_traj(coords, box), system, "all")
    np.testing.assert_allclose(zthick["values"][:, 0], [1.0, 1.0, 1.0, 1.0])

    angles = warp_md.lipid_z_angles(_traj(coords, box), system, "name H", "name T")
    np.testing.assert_allclose(angles["values"][:, 0], [0.0, 0.0, 180.0, 180.0])


def test_lipid_curved_leaflets_components_and_midplane():
    coords = np.array(
        [
            [
                [80.0, 50.0, 50.0],
                [85.0, 50.0, 50.0],
                [45.0, 50.0, 50.0],
                [40.0, 50.0, 50.0],
                [62.0, 50.0, 50.0],
            ]
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(
        {
            "name": ["H"] * 5,
            "resname": ["LIP"] * 5,
            "resid": [1, 2, 3, 4, 5],
            "chain_id": [0] * 5,
            "mass": [1.0] * 5,
        },
        positions0=coords[0],
    )
    box = np.array([[100.0, 100.0, 100.0]], dtype=np.float32)
    out = warp_md.lipid_curved_leaflets(
        warp_md.Trajectory.from_numpy(coords, box=box),
        system,
        "name H",
        cutoff=8.0,
        midplane_selection="resid 5",
        midplane_cutoff=8.0,
    )
    np.testing.assert_array_equal(out["values"], np.array([[1.0], [1.0], [-1.0], [-1.0], [0.0]], dtype=np.float32))
    assert out["kind"] == "curved_leaflets"


def test_lipid_area_accepts_framewise_and_constant_leaflets():
    system, coords, box = _bilayer_system()
    leaflets = warp_md.lipid_leaflets(_traj(coords, box), system, "name H")["values"].astype(np.int8)

    framewise = warp_md.lipid_area(_traj(coords, box), system, "name H", leaflets)
    constant = warp_md.lipid_area(_traj(coords, box), system, "name H", leaflets[:, 0])

    np.testing.assert_allclose(framewise["values"], np.full((4, 2), 8.0, dtype=np.float32))
    np.testing.assert_allclose(constant["values"], framewise["values"])
    assert framewise["kind"] == "area_per_lipid"


def test_lipid_flip_flop_reports_events():
    system, coords, box = _bilayer_system()
    leaflets = np.array([[1, 1, 0, -1, -1]], dtype=np.int8)

    out = warp_md.lipid_flip_flop(
        warp_md.Trajectory.from_numpy(coords[:1], box=box[:1]),
        system,
        leaflets,
        residue_ids=[7],
        frame_cutoff=1,
    )

    np.testing.assert_array_equal(out["events"], np.array([[7, 1, 3, -1]], dtype=np.int32))
    np.testing.assert_array_equal(out["success"], np.array(["Success"], dtype=object))


def test_lipid_neighbours_cluster_thickness_registration_and_msd():
    system, coords, box = _bilayer_system()

    neighbours = warp_md.lipid_neighbours(_traj(coords, box), system, "name H", cutoff=1.5)
    np.testing.assert_allclose(neighbours["values"][:, 0], [0.0, 0.0, 0.0, 0.0])

    cluster = warp_md.lipid_largest_cluster(_traj(coords, box), system, "name H", cutoff=3.0)
    np.testing.assert_array_equal(cluster["values"], np.array([[2.0, 2.0]], dtype=np.float32))

    adjacency = warp_md.lipid_neighbour_matrix(_traj(coords, box), system, "name H", cutoff=3.0)
    assert adjacency["values"].shape == (2, 4, 4)
    np.testing.assert_array_equal(adjacency["values"][0].sum(axis=1), np.array([1, 1, 1, 1]))
    composition = warp_md.lipid_neighbour_composition(
        adjacency,
        np.array([0, 0, 1, 1], dtype=np.int8),
        label_names={"upper": 0, "lower": 1},
        return_enrichment=True,
    )
    np.testing.assert_array_equal(composition["counts"][0], np.array([[1, 0], [1, 0], [0, 1], [0, 1]]))
    assert composition["enrichment"].shape == (2, 4, 2)

    leaflets = warp_md.lipid_leaflets(_traj(coords, box), system, "name H")["values"].astype(np.int8)
    thickness = warp_md.lipid_membrane_thickness(_traj(coords, box), system, "name H", leaflets)
    np.testing.assert_allclose(thickness["values"][0, 0], 4.0)

    registration = warp_md.lipid_registration(_traj(coords, box), system, "name H", "name H", leaflets, bins=2)
    np.testing.assert_allclose(registration["values"], np.array([[-1.0, -1.0]], dtype=np.float32))

    msd_coords = np.array(
        [
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [10.0, 2.0, 0.0]],
            [[2.0, 0.0, 0.0], [10.0, 4.0, 0.0]],
        ],
        dtype=np.float32,
    )
    msd_system = warp_md.System.from_arrays(
        {
            "name": ["H", "H"],
            "resname": ["LIP", "LIP"],
            "resid": [1, 2],
            "chain_id": [0, 0],
            "mass": [1.0, 1.0],
        },
        positions0=msd_coords[0],
    )
    msd = warp_md.lipid_msd(warp_md.Trajectory.from_numpy(msd_coords), msd_system, "name H")
    np.testing.assert_array_equal(msd["values"], np.array([[0.0, 1.0, 4.0], [0.0, 4.0, 16.0]], dtype=np.float32))


def test_lipid_scc_and_weighted_average():
    coords = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 0.0, 1.0],
                [10.0, 0.0, 2.0],
            ]
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(
        {
            "name": ["T1", "T2", "T3", "T1", "T2", "T3"],
            "resname": ["LIP"] * 6,
            "resid": [1, 1, 1, 2, 2, 2],
            "chain_id": [0] * 6,
            "mass": [1.0] * 6,
        },
        positions0=coords[0],
    )
    box = np.array([[20.0, 20.0, 20.0]], dtype=np.float32)
    normals = np.array([[[0.0, 0.0, 1.0]], [[0.0, 0.0, -1.0]]], dtype=np.float32)

    scc = warp_md.lipid_scc(warp_md.Trajectory.from_numpy(coords, box=box), system, "name T1 T2 T3", normals=normals)
    np.testing.assert_allclose(scc["values"][:, 0], [-0.5, 1.0])

    avg = warp_md.lipid_scc_weighted_average(scc, scc, sn1_weight=2.0, sn2_weight=1.0)
    np.testing.assert_allclose(avg["values"], scc["values"])


def test_lipid_projection_and_joint_density_helpers():
    projection = warp_md.lipid_project_values(
        [0.25, 0.75, 1.25],
        [0.25, 0.75, 1.25],
        [1.0, 3.0, 5.0],
        bins=[np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])],
    )
    np.testing.assert_allclose(projection["statistic"], np.array([[2.0, np.nan], [np.nan, 5.0]], dtype=np.float32))
    np.testing.assert_array_equal(projection["counts"], np.array([[2, 0], [0, 1]]))
    assert projection["kind"] == "projection"

    filled = warp_md.lipid_project_values(
        [0.25, 1.25],
        [0.25, 1.25],
        [1.0, 5.0],
        bins=[np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])],
        interpolate="nearest",
        tile=False,
    )
    assert np.isfinite(filled["statistic"]).all()

    density = warp_md.lipid_joint_density(
        [10.0, 10.0, 20.0],
        [-1.0, 1.0, 1.0],
        bins=[np.array([0.0, 15.0, 30.0]), np.array([-2.0, 0.0, 2.0])],
        temperature=300.0,
    )
    np.testing.assert_allclose(np.nansum(density["density"]), 1.0)
    assert density["density"].shape == (2, 2)
    assert density["pmf"].shape == (2, 2)
