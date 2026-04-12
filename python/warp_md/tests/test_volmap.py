import numpy as np

import warp_md
from warp_md.analysis.volmap import volmap


def test_volmap_returns_native_grid_output():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [0.4, 0.4, 0.4], [1.1, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.8, 0.2, 0.2], [0.9, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    system = warp_md.System.from_arrays(
        {
            "name": ["C1", "C2", "C3"],
            "resname": ["RES"] * 3,
            "resid": [1, 2, 3],
            "chain_id": [0] * 3,
            "mass": [1.0] * 3,
        },
        positions0=coords[0],
    )

    out = volmap(
        warp_md.Trajectory.from_numpy(coords),
        system,
        "all",
        "resid 1:1",
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    )
    assert sorted(out) == ["dims", "first", "last", "max", "mean", "min", "std"]
    assert out["dims"] == (2, 2, 2)
    assert out["mean"].shape == (8,)
    np.testing.assert_allclose(out["mean"][0], 2.5, atol=1e-6)
    np.testing.assert_allclose(out["std"][0], 0.5, atol=1e-6)
