import numpy as np

from warp_md.analysis.lowestcurve import lowestcurve


def test_lowestcurve_bins_and_points():
    data = np.array(
        [
            [0.00, 1.00],
            [0.05, 2.00],
            [0.15, 0.50],
            [0.25, 5.00],
            [0.35, 1.50],
        ],
        dtype=np.float64,
    )
    out = lowestcurve(data, points=2, step=0.2)
    expected = np.array(
        [
            [0.0, 0.2],
            [0.75, 3.25],
        ],
        dtype=np.float32,
    )
    assert out.shape == expected.shape
    assert np.allclose(out, expected, atol=1e-6)


def test_lowestcurve_accepts_transposed():
    data = np.array(
        [
            [0.0, 0.2, 0.4],
            [1.0, 0.5, 2.0],
        ],
        dtype=np.float64,
    )
    out = lowestcurve(data, points=1, step=0.2)
    out_t = lowestcurve(data.T, points=1, step=0.2)
    assert np.allclose(out, out_t, atol=1e-6)
