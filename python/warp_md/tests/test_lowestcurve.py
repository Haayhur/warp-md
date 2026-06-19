import numpy as np

import warp_md
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


def test_lowestcurve_uses_native_array_kernel_when_available(monkeypatch):
    called = {}

    def fake_native(xy, points, step):
        called["shape"] = xy.shape
        called["points"] = points
        called["step"] = step
        return np.array([[0.0], [4.0]], dtype=np.float32)

    monkeypatch.setattr(warp_md, "lowestcurve_array", fake_native, raising=False)
    out = lowestcurve(np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64), points=3, step=0.5)
    np.testing.assert_allclose(out, np.array([[0.0], [4.0]], dtype=np.float32))
    assert called == {"shape": (2, 2), "points": 3, "step": 0.5}
