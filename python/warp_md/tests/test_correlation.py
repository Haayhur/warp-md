import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

import warp_md
from warp_md.analysis.correlation import acorr, timecorr, xcorr


def test_acorr_includes_lag0():
    x = np.array([1.0, 2.0, 3.0], dtype=float)
    out = acorr(x, normalize=False)
    assert out.shape[0] == 3
    assert abs(out[0] - (1.0**2 + 2.0**2 + 3.0**2) / 3.0) < 1e-6


def test_xcorr_basic():
    a = np.array([1.0, 2.0, 3.0], dtype=float)
    b = np.array([2.0, 1.0, 0.0], dtype=float)
    out = xcorr(a, b, normalize=False)
    assert out.shape[0] == 3
    assert abs(out[0] - ((1.0 * 2.0 + 2.0 * 1.0 + 3.0 * 0.0) / 3.0)) < 1e-6


def test_timecorr_vectors():
    v = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    out = timecorr(v, order=1, normalize=True)
    assert out.shape[0] == 2
    assert abs(out[0] - 1.0) < 1e-6


def test_acorr_uses_native_array_kernel_when_available(monkeypatch):
    called = {}

    def fake_native(a, b, normalize):
        called["a_shape"] = a.shape
        called["same"] = a is b
        called["normalize"] = normalize
        return np.array([7.0, 3.0], dtype=np.float32)

    monkeypatch.setattr(warp_md, "array_cross_correlation", fake_native, raising=False)
    out = acorr(np.array([1.0, 2.0], dtype=float), normalize=False)
    np.testing.assert_allclose(out, np.array([7.0, 3.0], dtype=np.float32))
    assert called == {"a_shape": (2, 1), "same": True, "normalize": False}


def test_timecorr_uses_native_array_kernel_when_available(monkeypatch):
    called = {}

    def fake_native(vectors, order, normalize):
        called["shape"] = vectors.shape
        called["order"] = order
        called["normalize"] = normalize
        return np.array([1.0, 0.5], dtype=np.float32)

    monkeypatch.setattr(warp_md, "array_time_correlation", fake_native, raising=False)
    out = timecorr(np.ones((2, 3), dtype=float), order=2, normalize=False)
    np.testing.assert_allclose(out, np.array([1.0, 0.5], dtype=np.float32))
    assert called == {"shape": (2, 1, 3), "order": 2, "normalize": False}
