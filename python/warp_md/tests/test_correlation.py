import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

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
