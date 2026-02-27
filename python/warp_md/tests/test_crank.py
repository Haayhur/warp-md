import importlib

import numpy as np
import pytest

crank_mod = importlib.import_module("warp_md.analysis.crank")


def test_crank_distance(monkeypatch):
    def _kernel(a, b, mode="distance"):
        assert mode == "distance"
        return b - a

    monkeypatch.setattr(crank_mod, "_crank_kernel", _kernel, raising=True)
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.5, 1.0, 3.5])
    out = crank_mod.crank(a, b, mode="distance")
    assert np.allclose(out, b - a)


def test_crank_angle_wrap(monkeypatch):
    def _kernel(a, b, mode="distance"):
        assert mode == "angle"
        delta = b - a
        return (delta + 180.0) % 360.0 - 180.0

    monkeypatch.setattr(crank_mod, "_crank_kernel", _kernel, raising=True)
    a = np.array([170.0, -170.0, 10.0])
    b = np.array([-170.0, 170.0, -170.0])
    out = crank_mod.crank(a, b, mode="angle")
    assert np.all(out <= 180.0)
    assert np.all(out >= -180.0)


def test_crank_raises_when_binding_missing(monkeypatch):
    monkeypatch.setattr(crank_mod, "_crank_kernel", None, raising=True)
    with pytest.raises(RuntimeError, match="crank_delta binding unavailable"):
        crank_mod.crank(np.array([0.0]), np.array([1.0]))
