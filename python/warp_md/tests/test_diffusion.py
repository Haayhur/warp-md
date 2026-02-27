import numpy as np
import pytest

import warp_md
from warp_md.analysis.diffusion import diffusion


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms):
        self._atoms = {
            "name": ["A"] * n_atoms,
            "resid": list(range(1, n_atoms + 1)),
        }

    def atom_table(self):
        return self._atoms

    def select(self, _mask):
        return _DummySelection(range(len(self._atoms["name"])))


class _DummyTraj:
    def __init__(self, with_time=False):
        self._with_time = with_time
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        chunk = {"coords": np.zeros((1, 1, 3), dtype=np.float32)}
        if self._with_time:
            chunk["time_ps"] = np.array([0.0], dtype=np.float32)
        return chunk

    def reset(self):
        self._used = False


def test_diffusion_uses_msd_plan(monkeypatch):
    called = {}

    class _DummyPlan:
        def __init__(self, _sel, group_by="resid", lag_mode=None, group_types=None):
            called["group_by"] = group_by
            called["lag_mode"] = lag_mode
            called["group_types"] = group_types

        def run(
            self,
            _traj,
            _system,
            chunk_frames=None,
            device="auto",
            frame_indices=None,
        ):
            called["chunk_frames"] = chunk_frames
            called["device"] = device
            called["frame_indices"] = frame_indices
            time = np.array([1.0, 2.0], dtype=np.float32)
            data = np.array(
                [
                    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                    [4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0],
                ],
                dtype=np.float32,
            )
            return time, data

    monkeypatch.setattr(warp_md, "MsdPlan", _DummyPlan, raising=False)
    out = diffusion(
        _DummyTraj(with_time=False),
        _DummySystem(1),
        mask="all",
        tstep=2.0,
        frame_indices=[1, -1],
        chunk_frames=32,
    )
    assert called["group_by"] == "atom"
    assert called["lag_mode"] == "fft"
    assert called["group_types"] is None
    assert called["chunk_frames"] == 32
    assert called["device"] == "auto"
    assert called["frame_indices"] == [1, -1]
    np.testing.assert_allclose(out["time"], np.array([0.0, 2.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(out["X"], np.array([0.0, 1.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(out["Y"], np.array([0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(out["Z"], np.array([0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(out["MSD"], np.array([0.0, 1.0, 4.0], dtype=np.float32))
    assert np.isclose(out["D"], 0.25)


def test_diffusion_individual_uses_group_types(monkeypatch):
    called = {}

    class _DummyPlan:
        def __init__(self, _sel, group_by="resid", lag_mode=None, group_types=None):
            called["group_by"] = group_by
            called["lag_mode"] = lag_mode
            called["group_types"] = group_types

        def run(
            self,
            _traj,
            _system,
            chunk_frames=None,
            device="auto",
            frame_indices=None,
        ):
            time = np.array([1.0, 2.0], dtype=np.float32)
            data = np.array(
                [
                    [1.0, 9.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 9.0, 5.0],
                    [4.0, 16.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 16.0, 10.0],
                ],
                dtype=np.float32,
            )
            return time, data

    monkeypatch.setattr(warp_md, "MsdPlan", _DummyPlan, raising=False)
    out = diffusion(_DummyTraj(with_time=False), _DummySystem(2), mask="all", individual=True)
    assert called["group_by"] == "atom"
    assert called["lag_mode"] == "fft"
    assert called["group_types"] == [0, 1]
    np.testing.assert_allclose(out["MSD"], np.array([0.0, 5.0, 10.0], dtype=np.float32))
    np.testing.assert_allclose(
        out["MSD_individual"],
        np.array([[0.0, 1.0, 4.0], [0.0, 9.0, 16.0]], dtype=np.float32),
    )


def test_diffusion_raises_when_binding_missing(monkeypatch):
    class _Missing:
        pass

    monkeypatch.setattr(warp_md, "MsdPlan", _Missing, raising=False)
    with pytest.raises(RuntimeError, match="MsdPlan binding unavailable"):
        diffusion(_DummyTraj(with_time=False), _DummySystem(1), mask="all")
