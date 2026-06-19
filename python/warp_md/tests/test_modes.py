import numpy as np
import warp_md

from warp_md.analysis.modes import _corr, _trajout, analyze_modes


def test_modes_fluct_eigenval_rmsip():
    evals = np.array([2.0, 1.0], dtype=np.float64)
    evecs = np.zeros((2, 6), dtype=np.float64)
    evecs[0, 0] = 1.0
    evecs[1, 3] = 1.0

    out = analyze_modes("fluct", evecs, evals, scalar_type="covar", dtype="dict")
    rms = out["FLUCT[rms]"]
    assert rms.shape == (2,)
    np.testing.assert_allclose(rms, np.array([np.sqrt(2.0), 1.0], dtype=np.float32), rtol=1e-6)

    out2 = analyze_modes("eigenval", evecs, evals, scalar_type="covar", dtype="dict")
    frac = out2["EIGENVAL[Frac]"]
    np.testing.assert_allclose(frac, np.array([2.0 / 3.0, 1.0 / 3.0], dtype=np.float32), rtol=1e-6)

    rmsip = analyze_modes("rmsip", evecs, evals, scalar_type="covar", dtype="ndarray")
    assert 0.0 <= rmsip <= 1.0


def test_modes_trajout_and_corr():
    evals = np.array([1.0], dtype=np.float64)
    evecs = np.array([[1.0, 0.0, 0.0, -1.0, 0.0, 0.0]], dtype=np.float64)
    avg = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    traj = analyze_modes(
        "trajout",
        evecs,
        evals,
        scalar_type="covar",
        options="pcmin -1 pcmax 1 nframes 3 tmode 1",
        average_coords=avg,
    )
    chunk = traj.read_chunk(10)
    coords = chunk["coords"]
    assert coords.shape == (3, 2, 3)

    corr = analyze_modes(
        "corr",
        evecs,
        evals,
        scalar_type="covar",
        average_coords=avg,
        mask_pairs=[(0, 1)],
        dtype="ndarray",
    )
    assert corr.shape == (1, 1)


def test_modes_native_corr_route(monkeypatch):
    calls = []

    def fake_native(vecs, average_coords, pairs):
        calls.append((vecs.copy(), average_coords.copy(), pairs.copy()))
        return np.array([[-2.0]], dtype=np.float32)

    monkeypatch.setattr(warp_md, "mode_corr_array", fake_native, raising=False)
    fake_native.__name__ = "mode_corr_array"

    vecs = np.array([[1.0, 0.0, 0.0, -1.0, 0.0, 0.0]], dtype=np.float64)
    avg = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    out = _corr(vecs, avg, [(0, 1)])

    assert len(calls) == 1
    assert calls[0][2].dtype == np.int64
    assert np.allclose(out, [[-2.0]])


def test_modes_native_trajout_route(monkeypatch):
    calls = []

    def fake_native(average_coords, mode_vec, pcmin, pcmax, nframes, factor):
        calls.append((average_coords.copy(), mode_vec.copy(), pcmin, pcmax, nframes, factor))
        return np.zeros((3, 2, 3), dtype=np.float64)

    monkeypatch.setattr(warp_md, "mode_trajout_array", fake_native, raising=False)
    fake_native.__name__ = "mode_trajout_array"

    avg = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    mode = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float64)
    out = _trajout(avg, mode, -1.0, 1.0, 3, 2.0)

    assert len(calls) == 1
    assert calls[0][4] == 3
    assert calls[0][5] == 2.0
    assert out.shape == (3, 2, 3)
