import numpy as np

from warp_md.analysis.modes import analyze_modes


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
