import tempfile

import numpy as np
import pytest

import warp_md
from warp_md.analysis.nmr import ired_vector_and_matrix, jcoupling, nh_order_parameters


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms):
        self._atoms = {
            "name": ["N", "H"] * (n_atoms // 2),
            "resname": ["ALA"] * n_atoms,
            "resid": [1] * n_atoms,
            "chain_id": [0] * n_atoms,
        }

    def atom_table(self):
        return self._atoms

    def select(self, _mask):
        return _DummySelection(range(len(self._atoms["name"])))

    def select_indices(self, indices):
        return _DummySelection(indices)


class _DummyTraj:
    def __init__(self, coords, box=None):
        self._coords = coords
        self._box = box
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        out = {"coords": self._coords}
        if self._box is not None:
            out["box"] = self._box
        return out


def _normalize_frame_indices(frame_indices, n_frames):
    if frame_indices is None:
        return list(range(n_frames))
    out = []
    for raw in frame_indices:
        idx = int(raw)
        if idx < 0:
            idx += n_frames
        if 0 <= idx < n_frames:
            out.append(idx)
    return out


def _dihedral(p0, p1, p2, p3):
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / (np.linalg.norm(b1) + 1.0e-12)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return np.arctan2(y, x)


class _DefaultNmrPlan:
    def __init__(
        self,
        pairs,
        order=2,
        length_scale=0.1,
        pbc="none",
        corr_mode="tensor",
        return_corr=True,
    ):
        del length_scale, pbc
        self.pairs = [(int(a), int(b)) for a, b in pairs]
        self.order = int(order)
        self.corr_mode = corr_mode
        self.return_corr = bool(return_corr)

    def run(self, traj, system, chunk_frames=None, device="auto", frame_indices=None):
        del system, chunk_frames, device
        coords = np.asarray(traj._coords, dtype=np.float64)
        rows = _normalize_frame_indices(frame_indices, coords.shape[0])
        coords = coords[np.asarray(rows, dtype=np.int64)]
        n_frames = coords.shape[0]
        n_pairs = len(self.pairs)
        vecs = np.zeros((n_frames, n_pairs, 3), dtype=np.float32)
        for f in range(n_frames):
            for p, (a, b) in enumerate(self.pairs):
                v = coords[f, b] - coords[f, a]
                norm = float(np.linalg.norm(v))
                if norm > 0.0:
                    vecs[f, p] = (v / norm).astype(np.float32)
        if not self.return_corr:
            return vecs
        if self.corr_mode == "tensor":
            flat = vecs.reshape(n_frames, n_pairs * 3).astype(np.float64)
            mat = (flat.T @ flat) / float(max(n_frames, 1))
            if self.order == 2 and mat.size > 0 and mat[0, 0] != 0.0:
                mat = mat / mat[0, 0]
            return vecs, mat.astype(np.float32)
        corr = np.zeros((n_frames, n_pairs), dtype=np.float32)
        for lag in range(n_frames):
            span = n_frames - lag
            for p in range(n_pairs):
                dot = np.sum(vecs[:span, p, :] * vecs[lag:, p, :], axis=1, dtype=np.float64)
                if self.order == 2:
                    dot = 1.5 * dot * dot - 0.5
                corr[lag, p] = float(np.mean(dot)) if dot.size else 0.0
        return vecs, corr


class _DefaultMultiDihedralPlan:
    def __init__(self, groups, mass_weighted=False, pbc="none", degrees=True, range360=False):
        del mass_weighted, pbc, degrees, range360
        self.groups = groups

    def run(self, traj, system, chunk_frames=None, device="auto", frame_indices=None):
        del system, chunk_frames, device
        coords = np.asarray(traj._coords, dtype=np.float64)
        rows = _normalize_frame_indices(frame_indices, coords.shape[0])
        coords = coords[np.asarray(rows, dtype=np.int64)]
        n_frames = coords.shape[0]
        n_groups = len(self.groups)
        out = np.zeros((n_frames, n_groups), dtype=np.float32)
        for f in range(n_frames):
            for g, (sa, sb, sc, sd) in enumerate(self.groups):
                a = int(list(sa.indices)[0])
                b = int(list(sb.indices)[0])
                c = int(list(sc.indices)[0])
                d = int(list(sd.indices)[0])
                out[f, g] = float(_dihedral(coords[f, a], coords[f, b], coords[f, c], coords[f, d]))
        return out


@pytest.fixture(autouse=True)
def _mock_rust_plans(monkeypatch):
    monkeypatch.setattr(warp_md, "NmrIredPlan", _DefaultNmrPlan, raising=False)
    monkeypatch.setattr(warp_md, "MultiDihedralPlan", _DefaultMultiDihedralPlan, raising=False)


def test_ired_frame_indices():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    vecs, mat = ired_vector_and_matrix(
        traj,
        system,
        selection="all",
        frame_indices=[1],
    )
    assert vecs.shape[0] == 1
    assert mat.shape[0] == 3


def test_nh_order_parameters_shape():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    s2 = nh_order_parameters(traj, system, selection="all", frame_indices=[0, 1])
    assert s2.shape == (1,)


def test_jcoupling_kfile():
    coords = np.array(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    with tempfile.NamedTemporaryFile("w+", delete=False) as handle:
        handle.write("1.0 2.0 3.0\n")
        kfile = handle.name
    values = jcoupling(
        traj,
        system,
        dihedral_indices=[(0, 1, 2, 3)],
        kfile=kfile,
    )
    assert values.shape == (1, 1)
    assert np.allclose(values[0, 0], 6.0, atol=1e-5)


def test_jcoupling_frame_indices():
    coords = np.array(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    values = jcoupling(
        traj,
        system,
        dihedral_indices=[(0, 1, 2, 3)],
        frame_indices=[1],
    )
    assert values.shape == (1, 1)


def test_ired_timecorr_mode():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    vecs, corr = ired_vector_and_matrix(
        traj,
        system,
        selection="all",
        corr_mode="timecorr",
        order=2,
    )
    assert vecs.shape == (3, 1, 3)
    assert corr.shape == (3, 1)
    np.testing.assert_allclose(corr[:, 0], np.ones(3), atol=1e-6)


def test_nh_order_parameters_timecorr_fit():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    s2 = nh_order_parameters(traj, system, selection="all", method="timecorr_fit")
    assert s2.shape == (1,)
    np.testing.assert_allclose(s2[0], 1.0, atol=1e-5)


def test_nh_order_parameters_timecorr_fit_rejects_order_not_two():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    with pytest.raises(ValueError, match="supports order=2 only"):
        nh_order_parameters(traj, system, selection="all", method="timecorr_fit", order=1)


def test_nh_order_parameters_timecorr_fit_rejects_nonpositive_timing():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    with pytest.raises(ValueError, match="requires tstep > 0"):
        nh_order_parameters(traj, system, selection="all", method="timecorr_fit", tstep=0.0)
    with pytest.raises(ValueError, match="requires tcorr > 0"):
        nh_order_parameters(traj, system, selection="all", method="timecorr_fit", tcorr=0.0)


def test_jcoupling_return_dihedral():
    coords = np.array(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    jvals, angles = jcoupling(
        traj,
        system,
        dihedral_indices=[(0, 1, 2, 3)],
        return_dihedral=True,
    )
    assert jvals.shape == (2, 1)
    assert angles.shape == (2, 1)


def test_jcoupling_uses_multi_dihedral_plan(monkeypatch):
    coords = np.array(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyMultiPlan:
        def __init__(self, groups, mass_weighted=False, pbc="none", degrees=True, range360=False):
            called["groups"] = len(groups)
            called["pbc"] = pbc
            called["degrees"] = degrees
            called["range360"] = range360

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            del frame_indices
            return np.array([[0.0], [np.pi / 2.0]], dtype=np.float32)

    monkeypatch.setattr(warp_md, "MultiDihedralPlan", _DummyMultiPlan, raising=False)
    vals = jcoupling(
        traj,
        system,
        dihedral_indices=[(0, 1, 2, 3)],
        karplus=(1.0, 0.0, 0.0),
    )
    np.testing.assert_allclose(vals[:, 0], np.array([1.0, 0.0], dtype=np.float32), atol=1e-6)
    assert called["groups"] == 1
    assert called["pbc"] == "none"
    assert called["degrees"] is False
    assert called["range360"] is False


def test_jcoupling_frame_indices_forwarded_to_plan(monkeypatch):
    coords = np.array(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyMultiPlan:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["frame_indices"] = frame_indices
            assert frame_indices == [1, -1]
            return np.array([[0.0], [np.pi / 2.0]], dtype=np.float32)

    monkeypatch.setattr(warp_md, "MultiDihedralPlan", _DummyMultiPlan, raising=False)
    vals = jcoupling(
        traj,
        system,
        dihedral_indices=[(0, 1, 2, 3)],
        karplus=(1.0, 0.0, 0.0),
        frame_indices=[1, -1],
    )
    assert vals.shape == (2, 1)
    assert called["frame_indices"] == [1, -1]


def test_jcoupling_no_legacy_fallback_when_binding_lacks_frame_indices(monkeypatch):
    coords = np.array(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    class _LegacyMultiPlan:
        def __init__(self, *args, **kwargs):
            pass

        # Intentionally no frame_indices keyword to emulate old binding.
        def run(self, _traj, _system, chunk_frames=None, device="auto"):
            return np.array([[0.0]], dtype=np.float32)

    monkeypatch.setattr(warp_md, "MultiDihedralPlan", _LegacyMultiPlan, raising=False)
    with pytest.raises(RuntimeError, match="jcoupling requires Rust-backed trajectory/system objects"):
        jcoupling(
            traj,
            system,
            dihedral_indices=[(0, 1, 2, 3)],
            frame_indices=[0],
        )


def test_ired_uses_rust_plan(monkeypatch):
    coords = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float64)
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            pairs,
            order=2,
            length_scale=0.1,
            pbc="none",
            corr_mode="tensor",
            return_corr=True,
        ):
            called["pairs"] = pairs
            called["order"] = order
            called["length_scale"] = length_scale
            called["pbc"] = pbc
            called["corr_mode"] = corr_mode
            called["return_corr"] = return_corr

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            del frame_indices
            vecs = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
            corr = np.eye(3, dtype=np.float32)
            return vecs, corr

    monkeypatch.setattr(warp_md, "NmrIredPlan", _DummyPlan, raising=False)
    vecs, corr = ired_vector_and_matrix(
        traj,
        system,
        selection="all",
        order=2,
        length_scale=0.1,
        pbc="none",
        corr_mode="tensor",
    )
    assert vecs.shape == (1, 1, 3)
    assert corr.shape == (3, 3)
    assert called["pairs"] == [(0, 1)]
    assert called["order"] == 2
    assert called["corr_mode"] == "tensor"
    assert called["return_corr"] is True


def test_ired_frame_indices_uses_plan(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])
    called = {}

    class _DummyPlan:
        def __init__(self, *args, **kwargs):
            called["created"] = True

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            assert frame_indices == [1]
            vecs = np.array(
                [
                    [[0.0, 1.0, 0.0]],
                ],
                dtype=np.float32,
            )
            corr = np.eye(3, dtype=np.float32)
            return vecs, corr

    monkeypatch.setattr(warp_md, "NmrIredPlan", _DummyPlan, raising=False)
    vecs, corr = ired_vector_and_matrix(
        traj,
        system,
        selection="all",
        frame_indices=[1],
        corr_mode="tensor",
    )
    assert called["created"] is True
    assert vecs.shape[0] == 1
    assert corr.shape == (3, 3)
    np.testing.assert_allclose(vecs[0, 0], np.array([0.0, 1.0, 0.0]), atol=1e-6)
