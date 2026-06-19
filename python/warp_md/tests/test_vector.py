import importlib

import numpy as np

import warp_md
from warp_md.analysis.vector import _apply_pbc, vector, vector_mask

vector_mod = importlib.import_module("warp_md.analysis.vector")


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self, n_atoms):
        self._atoms = {
            "name": ["A"] * n_atoms,
            "resname": ["RES"] * n_atoms,
            "resid": list(range(1, n_atoms + 1)),
            "chain_id": [0] * n_atoms,
            "mass": [1.0] * n_atoms,
        }

    def atom_table(self):
        return self._atoms

    def select(self, _mask):
        return _DummySelection(range(len(self._atoms["name"])))


class _DummyTraj:
    def __init__(self, coords, box=None):
        self._coords = coords
        self._box = box
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        chunk = {"coords": self._coords}
        if self._box is not None:
            chunk["box"] = self._box
        return chunk


def test_vector_mask_basic():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    out = vector(traj, system, "@1 @2")
    np.testing.assert_allclose(out[:, 0], np.array([1.0, 2.0], dtype=np.float32), rtol=1e-6)

    traj2 = _DummyTraj(coords)
    out2 = vector_mask(traj2, system, np.array([[0, 1]], dtype=np.int64))
    np.testing.assert_allclose(out2[0, :, 0], np.array([1.0, 2.0], dtype=np.float32), rtol=1e-6)


def test_apply_pbc_wraps_orthorhombic_vectors():
    vec = np.array([[6.0, -6.0, 2.0], [2.5, 7.6, -7.6]], dtype=np.float64)
    box = np.array([[10.0, 10.0, 0.0], [5.0, 5.0, 5.0]], dtype=np.float64)
    out = _apply_pbc(vec, box)
    np.testing.assert_allclose(
        out,
        np.array([[-4.0, 4.0, 2.0], [2.5, -2.4, 2.4]], dtype=np.float64),
        atol=1e-12,
    )


def test_apply_pbc_uses_native_array_kernel_when_available(monkeypatch):
    called = {}

    def fake_native(vec, box):
        called["vec_shape"] = vec.shape
        called["box_shape"] = box.shape
        called["vec_dtype"] = vec.dtype
        called["box_dtype"] = box.dtype
        return np.array([[3.0, 2.0, 1.0]], dtype=np.float64)

    monkeypatch.setattr(warp_md, "apply_orthorhombic_pbc_vectors", fake_native, raising=False)
    out = _apply_pbc(
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        np.array([[10.0, 10.0, 10.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(out, np.array([[3.0, 2.0, 1.0]], dtype=np.float64))
    assert called == {
        "vec_shape": (1, 3),
        "box_shape": (1, 3),
        "vec_dtype": np.dtype("float64"),
        "box_dtype": np.dtype("float64"),
    }


def test_vector_mask_native_frame_indices():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = warp_md.Trajectory.from_numpy(coords)
    system = warp_md.System.from_arrays(_DummySystem(coords.shape[1]).atom_table(), positions0=coords[0])

    out = vector(traj, system, "@1 @2", frame_indices=[0, 2], chunk_frames=1)

    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[:, 0], np.array([1.0, 4.0], dtype=np.float32), rtol=1e-6)


def test_vector_center_native_does_not_materialize_in_python(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    atom_table = _DummySystem(coords.shape[1]).atom_table()
    atom_table["mass"] = [1.0, 3.0]
    system = warp_md.System.from_arrays(atom_table, positions0=coords[0])

    def _fail_read_all(*args, **kwargs):
        raise AssertionError("native vector center path should stream through Rust")

    monkeypatch.setattr(vector_mod, "read_all_frames", _fail_read_all)
    out = vector_mod.vector(
        warp_md.Trajectory.from_numpy(coords),
        system,
        "center @1 @2 mass",
        frame_indices=[2, 0],
        chunk_frames=1,
    )

    expected = np.array(
        [
            [4.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_vector_batch_native_uses_single_stream(monkeypatch):
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    atom_table = _DummySystem(coords.shape[1]).atom_table()
    atom_table["mass"] = [1.0, 3.0]
    system = warp_md.System.from_arrays(atom_table, positions0=coords[0])
    traj = warp_md.Trajectory.from_numpy(coords)

    def _fail_read_all(*args, **kwargs):
        raise AssertionError("native vector batch path should stream through Rust")

    reset_count = {"value": 0}
    original_reset = vector_mod.reset_traj

    def _count_reset(source):
        reset_count["value"] += 1
        return original_reset(source)

    monkeypatch.setattr(vector_mod, "read_all_frames", _fail_read_all)
    monkeypatch.setattr(vector_mod, "reset_traj", _count_reset)
    out = vector_mod.vector(
        traj,
        system,
        ["@1 @2", "center @1 @2 mass"],
        frame_indices=[2, 0],
        chunk_frames=1,
    )

    expected = np.array(
        [
            [[6.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[4.5, 0.0, 0.0], [1.5, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    assert reset_count["value"] == 1
    assert out.shape == (2, 2, 3)
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_vector_boxcenter():
    coords = np.zeros((1, 1, 3), dtype=np.float32)
    box = np.array([[4.0, 6.0, 8.0]], dtype=np.float32)
    traj = _DummyTraj(coords, box=box)
    system = _DummySystem(coords.shape[1])

    out = vector(traj, system, "boxcenter")
    np.testing.assert_allclose(out[0], np.array([2.0, 3.0, 4.0], dtype=np.float32), rtol=1e-6)
