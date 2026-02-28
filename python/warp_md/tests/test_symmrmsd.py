import numpy as np

from warp_md.analysis.symmrmsd import symmrmsd


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
    def __init__(self, coords):
        self._coords = coords
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        return {"coords": self._coords}


def test_symmrmsd_basic():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    out = symmrmsd(traj, system, mask="all", ref=0, fit=False)
    np.testing.assert_allclose(out, np.array([0.0, 0.70710677], dtype=np.float32), rtol=1e-6)


def test_symmrmsd_remap_swapped_atoms():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem(coords.shape[1])

    out_no = symmrmsd(traj, system, mask="all", ref=0, fit=False, remap=False)
    assert out_no[1] > 0.0

    traj2 = _DummyTraj(coords)
    out_yes = symmrmsd(
        traj2,
        system,
        mask="all",
        ref=0,
        fit=False,
        remap=True,
        symmetry_groups=[[1, 2]],
    )
    np.testing.assert_allclose(out_yes, np.array([0.0, 0.0], dtype=np.float32), atol=1e-6)
