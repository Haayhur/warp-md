import numpy as np

from warp_md.analysis.lipidscd import lipidscd


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self):
        self._atoms = {
            "name": ["C1", "C2", "C1", "C2"],
            "resname": ["LIP", "LIP", "LIP", "LIP"],
            "resid": [1, 1, 2, 2],
            "chain_id": [0, 0, 0, 0],
            "mass": [1.0, 1.0, 1.0, 1.0],
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


def test_lipidscd_pair_mode():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem()

    out_res = lipidscd(traj, system, selection="all", pair_mode="residue")
    assert out_res["bond_indices"].shape[0] == 2

    traj = _DummyTraj(coords)
    out_global = lipidscd(traj, system, selection="all", pair_mode="global")
    assert out_global["bond_indices"].shape[0] == 3


def test_lipidscd_frame_indices():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem()

    out = lipidscd(
        traj,
        system,
        selection="all",
        pair_mode="residue",
        per_frame=True,
        frame_indices=[1],
    )
    assert out["scd"].shape[0] == 1
