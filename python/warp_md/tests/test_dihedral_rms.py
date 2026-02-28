import numpy as np

from warp_md.analysis.dihedral_rms import dihedral_rms
from warp_md.analysis.geometry import dihedral


class _DummyTraj:
    def __init__(self, coords):
        self._coords = coords
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        return {"coords": self._coords}


class _DummySystem:
    def atom_table(self):
        return {"resid": [1, 1, 1, 1]}

    def select(self, _mask):
        class Sel:
            indices = [0, 1, 2, 3]

        return Sel()


def test_dihedral_rms_basic():
    coords = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
        ],
        dtype=np.float64,
    )
    system = _DummySystem()

    vals = dihedral(_DummyTraj(coords), system, "@1 @2 @3 @4")
    rms = dihedral_rms(_DummyTraj(coords), system, "@1 @2 @3 @4", ref=0)
    assert rms.shape == (2,)
    assert np.isclose(rms[0], 0.0, atol=1e-6)
    delta = (vals - vals[0] + 180.0) % 360.0 - 180.0
    assert np.isclose(rms[1], np.abs(delta[1]), atol=1e-5)
