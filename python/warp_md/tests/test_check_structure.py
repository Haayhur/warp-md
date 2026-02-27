import numpy as np

from warp_md.analysis.check_structure import check_structure


class _DummyTraj:
    def __init__(self, coords):
        self._coords = coords
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        return {"coords": self._coords}


def test_check_structure_counts():
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = _DummyTraj(coords)
    counts, report = check_structure(traj, system=None)
    np.testing.assert_allclose(counts, np.array([1, 0], dtype=np.int64))
    assert "frame 0" in report
