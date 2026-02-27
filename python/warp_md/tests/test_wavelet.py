import numpy as np

from warp_md.analysis.wavelet import wavelet


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self):
        self._atoms = {
            "name": ["A", "B"],
            "resname": ["RES", "RES"],
            "resid": [1, 1],
            "chain_id": [0, 0],
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


def test_wavelet_details():
    # distances: 0,2,4,6
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem()
    details, series = wavelet(
        traj,
        system,
        selection_a=[0],
        selection_b=[1],
        return_series=True,
        length_scale=1.0,
    )
    assert series.shape == (4,)
    assert np.allclose(series, [0.0, 2.0, 4.0, 6.0])
    assert details.shape == (2, 2)
    assert np.allclose(details[0], [-1.0, -1.0])
    assert np.allclose(details[1, 0], -2.0)
