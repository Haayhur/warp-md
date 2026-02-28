from __future__ import annotations

import numpy as np

from warp_md.analysis._stream import iter_frames


class _IntoOnlyTrajectory:
    def __init__(self, coords: np.ndarray):
        self._coords = np.asarray(coords, dtype=np.float32)
        self._cursor = 0
        self.calls_into = 0
        self.calls_chunk = 0

    def n_atoms(self) -> int:
        return int(self._coords.shape[1])

    def read_chunk_into(self, coords, box_out=None, time_out=None, max_frames=None) -> int:
        self.calls_into += 1
        cap = int(coords.shape[0])
        if max_frames is not None:
            cap = min(cap, int(max_frames))
        if self._cursor >= self._coords.shape[0]:
            return 0
        take = min(cap, self._coords.shape[0] - self._cursor)
        coords[:take] = self._coords[self._cursor : self._cursor + take]
        self._cursor += take
        return int(take)

    def read_chunk(self, _max_frames=128):
        self.calls_chunk += 1
        raise AssertionError("iter_frames should use read_chunk_into for coords-only streaming")


def test_iter_frames_uses_coords_only_read_chunk_into_fast_path():
    coords = np.array(
        [
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[3.0, 1.0, 0.0], [4.0, 1.0, 0.0]],
            [[5.0, 2.0, 0.0], [6.0, 2.0, 0.0]],
        ],
        dtype=np.float32,
    )
    traj = _IntoOnlyTrajectory(coords)

    got = list(iter_frames(traj, chunk_frames=2, include_box=False, include_time=False))

    assert len(got) == 3
    assert traj.calls_into >= 2
    assert traj.calls_chunk == 0
    for i, (_, frame_coords, box_row, t) in enumerate(got):
        np.testing.assert_allclose(frame_coords, coords[i].astype(np.float64))
        assert box_row is None
        assert t is None
