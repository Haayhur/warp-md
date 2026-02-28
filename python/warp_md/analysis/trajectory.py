# Usage:
# from warp_md.analysis.trajectory import ArrayTrajectory
# aligned = ArrayTrajectory(coords, box=box, time_ps=time_ps)
# chunk = aligned.read_chunk(256)

from __future__ import annotations

from typing import Optional

import numpy as np


class ArrayTrajectory:
    """Trajectory-like wrapper around in-memory arrays."""

    def __init__(
        self,
        coords,
        box: Optional[np.ndarray] = None,
        time_ps: Optional[np.ndarray] = None,
    ):
        self._coords = np.asarray(coords, dtype=np.float32)
        if self._coords.ndim != 3 or self._coords.shape[-1] != 3:
            raise ValueError("coords must be (n_frames, n_atoms, 3)")
        self._box = None if box is None else np.asarray(box, dtype=np.float32)
        self._time = None if time_ps is None else np.asarray(time_ps, dtype=np.float64)
        self._index = 0

    def reset(self):
        self._index = 0

    def n_atoms(self) -> int:
        return int(self._coords.shape[1])

    def n_frames(self) -> int:
        return int(self._coords.shape[0])

    def read_chunk(self, max_frames: int = 128):
        if self._index >= self._coords.shape[0]:
            return None
        max_frames = max(1, int(max_frames))
        end = min(self._index + max_frames, self._coords.shape[0])
        chunk = {"coords": self._coords[self._index:end]}
        if self._box is not None:
            chunk["box"] = self._box[self._index:end]
        if self._time is not None:
            chunk["time"] = self._time[self._index:end]
        self._index = end
        return chunk
