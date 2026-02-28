from __future__ import annotations

from typing import Optional


def read_chunk_fields(
    traj,
    max_frames: Optional[int],
    *,
    include_box: bool = False,
    include_box_matrix: bool = False,
    include_time: bool = False,
):
    """Read a chunk requesting only needed metadata fields when supported.

    Falls back to legacy `read_chunk(max_frames)` for dummy trajectories/tests
    that do not accept keyword arguments.
    """
    kwargs = {
        "include_box": include_box,
        "include_box_matrix": include_box_matrix,
        "include_time": include_time,
    }
    try:
        if max_frames is None:
            return traj.read_chunk(**kwargs)
        return traj.read_chunk(max_frames, **kwargs)
    except TypeError:
        if max_frames is None:
            return traj.read_chunk()
        return traj.read_chunk(max_frames)
