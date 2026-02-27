# Usage:
# from warp_md.analysis.permute_dihedrals import permute_dihedrals
# permute_dihedrals(traj, system, "out.npy", dihedral_types="phi psi")

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from .multidihedral import multidihedral


def permute_dihedrals(
    traj,
    system,
    filename: str,
    dihedral_types: Optional[str] = None,
    resrange: Optional[Union[str, Sequence[int]]] = None,
    range360: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
):
    """Compute multidihedral and save to a file (npz).

    This is a simplified stand-in for cpptraj permutedihedrals.
    """
    if not filename:
        raise ValueError("filename is required")
    data = multidihedral(
        traj,
        system,
        dihedral_types=dihedral_types,
        resrange=resrange,
        range360=range360,
        frame_indices=frame_indices,
        dtype="dict",
    )
    np.savez(filename, **data)
    return None


__all__ = ["permute_dihedrals"]
