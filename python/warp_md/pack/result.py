"""Pack result container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class PackResult:
    coords: "numpy.ndarray"
    box: Tuple[float, float, float]
    name: List[str]
    element: List[str]
    resname: List[str]
    resid: List[int]
    chain: List[str]
    charge: List[float]
    mol_id: List[int]
    segid: Optional[List[str]] = None
    bonds: Optional[List[Tuple[int, int]]] = None
    record_kind: Optional[List[str]] = None
    ter_after: Optional[List[int]] = None
