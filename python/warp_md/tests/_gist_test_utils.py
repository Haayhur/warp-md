import sys
from pathlib import Path

import numpy as np
import pytest
import warp_md

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))


def _require_openmm():
    try:
        from openmm import openmm as mm
        from openmm.app import Topology, element
    except Exception:  # pragma: no cover
        pytest.skip("openmm not available")
    return mm, Topology, element


class _FakeTraj:
    def __init__(self, coords, box=None, box_matrix=None):
        self.coords = coords
        self.box = box
        self.box_matrix = box_matrix
        self._used = False
        self.reset_calls = 0

    def reset(self):
        self._used = False
        self.reset_calls += 1

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        chunk = {"coords": self.coords}
        if self.box is not None:
            chunk["box"] = self.box
        if self.box_matrix is not None:
            chunk["box_matrix"] = self.box_matrix
        return chunk


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def select(self, _mask):
        return _DummySelection([0])

    def select_indices(self, indices):
        return _DummySelection([int(i) for i in indices])


def _build_system_and_topology():
    mm, Topology, element = _require_openmm()
    top = Topology()
    chain = top.addChain()
    res_sol = top.addResidue("LIG", chain)
    top.addAtom("C1", element.carbon, res_sol)
    res_wat = top.addResidue("HOH", chain)
    top.addAtom("O", element.oxygen, res_wat)
    top.addAtom("H1", element.hydrogen, res_wat)
    top.addAtom("H2", element.hydrogen, res_wat)

    system = mm.System()
    for _ in range(4):
        system.addParticle(1.0)
    nb = mm.NonbondedForce()
    for _ in range(4):
        nb.addParticle(0.0, 0.3, 0.0)
    nb.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    system.addForce(nb)
    return system, top


def _direct_payload(
    counts,
    orient,
    energy_sw,
    energy_ww,
    origin,
    n_frames,
    direct_sw_total,
    direct_ww_total,
    frame_direct_sw=None,
    frame_direct_ww=None,
    frame_offsets=None,
    frame_cells=None,
    frame_sw=None,
    frame_ww=None,
    frame_pme_sw=None,
    frame_pme_ww=None,
):
    frame_direct_sw = np.asarray(
        [] if frame_direct_sw is None else frame_direct_sw, dtype=np.float64
    )
    frame_direct_ww = np.asarray(
        [] if frame_direct_ww is None else frame_direct_ww, dtype=np.float64
    )
    frame_offsets = np.asarray([] if frame_offsets is None else frame_offsets, dtype=np.uint64)
    frame_cells = np.asarray([] if frame_cells is None else frame_cells, dtype=np.uint32)
    frame_sw = np.asarray([] if frame_sw is None else frame_sw, dtype=np.float64)
    frame_ww = np.asarray([] if frame_ww is None else frame_ww, dtype=np.float64)
    frame_pme_sw = np.asarray(
        [] if frame_pme_sw is None else frame_pme_sw, dtype=np.float64
    )
    frame_pme_ww = np.asarray(
        [] if frame_pme_ww is None else frame_pme_ww, dtype=np.float64
    )
    return (
        counts,
        orient,
        energy_sw,
        energy_ww,
        origin,
        n_frames,
        direct_sw_total,
        direct_ww_total,
        frame_direct_sw,
        frame_direct_ww,
        frame_offsets,
        frame_cells,
        frame_sw,
        frame_ww,
        frame_pme_sw,
        frame_pme_ww,
    )
