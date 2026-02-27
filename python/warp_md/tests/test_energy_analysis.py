import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))


def _require_openmm():
    try:
        from openmm import openmm as mm
    except Exception:  # pragma: no cover
        pytest.skip("openmm not available")
    return mm


class _FakeTraj:
    def __init__(self, coords, box=None):
        self.coords = coords
        self.box = box
        self._used = False

    def read_chunk(self, _max_frames=128):
        if self._used:
            return None
        self._used = True
        return {"coords": self.coords, "box": self.box, "frames": self.coords.shape[0]}


def _build_system():
    mm = _require_openmm()
    system = mm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    nb = mm.NonbondedForce()
    nb.addParticle(1.0, 0.3, 0.1)
    nb.addParticle(-1.0, 0.3, 0.1)
    nb.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    system.addForce(nb)
    return system


def test_energy_analysis_groups():
    mm = _require_openmm()
    from warp_md.analysis.energy_analysis import energy_analysis

    system = _build_system()
    coords = np.array([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]], dtype=float)
    traj = _FakeTraj(coords)
    out = energy_analysis(traj, None, system, energy_groups=True, summarize=True, length_scale=1.0)
    assert "potential" in out
    assert "groups" in out
    assert "group_names" in out
    assert "potential_mean" in out
    assert len(out["group_names"]) == len(system.getForces())


def test_lie_basic():
    mm = _require_openmm()
    from warp_md.analysis.energy_analysis import lie

    system = _build_system()
    coords = np.array([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]], dtype=float)
    traj = _FakeTraj(coords)
    out = lie(traj, None, system, selection=[0], length_scale=1.0)
    assert "vdw" in out
    assert len(out["vdw"]) == 1
