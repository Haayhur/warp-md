import numpy as np

from warp_md.analysis.multidihedral import multidihedral


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self):
        self._atoms = {
            "name": ["N", "CA", "C", "N", "CA", "C"],
            "resid": [1, 1, 1, 2, 2, 2],
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


def test_multidihedral_phi_psi():
    coords = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
                [3.0, 2.0, 0.0],
            ]
        ],
        dtype=np.float64,
    )
    traj = _DummyTraj(coords)
    system = _DummySystem()

    out = multidihedral(traj, system, dihedral_types="phi psi", dtype="dict")
    assert isinstance(out, dict)
    assert out
    for key, val in out.items():
        assert val.shape == (1,)


class _ArgSystem:
    def __init__(self):
        names = ["N", "CA", "CB", "CG", "CD", "NE", "CZ", "NH1"]
        self._atoms = {
            "name": names,
            "resname": ["ARG"] * len(names),
            "resid": [1] * len(names),
        }

    def atom_table(self):
        return self._atoms

    def select(self, _mask):
        return _DummySelection(range(len(self._atoms["name"])))


def _arg_coords():
    return np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.1, 0.0],
                [1.2, 1.1, 0.3],
                [2.3, 1.0, 0.1],
                [2.5, 2.0, 0.5],
                [3.6, 2.2, 0.2],
                [4.1, 3.1, 0.8],
                [5.2, 3.0, 0.4],
            ]
        ],
        dtype=np.float64,
    )


def test_multidihedral_named_chi_sidechain_torsions():
    out = multidihedral(
        _DummyTraj(_arg_coords()),
        _ArgSystem(),
        dihedral_types="chi1 chi2 chi3 chi4 chi5",
        dtype="dict",
    )

    assert list(out.keys()) == ["chi1:1", "chi2:1", "chi3:1", "chi4:1", "chi5:1"]
    for val in out.values():
        assert val.shape == (1,)
        assert np.isfinite(val).all()


def test_multidihedral_chi_expands_to_available_chi_torsions():
    out = multidihedral(
        _DummyTraj(_arg_coords()),
        _ArgSystem(),
        dihedral_types="chi",
        dtype="dict",
    )

    assert list(out.keys()) == ["chi1:1", "chi2:1", "chi3:1", "chi4:1", "chi5:1"]


def test_multidihedral_single_backbone_request_is_not_overbroad():
    phi = multidihedral(
        _DummyTraj(_arg_coords()),
        _ArgSystem(),
        dihedral_types="phi",
        dtype="dict",
    )
    psi = multidihedral(
        _DummyTraj(_arg_coords()),
        _ArgSystem(),
        dihedral_types="psi",
        dtype="dict",
    )

    assert phi == {}
    assert psi == {}
