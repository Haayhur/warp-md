import importlib

import numpy as np
import pytest

spol_mod = importlib.import_module("warp_md.analysis.spol")


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self):
        self._atoms = {
            "name": ["CA", "OW", "HW1", "HW2"],
            "resname": ["ALA", "SOL", "SOL", "SOL"],
            "resid": [1, 2, 2, 2],
            "element": ["C", "O", "H", "H"],
            "chain_id": [0, 0, 0, 0],
            "charge": [0.0, -0.834, 0.417, 0.417],
        }

    def atom_table(self):
        return self._atoms

    def select(self, mask):
        if mask == "resid 1":
            return _DummySelection([0])
        return _DummySelection([1, 2, 3])

    def select_indices(self, indices):
        return _DummySelection(list(indices))


class _DummyTraj:
    pass


def test_spol_detects_triplets_charges_and_forwards(monkeypatch):
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            solute_selection,
            atom1_indices,
            atom2_indices,
            atom3_indices,
            charges,
            r_min=0.0,
            r_max=0.32,
            bin=0.01,
            use_com=False,
            reference_atom=0,
            direction_atoms=None,
            refdip=0.0,
            r_hist_max=None,
            length_scale=None,
            molecule_atoms=None,
            molecule_offsets=None,
        ):
            called["solute_indices"] = list(solute_selection.indices)
            called["atom1_indices"] = list(atom1_indices)
            called["atom2_indices"] = list(atom2_indices)
            called["atom3_indices"] = list(atom3_indices)
            called["charges"] = list(charges)
            called["r_min"] = r_min
            called["r_max"] = r_max
            called["bin"] = bin
            called["use_com"] = use_com
            called["reference_atom"] = reference_atom
            called["direction_atoms"] = direction_atoms
            called["refdip"] = refdip
            called["r_hist_max"] = r_hist_max
            called["length_scale"] = length_scale
            called["molecule_atoms"] = molecule_atoms
            called["molecule_offsets"] = molecule_offsets

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["chunk_frames"] = chunk_frames
            called["device"] = device
            called["frame_indices"] = frame_indices
            return {
                "r": np.array([0.5, 1.0, 1.5], dtype=np.float32),
                "cumulative_count": np.array([0.0, 1.0, 1.0], dtype=np.float32),
                "shell_count": np.array([0, 1, 0], dtype=np.uint64),
                "shell_count_per_frame": np.array([0.0, 0.5, 0.0], dtype=np.float32),
                "average_shell_size": 0.5,
                "average_dipole": 2.0,
                "dipole_std": 0.1,
                "average_radial_dipole": 1.5,
                "average_radial_polarization": 1.2,
                "window_count": 1,
                "r_window": np.array([0.1, 1.5], dtype=np.float32),
                "bin_width": 0.5,
                "r_hist_max": 1.5,
                "use_com": True,
                "reference_atom": 1,
                "refdip": 1.0,
                "n_frames": 2,
                "used_box": False,
                "length_scale": 0.1,
                "dipole_unit": "debye",
            }

    monkeypatch.setattr(spol_mod, "_SolventPolarizationPlan", _DummyPlan, raising=True)
    out = spol_mod.spol(
        _DummyTraj(),
        _DummySystem(),
        solute_selection="resid 1",
        solvent_selection="resname SOL",
        r_min=0.1,
        r_max=1.5,
        bin=0.5,
        use_com=True,
        reference_atom=1,
        refdip=1.0,
        r_hist_max=1.5,
        length_scale=0.1,
        chunk_frames=32,
        frame_indices=[0, 2],
    )
    assert called["solute_indices"] == [0]
    assert called["atom1_indices"] == []
    assert called["atom2_indices"] == []
    assert called["atom3_indices"] == []
    assert called["molecule_atoms"] == [1, 2, 3]
    assert called["molecule_offsets"] == [0, 3]
    assert called["charges"][:4] == [0.0, -0.834, 0.417, 0.417]
    assert called["use_com"] is True
    assert called["reference_atom"] == 1
    assert called["direction_atoms"] == [0, 1, 2]
    assert called["refdip"] == pytest.approx(1.0)
    assert called["chunk_frames"] == 32
    assert called["frame_indices"] == [0, 2]
    assert out["dipole_unit"] == "debye"
    assert out["reference_atom"] == 1
    np.testing.assert_allclose(
        out["cumulative_count"], np.array([0.0, 1.0, 1.0], dtype=np.float32)
    )


def test_spol_raises_when_binding_missing(monkeypatch):
    monkeypatch.setattr(spol_mod, "_SolventPolarizationPlan", None, raising=True)
    with pytest.raises(RuntimeError, match="PySolventPolarizationPlan binding unavailable"):
        spol_mod.spol(_DummyTraj(), _DummySystem(), solute_selection="resid 1")


def test_spol_supports_explicit_molecules(monkeypatch):
    called = {}

    class _DummyPlan:
        def __init__(self, *_args, molecule_atoms=None, molecule_offsets=None, direction_atoms=None, **_kwargs):
            called["molecule_atoms"] = molecule_atoms
            called["molecule_offsets"] = molecule_offsets
            called["direction_atoms"] = direction_atoms

        def run(self, *_args, **_kwargs):
            return {
                "r": np.array([], dtype=np.float32),
                "cumulative_count": np.array([], dtype=np.float32),
                "shell_count": np.array([], dtype=np.uint64),
                "shell_count_per_frame": np.array([], dtype=np.float32),
                "average_shell_size": 0.0,
                "average_dipole": 0.0,
                "dipole_std": 0.0,
                "average_radial_dipole": 0.0,
                "average_radial_polarization": 0.0,
                "window_count": 0,
                "r_window": np.array([0.0, 0.32], dtype=np.float32),
                "bin_width": 0.01,
                "r_hist_max": 0.32,
                "use_com": False,
                "reference_atom": 0,
                "refdip": 0.0,
                "n_frames": 0,
                "used_box": False,
                "length_scale": 1.0,
                "dipole_unit": "debye",
            }

    monkeypatch.setattr(spol_mod, "_SolventPolarizationPlan", _DummyPlan, raising=True)
    spol_mod.spol(
        _DummyTraj(),
        _DummySystem(),
        solute_selection="resid 1",
        molecules=[[1, 2, 3]],
        direction_atom_offsets=(0, 1, 2),
    )
    assert called["molecule_atoms"] == [1, 2, 3]
    assert called["molecule_offsets"] == [0, 3]
    assert called["direction_atoms"] == [0, 1, 2]


class _MolIdSystem:
    def __init__(self):
        self._atoms = {
            "name": ["CA", "OW", "HW1", "HW2", "OW", "HW1", "HW2"],
            "resname": ["ALA", "SOL", "SOL", "SOL", "SOL", "SOL", "SOL"],
            "resid": [1, 2, 2, 2, 2, 2, 2],
            "element": ["C", "O", "H", "H", "O", "H", "H"],
            "chain_id": [0, 0, 0, 0, 0, 0, 0],
            "charge": [0.0, -0.834, 0.417, 0.417, -0.834, 0.417, 0.417],
            "mol_id": [0, 10, 10, 10, 11, 11, 11],
        }

    def atom_table(self):
        return self._atoms

    def select(self, mask):
        if mask == "resid 1":
            return _DummySelection([0])
        if mask == "name OW":
            return _DummySelection([1, 4])
        return _DummySelection([1, 2, 3, 4, 5, 6])

    def select_indices(self, indices):
        return _DummySelection(list(indices))


def test_spol_prefers_mol_id_whole_molecule_groups(monkeypatch):
    called = {}

    class _DummyPlan:
        def __init__(self, *_args, molecule_atoms=None, molecule_offsets=None, **_kwargs):
            called["molecule_atoms"] = molecule_atoms
            called["molecule_offsets"] = molecule_offsets

        def run(self, *_args, **_kwargs):
            return {
                "r": np.array([], dtype=np.float32),
                "cumulative_count": np.array([], dtype=np.float32),
                "shell_count": np.array([], dtype=np.uint64),
                "shell_count_per_frame": np.array([], dtype=np.float32),
                "average_shell_size": 0.0,
                "average_dipole": 0.0,
                "dipole_std": 0.0,
                "average_radial_dipole": 0.0,
                "average_radial_polarization": 0.0,
                "window_count": 0,
                "r_window": np.array([0.0, 0.32], dtype=np.float32),
                "bin_width": 0.01,
                "r_hist_max": 0.32,
                "use_com": False,
                "reference_atom": 0,
                "refdip": 0.0,
                "n_frames": 0,
                "used_box": False,
                "length_scale": 1.0,
                "dipole_unit": "debye",
            }

    monkeypatch.setattr(spol_mod, "_SolventPolarizationPlan", _DummyPlan, raising=True)
    spol_mod.spol(
        _DummyTraj(),
        _MolIdSystem(),
        solute_selection="resid 1",
        solvent_selection="resname SOL",
    )
    assert called["molecule_atoms"] == [1, 2, 3, 4, 5, 6]
    assert called["molecule_offsets"] == [0, 3, 6]


def test_spol_rejects_partial_molecule_selection():
    with pytest.raises(ValueError, match="whole molecules"):
        spol_mod._molecule_groups(_MolIdSystem(), "name OW", ("SOL",))
