import numpy as np
import pytest

import warp_md
from warp_md.analysis.docking import docking, docking_ligplot_svg


class _DummySelection:
    def __init__(self, indices):
        self.indices = list(indices)


class _DummySystem:
    def __init__(self, n_atoms):
        self._n_atoms = n_atoms

    def select(self, _mask):
        return _DummySelection(range(self._n_atoms))

    def select_indices(self, indices):
        return _DummySelection(indices)

    def atom_table(self):
        return {
            "name": ["O1", "C1", "N1", "C2"][: self._n_atoms],
            "resname": ["REC", "REC", "LIG", "LIG"][: self._n_atoms],
            "resid": [10, 10, 20, 20][: self._n_atoms],
            "chain_id": [0, 0, 0, 0][: self._n_atoms],
        }


class _DummyTraj:
    pass


def test_docking_agent_contract(monkeypatch):
    traj = _DummyTraj()
    system = _DummySystem(4)
    called = {}

    class _DummyPlan:
        def __init__(
            self,
            _receptor,
            _ligand,
            close_contact_cutoff=4.0,
            hydrophobic_cutoff=4.0,
            hydrogen_bond_cutoff=3.5,
            clash_cutoff=2.5,
            salt_bridge_cutoff=5.5,
            halogen_bond_cutoff=5.5,
            metal_coordination_cutoff=3.5,
            cation_pi_cutoff=6.0,
            pi_pi_cutoff=7.5,
            hbond_min_angle_deg=120.0,
            donor_hydrogen_cutoff=1.25,
            allow_missing_hydrogen=True,
            length_scale=1.0,
            max_events_per_frame=20_000,
        ):
            called["close_contact_cutoff"] = close_contact_cutoff
            called["hydrophobic_cutoff"] = hydrophobic_cutoff
            called["hydrogen_bond_cutoff"] = hydrogen_bond_cutoff
            called["clash_cutoff"] = clash_cutoff
            called["salt_bridge_cutoff"] = salt_bridge_cutoff
            called["halogen_bond_cutoff"] = halogen_bond_cutoff
            called["metal_coordination_cutoff"] = metal_coordination_cutoff
            called["cation_pi_cutoff"] = cation_pi_cutoff
            called["pi_pi_cutoff"] = pi_pi_cutoff
            called["hbond_min_angle_deg"] = hbond_min_angle_deg
            called["donor_hydrogen_cutoff"] = donor_hydrogen_cutoff
            called["allow_missing_hydrogen"] = allow_missing_hydrogen
            called["length_scale"] = length_scale
            called["max_events_per_frame"] = max_events_per_frame

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            called["chunk_frames"] = chunk_frames
            called["device"] = device
            called["frame_indices"] = frame_indices
            return np.array(
                [
                    [0.0, 0.0, 2.0, 1.0, 2.8, 0.2],
                    [0.0, 1.0, 3.0, 2.0, 3.0, 0.25],
                    [1.0, 0.0, 2.0, 4.0, 1.0, 0.4],
                ],
                dtype=np.float32,
            )

    monkeypatch.setattr(warp_md, "DockingPlan", _DummyPlan, raising=False)
    out = docking(
        traj,
        system,
        receptor_mask=[0, 1],
        ligand_mask=[2, 3],
        close_contact_cutoff=4.0,
        hydrophobic_cutoff=4.0,
        hydrogen_bond_cutoff=3.5,
        clash_cutoff=2.5,
        salt_bridge_cutoff=5.5,
        halogen_bond_cutoff=5.5,
        metal_coordination_cutoff=3.5,
        cation_pi_cutoff=6.0,
        pi_pi_cutoff=7.5,
        hbond_min_angle_deg=130.0,
        donor_hydrogen_cutoff=1.2,
        allow_missing_hydrogen=False,
        length_scale=0.1,
        frame_indices=[1],
        chunk_frames=64,
        device="cpu",
    )
    assert out["schema_version"] == "warp_md.docking.interactions.v1"
    assert out["summary"]["n_events"] == 3
    assert out["summary"]["counts_by_type"]["hydrogen_bond"] == 1
    assert out["summary"]["counts_by_type"]["hydrophobic_contact"] == 1
    assert out["summary"]["counts_by_type"]["clash"] == 1
    assert out["summary"]["counts_by_type"]["salt_bridge"] == 0
    assert out["events"][0]["interaction_type"] == "hydrogen_bond"
    assert out["events"][0]["receptor_atom"]["resname"] == "REC"
    assert out["events"][0]["ligand_atom"]["resname"] == "LIG"
    assert out["events"][2]["frame_index"] == 1
    assert called["close_contact_cutoff"] == 4.0
    assert called["hydrophobic_cutoff"] == 4.0
    assert called["hydrogen_bond_cutoff"] == 3.5
    assert called["clash_cutoff"] == 2.5
    assert called["salt_bridge_cutoff"] == 5.5
    assert called["halogen_bond_cutoff"] == 5.5
    assert called["metal_coordination_cutoff"] == 3.5
    assert called["cation_pi_cutoff"] == 6.0
    assert called["pi_pi_cutoff"] == 7.5
    assert called["hbond_min_angle_deg"] == 130.0
    assert called["donor_hydrogen_cutoff"] == 1.2
    assert called["allow_missing_hydrogen"] is False
    assert called["max_events_per_frame"] == 20_000
    assert called["chunk_frames"] == 64
    assert called["frame_indices"] == [1]


def test_docking_rejects_invalid_contract(monkeypatch):
    traj = _DummyTraj()
    system = _DummySystem(4)

    class _DummyPlan:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, _traj, _system, chunk_frames=None, device="auto", frame_indices=None):
            return np.array([1.0, 2.0, 3.0], dtype=np.float32)

    monkeypatch.setattr(warp_md, "DockingPlan", _DummyPlan, raising=False)
    with pytest.raises(RuntimeError, match="output shape mismatch"):
        docking(traj, system, receptor_mask=[0, 1], ligand_mask=[2, 3])


def test_docking_rejects_invalid_numeric_contract():
    traj = _DummyTraj()
    system = _DummySystem(4)
    with pytest.raises(ValueError, match="close_contact_cutoff must be finite and > 0"):
        docking(traj, system, receptor_mask=[0, 1], ligand_mask=[2, 3], close_contact_cutoff=0.0)
    with pytest.raises(ValueError, match="hydrogen_bond_cutoff must be <= close_contact_cutoff"):
        docking(
            traj,
            system,
            receptor_mask=[0, 1],
            ligand_mask=[2, 3],
            close_contact_cutoff=3.0,
            hydrophobic_cutoff=3.0,
            hydrogen_bond_cutoff=3.5,
        )


def test_docking_ligplot_svg_renderer(tmp_path):
    result = {
        "residues": [
            {
                "chain_id": 0,
                "resid": 10,
                "resname": "ASP",
                "interaction_count": 5,
                "counts_by_type": {"salt_bridge": 3, "hydrogen_bond": 2},
            },
            {
                "chain_id": 0,
                "resid": 42,
                "resname": "TRP",
                "interaction_count": 4,
                "counts_by_type": {"pi_pi_stacking": 3, "hydrophobic_contact": 1},
            },
        ]
    }
    svg = docking_ligplot_svg(result, max_residues=2, width=640, height=480, title="Docking Map")
    assert svg.startswith("<svg")
    assert "Docking Map" in svg
    assert "ASP 10" in svg
    path = tmp_path / "docking.svg"
    out = docking_ligplot_svg(result, path=str(path))
    assert out.startswith("<svg")
    assert path.exists()


def test_docking_no_python_fallback_when_plan_missing(monkeypatch):
    traj = _DummyTraj()
    system = _DummySystem(4)

    class _MissingPlan:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("DockingPlan binding unavailable in this build.")

    monkeypatch.setattr(warp_md, "DockingPlan", _MissingPlan, raising=False)
    with pytest.raises(RuntimeError, match="DockingPlan binding unavailable"):
        docking(traj, system, receptor_mask=[0, 1], ligand_mask=[2, 3])
