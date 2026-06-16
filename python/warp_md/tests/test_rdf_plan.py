import importlib

import numpy as np
import pytest

import warp_md as wmd

rdf_mod = importlib.import_module("warp_md.analysis.rdf")


def test_rdf_plan_accepts_frame_indices_with_multimodel_pdb(tmp_path):
    pdb = tmp_path / "two_frame.pdb"
    pdb.write_text(
        "MODEL        1\n"
        "ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\n"
        "ATOM      2  O   HOH A   2       1.000   0.000   0.000  1.00  0.00           O\n"
        "ENDMDL\n"
        "MODEL        2\n"
        "ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\n"
        "ATOM      2  O   HOH A   2       3.000   0.000   0.000  1.00  0.00           O\n"
        "ENDMDL\n"
        "END\n",
        encoding="ascii",
    )

    system = wmd.System.from_file(str(pdb))
    sel = system.select("name O")

    traj_full = wmd.Trajectory.open_pdb(str(pdb), system)
    _r_full, _g_full, counts_full = wmd.RdfPlan(sel, sel, bins=4, r_max=4.0, pbc="none").run(
        traj_full,
        system,
        device="cpu",
    )

    traj_subset = wmd.Trajectory.open_pdb(str(pdb), system)
    _r_sub, _g_sub, counts_sub = wmd.RdfPlan(sel, sel, bins=4, r_max=4.0, pbc="none").run(
        traj_subset,
        system,
        device="cpu",
        frame_indices=[1, -1, 99],
    )

    np.testing.assert_array_equal(np.asarray(counts_full, dtype=np.int64), np.array([0, 2, 0, 2]))
    np.testing.assert_array_equal(np.asarray(counts_sub, dtype=np.int64), np.array([0, 0, 0, 4]))


class _DummySelection:
    def __init__(self, indices):
        self.indices = indices


class _DummySystem:
    def __init__(self):
        self._atoms = {"resid": [1, 2], "chain_id": [0, 0], "resname": ["WAT", "WAT"]}

    def atom_table(self):
        return self._atoms

    def select(self, _mask):
        return _DummySelection([0, 1])

    def select_indices(self, indices):
        return _DummySelection(indices)


class _DummyTraj:
    def __init__(self):
        self.count_calls = []

    def count_frames(self, chunk_frames=None):
        self.count_calls.append(chunk_frames)
        return 4


def test_rdf_wrapper_passes_native_trajectory_to_rust(monkeypatch):
    seen = {}

    class _DummyPlan:
        def __init__(self, sel_a, sel_b, **kwargs):
            seen["sel_a"] = list(sel_a.indices)
            seen["sel_b"] = list(sel_b.indices)
            seen["kwargs"] = kwargs

        def run(self, traj, system, chunk_frames=None, device="auto", frame_indices=None):
            seen["traj"] = traj
            seen["system"] = system
            seen["chunk_frames"] = chunk_frames
            seen["device"] = device
            seen["frame_indices"] = frame_indices
            return (
                np.array([0.5, 1.5], dtype=np.float32),
                np.array([8.0, 0.0], dtype=np.float32),
                np.array([8, 0], dtype=np.uint64),
                np.array([1.0, 1.0], dtype=np.float32),
            )

    traj = _DummyTraj()
    system = _DummySystem()
    monkeypatch.setattr(rdf_mod, "load_native_symbol", lambda _name: _DummyPlan)
    monkeypatch.setattr(rdf_mod, "is_native_traj", lambda got: got is traj)
    monkeypatch.setattr(rdf_mod, "coerce_native_system", lambda got: got)

    out = rdf_mod.rdf(
        traj,
        system,
        solvent_mask="all",
        solute_mask="all",
        maximum=2.0,
        bin_spacing=1.0,
        image=False,
        density=1.0,
        raw_rdf=True,
        intrdf=True,
        dimension="xy",
        frame_indices=[0, -1, 99],
        chunk_frames=7,
        dtype="dict",
    )

    assert seen["traj"] is traj
    assert seen["system"] is system
    assert seen["chunk_frames"] == 7
    assert seen["device"] == "auto"
    assert seen["frame_indices"] == [0, -1, 99]
    assert seen["kwargs"]["pbc"] == "none"
    assert seen["kwargs"]["density"] == 1.0
    assert seen["kwargs"]["volume"] is False
    assert seen["kwargs"]["raw_rdf"] is True
    assert seen["kwargs"]["intrdf"] is True
    assert seen["kwargs"]["dimension"] == "xy"
    assert traj.count_calls == []
    assert out["rdf"].tolist() == [8.0, 0.0]
    assert out["counts"].tolist() == [8, 0]
    assert out["integral_rdf"].tolist() == [1.0, 1.0]


def test_rdf_wrapper_rejects_non_native_trajectory(monkeypatch):
    monkeypatch.setattr(rdf_mod, "load_native_symbol", lambda _name: object)
    monkeypatch.setattr(rdf_mod, "is_native_traj", lambda _traj: False)
    with pytest.raises(RuntimeError, match="Rust-backed trajectory"):
        rdf_mod.rdf(_DummyTraj(), _DummySystem(), image=False)
