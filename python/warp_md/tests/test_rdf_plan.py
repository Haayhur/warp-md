import numpy as np

import warp_md as wmd


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

    system = wmd.System.from_pdb(str(pdb))
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
