use super::*;
use crate::bonded_terms::{BondTermGroup, VirtualSiteTerm};
use std::fs;

#[test]
fn map_frame_can_use_mass_weighted_centers() {
    let source = vec![[0.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, 1.0]];
    let mapping = vec![vec![0, 1]];
    let weights = vec![vec![1.0, 3.0]];

    let cog = map_frame(&source, &mapping, None);
    let com = map_frame(&source, &mapping, Some(&weights));

    assert!((cog[0][0] - 5.0).abs() < 1.0e-6);
    assert!((com[0][0] - 7.5).abs() < 1.0e-6);
}

#[test]
fn mass_weighted_mapping_splits_shared_atom_mass_like_swarm_cg() {
    let dir = tempfile::tempdir().unwrap();
    let top = dir.path().join("split.prmtop");
    fs::write(
        &top,
        concat!(
            "%FLAG TITLE\n",
            "%FORMAT(20a4)\n",
            "TEST PRMTOP\n",
            "%FLAG POINTERS\n",
            "%FORMAT(10I8)\n",
            "       3       1       0       0       0       0       0       0       0       0\n",
            "%FLAG ATOM_NAME\n",
            "%FORMAT(20a4)\n",
            "C1  H1  H2  \n",
            "%FLAG CHARGE\n",
            "%FORMAT(5E16.8)\n",
            "  0.00000000E+00  0.00000000E+00  0.00000000E+00\n",
            "%FLAG MASS\n",
            "%FORMAT(5E16.8)\n",
            "  1.20000000E+01  1.00000000E+00  1.00000000E+00\n",
            "%FLAG RESIDUE_LABEL\n",
            "%FORMAT(20a4)\n",
            "MOL \n",
            "%FLAG RESIDUE_POINTER\n",
            "%FORMAT(10I8)\n",
            "       1\n",
            "%FLAG ATOMIC_NUMBER\n",
            "%FORMAT(10I8)\n",
            "       6       1       1\n",
            "%FLAG ATOM_TYPE_INDEX\n",
            "%FORMAT(10I8)\n",
            "       1       1       1\n"
        ),
    )
    .unwrap();
    let options = NativeTrajectoryOptions {
        topology: Some(top.to_string_lossy().to_string()),
        topology_format: Some("prmtop".to_string()),
        mass_weighted: true,
        ..NativeTrajectoryOptions::default()
    };

    let weights = resolve_bead_weights(&options, 3, &[vec![0, 1], vec![0, 2]]).unwrap();
    assert_eq!(weights.len(), 2);
    assert!((weights[0][0] - 6.0).abs() < 1.0e-6);
    assert!((weights[0][1] - 1.0).abs() < 1.0e-6);
    assert!((weights[1][0] - 6.0).abs() < 1.0e-6);
    assert!((weights[1][1] - 1.0).abs() < 1.0e-6);

    let duplicate_weights =
        resolve_bead_weights(&options, 3, &[vec![0, 0, 1], vec![0, 2]]).unwrap();
    assert!((duplicate_weights[0][0] - 4.0).abs() < 1.0e-6);
    assert!((duplicate_weights[0][1] - 4.0).abs() < 1.0e-6);
    assert!((duplicate_weights[1][0] - 4.0).abs() < 1.0e-6);
}

#[test]
fn make_whole_by_bonded_connectivity_repairs_pbc_split_beads() {
    let coords = [[9.8, 0.0, 0.0], [0.2, 0.0, 0.0], [4.0, 0.0, 0.0]];
    let repaired = make_whole_by_bonded_connectivity(
        &coords,
        &[(0, 1)],
        Box3::Orthorhombic {
            lx: 10.0,
            ly: 10.0,
            lz: 10.0,
        },
    )
    .unwrap();

    assert!((repaired[0][0] - 9.8).abs() < 1.0e-6);
    assert!((repaired[1][0] - 10.2).abs() < 1.0e-5);
    assert!((repaired[2][0] - 4.0).abs() < 1.0e-6);
}

#[test]
fn native_mapping_make_whole_repairs_bonded_reference_distribution() {
    let dir = tempfile::tempdir().unwrap();
    let traj = dir.path().join("wrapped.xtc");
    let mut writer = XtcWriter::create(&traj, 2).unwrap();
    writer
        .write_frame(
            &[[9.8, 0.0, 0.0], [0.2, 0.0, 0.0]],
            Box3::Orthorhombic {
                lx: 10.0,
                ly: 10.0,
                lz: 10.0,
            },
            0,
            Some(0.0),
        )
        .unwrap();
    writer.flush().unwrap();

    let mapping = BeadMapping {
        bead_names: vec!["B1".to_string(), "B2".to_string()],
        atom_indices: vec![vec![0], vec![1]],
    };
    let terms = BondedTermSet::from_connections(2, &[(0, 1)]);
    let wrapped = map_native_trajectory_with_terms(
        &traj,
        None,
        &mapping,
        &terms,
        &NativeTrajectoryOptions {
            format: Some("xtc".to_string()),
            ..NativeTrajectoryOptions::default()
        },
    )
    .unwrap();
    let whole = map_native_trajectory_with_terms(
        &traj,
        None,
        &mapping,
        &terms,
        &NativeTrajectoryOptions {
            format: Some("xtc".to_string()),
            make_whole: true,
            ..NativeTrajectoryOptions::default()
        },
    )
    .unwrap();

    assert!((wrapped.bond_stats[0].mean - 9.6).abs() < 1.0e-5);
    assert!((whole.bond_stats[0].mean - 0.4).abs() < 1.0e-5);
}

#[test]
fn native_mapping_make_whole_repairs_source_atoms_before_com_mapping() {
    let dir = tempfile::tempdir().unwrap();
    let top = dir.path().join("wrapped.gro");
    fs::write(
        &top,
        concat!(
            "wrapped two atom molecule\n",
            "2\n",
            "    1MOL     A1    1   9.800   0.000   0.000\n",
            "    1MOL     A2    2   0.200   0.000   0.000\n",
            "   10.00000 10.00000 10.00000\n",
        ),
    )
    .unwrap();
    let traj = dir.path().join("wrapped.xtc");
    let mut writer = XtcWriter::create(&traj, 2).unwrap();
    writer
        .write_frame(
            &[[9.8, 0.0, 0.0], [0.2, 0.0, 0.0]],
            Box3::Orthorhombic {
                lx: 10.0,
                ly: 10.0,
                lz: 10.0,
            },
            0,
            Some(0.0),
        )
        .unwrap();
    writer.flush().unwrap();

    let mapping = BeadMapping {
        bead_names: vec!["B1".to_string()],
        atom_indices: vec![vec![0, 1]],
    };
    let wrapped = map_native_trajectory(
        &traj,
        None,
        &mapping,
        &[],
        &NativeTrajectoryOptions {
            format: Some("xtc".to_string()),
            ..NativeTrajectoryOptions::default()
        },
    )
    .unwrap();
    let whole = map_native_trajectory(
        &traj,
        None,
        &mapping,
        &[],
        &NativeTrajectoryOptions {
            topology: Some(top.to_string_lossy().to_string()),
            topology_format: Some("gro".to_string()),
            format: Some("xtc".to_string()),
            make_whole: true,
            ..NativeTrajectoryOptions::default()
        },
    )
    .unwrap();

    let wrapped_x = wrapped.first_cg_coords.unwrap()[0][0];
    let whole_x = whole.first_cg_coords.unwrap()[0][0];
    assert!((wrapped_x - 5.0).abs() < 1.0e-5);
    assert!((whole_x - 10.0).abs() < 1.0e-5);
}

#[test]
fn native_mapping_make_whole_uses_topology_bonds_not_atom_adjacency() {
    let dir = tempfile::tempdir().unwrap();
    let top = dir.path().join("non_adjacent_bond.pdb");
    fs::write(
        &top,
        concat!(
            "ATOM      1  A1  MOL A   1       9.800   0.000   0.000  1.00  0.00           C\n",
            "ATOM      2  A2  MOL A   1       5.000   0.000   0.000  1.00  0.00           C\n",
            "ATOM      3  A3  MOL A   1       0.200   0.000   0.000  1.00  0.00           C\n",
            "CONECT    1    3\n",
            "END\n",
        ),
    )
    .unwrap();
    let traj = dir.path().join("non_adjacent_bond.xtc");
    let mut writer = XtcWriter::create(&traj, 3).unwrap();
    writer
        .write_frame(
            &[[9.8, 0.0, 0.0], [5.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
            Box3::Orthorhombic {
                lx: 10.0,
                ly: 10.0,
                lz: 10.0,
            },
            0,
            Some(0.0),
        )
        .unwrap();
    writer.flush().unwrap();

    let mapping = BeadMapping {
        bead_names: vec!["B1".to_string()],
        atom_indices: vec![vec![0, 2]],
    };
    let wrapped = map_native_trajectory(
        &traj,
        None,
        &mapping,
        &[],
        &NativeTrajectoryOptions {
            format: Some("xtc".to_string()),
            ..NativeTrajectoryOptions::default()
        },
    )
    .unwrap();
    let whole = map_native_trajectory(
        &traj,
        None,
        &mapping,
        &[],
        &NativeTrajectoryOptions {
            topology: Some(top.to_string_lossy().to_string()),
            topology_format: Some("pdb".to_string()),
            format: Some("xtc".to_string()),
            make_whole: true,
            ..NativeTrajectoryOptions::default()
        },
    )
    .unwrap();

    let wrapped_x = wrapped.first_cg_coords.unwrap()[0][0];
    let whole_x = whole.first_cg_coords.unwrap()[0][0];
    assert!((wrapped_x - 5.0).abs() < 1.0e-5);
    assert!((whole_x - 10.0).abs() < 1.0e-5);
}

#[test]
fn native_mapping_reports_mapped_radius_of_gyration_stats() {
    let dir = tempfile::tempdir().unwrap();
    let traj = dir.path().join("rg.xtc");
    let mut writer = XtcWriter::create(&traj, 2).unwrap();
    writer
        .write_frame(
            &[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            Box3::None,
            0,
            Some(0.0),
        )
        .unwrap();
    writer
        .write_frame(
            &[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            Box3::None,
            1,
            Some(1.0),
        )
        .unwrap();
    writer.flush().unwrap();

    let report = map_native_trajectory(
        &traj,
        None,
        &BeadMapping {
            bead_names: vec!["B0".to_string(), "B1".to_string()],
            atom_indices: vec![vec![0], vec![1]],
        },
        &[],
        &NativeTrajectoryOptions {
            format: Some("xtc".to_string()),
            ..NativeTrajectoryOptions::default()
        },
    )
    .unwrap();
    let rg = report.rg_stats.unwrap();

    assert_eq!(rg.samples, 2);
    assert!((rg.mean - 1.5).abs() < 1.0e-12);
    assert!((rg.std - 0.5).abs() < 1.0e-12);
}

#[test]
fn native_mapping_reports_mapped_sasa_approx_stats() {
    let dir = tempfile::tempdir().unwrap();
    let traj = dir.path().join("sasa.xtc");
    let mut writer = XtcWriter::create(&traj, 1).unwrap();
    writer
        .write_frame(&[[0.0, 0.0, 0.0]], Box3::None, 0, Some(0.0))
        .unwrap();
    writer
        .write_frame(&[[1.0, 0.0, 0.0]], Box3::None, 1, Some(1.0))
        .unwrap();
    writer.flush().unwrap();

    let report = map_native_trajectory(
        &traj,
        None,
        &BeadMapping {
            bead_names: vec!["B0".to_string()],
            atom_indices: vec![vec![0]],
        },
        &[],
        &NativeTrajectoryOptions {
            format: Some("xtc".to_string()),
            ..NativeTrajectoryOptions::default()
        },
    )
    .unwrap();
    let sasa = report.sasa_stats.unwrap();
    let radius = DEFAULT_SASA_BEAD_RADIUS_NM + DEFAULT_SASA_PROBE_RADIUS_NM;
    let expected = 4.0 * std::f64::consts::PI * radius * radius;

    assert_eq!(sasa.samples, 2);
    assert!((sasa.mean - expected).abs() < 1.0e-12);
    assert!(sasa.std < 1.0e-12);
}

#[test]
fn native_mapping_computes_virtual_site_coordinates_before_reference_stats() {
    let dir = tempfile::tempdir().unwrap();
    let traj = dir.path().join("vs.xtc");
    let mut writer = XtcWriter::create(&traj, 2).unwrap();
    writer
        .write_frame(
            &[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            Box3::None,
            0,
            Some(0.0),
        )
        .unwrap();
    writer.flush().unwrap();

    let terms = BondedTermSet {
        bonds: vec![BondTermGroup {
            label: Some("real_to_virtual".to_string()),
            members: vec![[0, 2]],
        }],
        virtual_sites: vec![VirtualSiteTerm {
            site: 2,
            kind: "2".to_string(),
            function: 1,
            defining_beads: vec![0, 1],
            parameters: vec![0.25],
        }],
        ..BondedTermSet::default()
    };
    let report = map_native_trajectory_with_terms(
        &traj,
        None,
        &BeadMapping {
            bead_names: vec!["B0".to_string(), "B1".to_string()],
            atom_indices: vec![vec![0], vec![1]],
        },
        &terms,
        &NativeTrajectoryOptions {
            format: Some("xtc".to_string()),
            ..NativeTrajectoryOptions::default()
        },
    )
    .unwrap();

    let coords = report.first_cg_coords.unwrap();
    assert_eq!(coords.len(), 3);
    assert!((coords[2][0] - 0.5).abs() < 1.0e-6);
    assert!((report.bond_stats[0].mean - 0.5).abs() < 1.0e-6);
    assert_eq!(report.bonded_values.bonds[0].members, vec![[0, 2]]);

    let out = dir.path().join("mapped_vs.xtc");
    let written = map_native_trajectory_with_terms(
        &traj,
        Some(&out),
        &BeadMapping {
            bead_names: vec!["B0".to_string(), "B1".to_string()],
            atom_indices: vec![vec![0], vec![1]],
        },
        &terms,
        &NativeTrajectoryOptions {
            format: Some("xtc".to_string()),
            ..NativeTrajectoryOptions::default()
        },
    )
    .unwrap();
    assert_eq!(written.frames_written, 1);
    assert!(out.is_file());
}

#[test]
fn native_writer_supports_text_and_checkpoint_formats() {
    let dir = tempfile::tempdir().unwrap();
    let coords = [[1.0, 2.0, 3.0]];
    for ext in ["gro", "g96", "cpt"] {
        let path = dir.path().join(format!("mapped.{ext}"));
        let mut writer = open_writer(&path, 1).unwrap();
        writer
            .write_frame(&coords, Box3::None, 0, Some(0.0))
            .unwrap();
        assert!(path.is_file());
    }
}

#[test]
fn cpt_writer_reports_single_frame_contract() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mapped.cpt");
    let writer = open_writer(&path, 1).unwrap();

    assert!(writer.is_single_frame());
}

#[test]
fn native_mapping_uses_target_selection_in_solvated_trajectory() {
    let dir = tempfile::tempdir().unwrap();
    let top = dir.path().join("solvated.gro");
    fs::write(
        &top,
        concat!(
            "solvated ethanol\n",
            "5\n",
            "    1EOH     C1    1   0.000   0.000   0.000\n",
            "    1EOH     C2    2   0.100   0.000   0.000\n",
            "    1EOH     O1    3   0.200   0.000   0.000\n",
            "    2SOL     OW    4   0.500   0.500   0.500\n",
            "    2SOL    HW1    5   0.600   0.500   0.500\n",
            "   1.00000 1.00000 1.00000\n",
        ),
    )
    .unwrap();
    let traj = dir.path().join("solvated.xtc");
    let mut writer = XtcWriter::create(&traj, 5).unwrap();
    writer
        .write_frame(
            &[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [5.0, 5.0, 5.0],
                [6.0, 5.0, 5.0],
            ],
            Box3::Orthorhombic {
                lx: 10.0,
                ly: 10.0,
                lz: 10.0,
            },
            0,
            Some(0.0),
        )
        .unwrap();
    writer
        .write_frame(
            &[
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [5.0, 5.0, 5.0],
                [6.0, 5.0, 5.0],
            ],
            Box3::Orthorhombic {
                lx: 10.0,
                ly: 10.0,
                lz: 10.0,
            },
            1,
            Some(1.0),
        )
        .unwrap();
    writer.flush().unwrap();

    let out = dir.path().join("ethanol_cg.xtc");
    let report = map_native_trajectory(
        &traj,
        Some(&out),
        &BeadMapping {
            bead_names: vec!["SP1".to_string()],
            atom_indices: vec![vec![0, 1, 2]],
        },
        &[],
        &NativeTrajectoryOptions {
            topology: Some(top.to_string_lossy().to_string()),
            topology_format: Some("gro".to_string()),
            format: Some("xtc".to_string()),
            target_selection: Some("resname EOH".to_string()),
            ..NativeTrajectoryOptions::default()
        },
    )
    .unwrap();

    assert_eq!(report.frames_read, 2);
    assert_eq!(report.frames_written, 2);
    assert!(out.is_file());
}

#[test]
fn native_mapping_supports_prmtop_topology() {
    let dir = tempfile::tempdir().unwrap();
    let top = dir.path().join("system.prmtop");
    fs::write(
        &top,
        concat!(
            "%FLAG TITLE\n",
            "%FORMAT(20a4)\n",
            "TEST PRMTOP\n",
            "%FLAG POINTERS\n",
            "%FORMAT(10I8)\n",
            "       2       1       0       0       0       0       0       0       0       0\n",
            "%FLAG ATOM_NAME\n",
            "%FORMAT(20a4)\n",
            "H1  H2  \n",
            "%FLAG CHARGE\n",
            "%FORMAT(5E16.8)\n",
            "  4.20000000E-01  4.20000000E-01\n",
            "%FLAG MASS\n",
            "%FORMAT(5E16.8)\n",
            "  1.00800000E+00  1.00800000E+00\n",
            "%FLAG RESIDUE_LABEL\n",
            "%FORMAT(20a4)\n",
            "SOL \n",
            "%FLAG RESIDUE_POINTER\n",
            "%FORMAT(10I8)\n",
            "       1\n",
            "%FLAG ATOMIC_NUMBER\n",
            "%FORMAT(10I8)\n",
            "       1       1\n",
            "%FLAG ATOM_TYPE_INDEX\n",
            "%FORMAT(10I8)\n",
            "       1       1\n"
        ),
    )
    .unwrap();

    let traj = dir.path().join("solvated.xtc");
    let mut writer = XtcWriter::create(&traj, 2).unwrap();
    writer
        .write_frame(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            Box3::Orthorhombic {
                lx: 10.0,
                ly: 10.0,
                lz: 10.0,
            },
            0,
            Some(0.0),
        )
        .unwrap();
    writer.flush().unwrap();

    let out = dir.path().join("sol_cg.xtc");
    let report = map_native_trajectory(
        &traj,
        Some(&out),
        &BeadMapping {
            bead_names: vec!["SP1".to_string()],
            atom_indices: vec![vec![0]],
        },
        &[],
        &NativeTrajectoryOptions {
            topology: Some(top.to_string_lossy().to_string()),
            topology_format: Some("prmtop".to_string()),
            format: Some("xtc".to_string()),
            target_selection: Some("resname SOL".to_string()),
            ..NativeTrajectoryOptions::default()
        },
    )
    .unwrap();

    assert_eq!(report.frames_read, 1);
    assert_eq!(report.frames_written, 1);
    assert!(out.is_file());
}
