use std::fs;

use warp_pack::io::{read_amber_inpcrd, read_prmtop_topology, write_minimal_prmtop, AmberTopology};

mod common;
use common::{base_config, base_structure, temp_path, write_text};

#[test]
fn read_amber_inpcrd_parses_coords() {
    let path = temp_path("inpcrd.inpcrd");
    write_text(&path, "TEST\n1\n1.0 2.0 3.0\n");
    let mol = read_amber_inpcrd(&path, None).expect("parse inpcrd");
    assert_eq!(mol.atoms.len(), 1);
    let pos = mol.atoms[0].position;
    assert!((pos.x - 1.0).abs() < 1e-6);
    assert!((pos.y - 2.0).abs() < 1e-6);
    assert!((pos.z - 3.0).abs() < 1e-6);
    let _ = fs::remove_file(&path);
}

#[test]
fn avoid_overlap_controls_fixed_overlap() {
    let pdb_path = temp_path("single_pdb.pdb");
    write_text(
        &pdb_path,
        "ATOM      1  H   MOL A   1       0.000   0.000   0.000           H\nEND\n",
    );

    let mut s1 = base_structure(&pdb_path);
    s1.fixed = true;
    s1.positions = Some(vec![[0.0, 0.0, 0.0]]);
    let mut s2 = base_structure(&pdb_path);
    s2.fixed = true;
    s2.positions = Some(vec![[0.0, 0.0, 0.0]]);

    let mut cfg = base_config(vec![s1, s2]);
    cfg.avoid_overlap = false;
    let res = warp_pack::pack::run(&cfg);
    if let Err(err) = &res {
        eprintln!("avoid_overlap=false error: {err}");
    }
    assert!(res.is_ok());

    cfg.avoid_overlap = true;
    assert!(warp_pack::pack::run(&cfg).is_err());
    let _ = fs::remove_file(&pdb_path);
}

#[test]
fn short_tol_dist_validation_fails_when_too_large() {
    let pdb_path = temp_path("single_pdb_short.pdb");
    write_text(
        &pdb_path,
        "ATOM      1  H   MOL A   1       0.000   0.000   0.000           H\nEND\n",
    );
    let s1 = base_structure(&pdb_path);
    let mut cfg = base_config(vec![s1]);
    cfg.min_distance = Some(2.0);
    cfg.short_tol_dist = Some(3.0);
    assert!(cfg.normalized().is_err());
    let _ = fs::remove_file(&pdb_path);
}

#[test]
fn fbins_validation_rejects_too_small() {
    let pdb_path = temp_path("single_pdb_fbins.pdb");
    write_text(
        &pdb_path,
        "ATOM      1  H   MOL A   1       0.000   0.000   0.000           H\nEND\n",
    );
    let s1 = base_structure(&pdb_path);
    let mut cfg = base_config(vec![s1]);
    cfg.fbins = Some(0.5);
    assert!(cfg.normalized().is_err());
    let _ = fs::remove_file(&pdb_path);
}

#[test]
fn read_amber_inpcrd_without_count_inferrs_atoms() {
    let path = temp_path("inpcrd_no_count.inpcrd");
    write_text(&path, "TEST\n1.0 2.0 3.0 4.0 5.0 6.0\n");
    let mol = read_amber_inpcrd(&path, None).expect("parse inpcrd");
    assert_eq!(mol.atoms.len(), 2);
    let _ = fs::remove_file(&path);
}

#[test]
fn read_amber_inpcrd_with_prmtop_metadata() {
    let inpcrd_path = temp_path("inpcrd_topo.inpcrd");
    let prmtop_path = temp_path("topo.prmtop");
    write_text(&inpcrd_path, "TEST\n2\n1.0 2.0 3.0 4.0 5.0 6.0\n");
    write_text(
        &prmtop_path,
        "%FLAG ATOM_NAME\n%FORMAT(20a4)\nH1 O1\n%FLAG RESIDUE_LABEL\n%FORMAT(20a4)\nWAT\n%FLAG RESIDUE_POINTER\n%FORMAT(10I8)\n1\n%FLAG ATOMIC_NUMBER\n%FORMAT(10I8)\n1 8\n%FLAG CHARGE\n%FORMAT(5E16.8)\n18.2223 -18.2223\n%FLAG BONDS_INC_HYDROGEN\n%FORMAT(10I8)\n0 3 1\n%FLAG BONDS_WITHOUT_HYDROGEN\n%FORMAT(10I8)\n\n",
    );
    let mol = read_amber_inpcrd(&inpcrd_path, Some(&prmtop_path)).expect("parse inpcrd");
    assert_eq!(mol.atoms.len(), 2);
    assert_eq!(mol.atoms[0].name.trim(), "H1");
    assert_eq!(mol.atoms[1].element, "O");
    assert_eq!(mol.atoms[0].resname.trim(), "WAT");
    assert!((mol.atoms[0].charge - 1.0).abs() < 1e-4);
    assert_eq!(mol.bonds, vec![(0, 1)]);
    let _ = fs::remove_file(&inpcrd_path);
    let _ = fs::remove_file(&prmtop_path);
}

#[test]
fn prmtop_parse_preserves_term_order_and_multi_term_dihedrals() {
    let prmtop_path = temp_path("ordered_terms.prmtop");
    write_text(
        &prmtop_path,
        "%FLAG ATOM_NAME\n%FORMAT(20a4)\nC1 C2 C3 C4\n%FLAG RESIDUE_LABEL\n%FORMAT(20a4)\nMOL\n%FLAG RESIDUE_POINTER\n%FORMAT(10I8)\n1\n%FLAG ATOMIC_NUMBER\n%FORMAT(10I8)\n6 6 6 6\n%FLAG CHARGE\n%FORMAT(5E16.8)\n  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00\n%FLAG BONDS_WITHOUT_HYDROGEN\n%FORMAT(10I8)\n       6       3       7       3       0       2\n%FLAG ANGLES_WITHOUT_HYDROGEN\n%FORMAT(10I8)\n       6       3       0      11       9       6       3      12\n%FLAG DIHEDRALS_WITHOUT_HYDROGEN\n%FORMAT(10I8)\n       0       3       6       9      21       0       3       6       9      22\n",
    );

    let parsed = read_prmtop_topology(&prmtop_path).expect("parse prmtop");
    assert_eq!(parsed.bonds, vec![(1, 2), (0, 1)]);
    assert_eq!(parsed.bond_type_indices, vec![7, 2]);
    assert_eq!(parsed.angles, vec![[2, 1, 0], [3, 2, 1]]);
    assert_eq!(parsed.angle_type_indices, vec![11, 12]);
    assert_eq!(parsed.dihedrals, vec![[0, 1, 2, 3], [0, 1, 2, 3]]);
    assert_eq!(parsed.dihedral_type_indices, vec![21, 22]);

    let _ = fs::remove_file(&prmtop_path);
}

#[test]
fn prmtop_roundtrip_preserves_rich_sections() {
    let prmtop_path = temp_path("roundtrip.prmtop");
    let topology = AmberTopology {
        atom_names: vec!["C1".into(), "C2".into(), "O1".into()],
        residue_labels: vec!["RPT".into(), "CAP".into()],
        residue_pointers: vec![1, 3],
        atomic_numbers: vec![6, 6, 8],
        masses: vec![12.01, 12.01, 16.0],
        charges: vec![0.5, -0.25, -0.25],
        atom_type_indices: vec![1, 1, 2],
        amber_atom_types: vec!["CT".into(), "CT".into(), "OS".into()],
        radii: vec![1.7, 1.7, 1.5],
        screen: vec![0.8, 0.8, 0.85],
        bonds: vec![(0, 1), (1, 2)],
        bond_type_indices: vec![1, 2],
        bond_force_constants: vec![310.0, 450.0],
        bond_equil_values: vec![1.53, 1.23],
        angles: vec![[0, 1, 2]],
        angle_type_indices: vec![1],
        angle_force_constants: vec![55.0],
        angle_equil_values: vec![109.5],
        dihedrals: vec![[0, 1, 2, 0]],
        dihedral_type_indices: vec![1],
        dihedral_force_constants: vec![0.75],
        dihedral_periodicities: vec![3.0],
        dihedral_phases: vec![0.0],
        scee_scale_factors: vec![1.2],
        scnb_scale_factors: vec![2.0],
        solty: vec![0.0, 0.0],
        impropers: vec![[0, 1, 2, 0]],
        improper_type_indices: vec![1],
        excluded_atoms: vec![vec![2, 3], vec![1, 3], vec![1, 2]],
        nonbonded_parm_index: vec![1, 2, 2, 3],
        lennard_jones_acoef: vec![1.0, 1.5, 2.0, 2.5],
        lennard_jones_bcoef: vec![0.5, 0.75, 1.0, 1.25],
        lennard_jones_14_acoef: vec![0.8, 1.2, 1.6, 2.0],
        lennard_jones_14_bcoef: vec![0.4, 0.6, 0.8, 1.0],
        hbond_acoef: vec![0.0],
        hbond_bcoef: vec![0.0],
        hbcut: vec![0.0],
        tree_chain_classification: vec!["M".into(), "M".into(), "E".into()],
        join_array: vec![0, 0, 0],
        irotat: vec![0, 0, 0],
        solvent_pointers: vec![2, 1, 2],
        atoms_per_molecule: vec![3],
        box_dimensions: vec![90.0, 10.0, 10.0, 10.0],
        radius_set: Some("modified Bondi radii".into()),
        ipol: 0,
    };
    write_minimal_prmtop(prmtop_path.to_string_lossy().as_ref(), &topology).expect("write prmtop");
    let written = fs::read_to_string(&prmtop_path).expect("read written prmtop");
    assert!(written.contains("%FLAG POINTERS"));
    assert!(written.contains("%FLAG SCEE_SCALE_FACTOR"));
    assert!(written.contains("%FLAG SOLTY"));
    assert!(written.contains("%FLAG HBOND_ACOEF"));
    assert!(written.contains("%FLAG RADIUS_SET"));
    let parsed = read_prmtop_topology(&prmtop_path).expect("read prmtop");
    assert_eq!(parsed.atom_names, topology.atom_names);
    assert_eq!(parsed.residue_labels, topology.residue_labels);
    assert_eq!(parsed.atom_type_indices, topology.atom_type_indices);
    assert_eq!(parsed.amber_atom_types, topology.amber_atom_types);
    assert_eq!(parsed.charges, topology.charges);
    assert_eq!(parsed.bonds, topology.bonds);
    assert_eq!(parsed.bond_type_indices, topology.bond_type_indices);
    assert_eq!(parsed.bond_force_constants, topology.bond_force_constants);
    assert_eq!(parsed.bond_equil_values, topology.bond_equil_values);
    assert_eq!(parsed.angles, topology.angles);
    assert_eq!(parsed.angle_type_indices, topology.angle_type_indices);
    assert_eq!(parsed.angle_force_constants, topology.angle_force_constants);
    assert_eq!(parsed.angle_equil_values, topology.angle_equil_values);
    assert!(!parsed.dihedrals.is_empty());
    assert_eq!(parsed.dihedral_type_indices, topology.dihedral_type_indices);
    assert_eq!(
        parsed.dihedral_force_constants,
        topology.dihedral_force_constants
    );
    assert_eq!(
        parsed.dihedral_periodicities,
        topology.dihedral_periodicities
    );
    assert_eq!(parsed.dihedral_phases, topology.dihedral_phases);
    assert_eq!(parsed.scee_scale_factors, topology.scee_scale_factors);
    assert_eq!(parsed.scnb_scale_factors, topology.scnb_scale_factors);
    assert_eq!(parsed.solty, topology.solty);
    assert_eq!(parsed.nonbonded_parm_index, topology.nonbonded_parm_index);
    assert_eq!(parsed.lennard_jones_acoef, topology.lennard_jones_acoef);
    assert_eq!(parsed.lennard_jones_bcoef, topology.lennard_jones_bcoef);
    assert_eq!(
        parsed.lennard_jones_14_acoef,
        topology.lennard_jones_14_acoef
    );
    assert_eq!(
        parsed.lennard_jones_14_bcoef,
        topology.lennard_jones_14_bcoef
    );
    assert_eq!(parsed.hbond_acoef, topology.hbond_acoef);
    assert_eq!(parsed.hbond_bcoef, topology.hbond_bcoef);
    assert_eq!(parsed.hbcut, topology.hbcut);
    assert_eq!(parsed.solvent_pointers, topology.solvent_pointers);
    assert_eq!(parsed.atoms_per_molecule, topology.atoms_per_molecule);
    assert_eq!(parsed.box_dimensions, topology.box_dimensions);
    assert_eq!(parsed.radius_set, topology.radius_set);
    assert_eq!(parsed.ipol, topology.ipol);
    let _ = fs::remove_file(&prmtop_path);
}

#[test]
fn prmtop_parse_rejects_pointer_length_mismatch() {
    let prmtop_path = temp_path("bad_pointer.prmtop");
    write_text(
        &prmtop_path,
        "%VERSION  VERSION_STAMP = V0001.000\n%FLAG POINTERS\n%FORMAT(10I8)\n       2       1       0       0       0       0       0       0       0       0\n       0       1       0       0       0       1       0       0       1       0\n       0       0       0       0       0       0       0       0       2       0\n       0       0\n%FLAG ATOM_NAME\n%FORMAT(20a4)\nC1\n%FLAG RESIDUE_LABEL\n%FORMAT(20a4)\nMOL\n%FLAG RESIDUE_POINTER\n%FORMAT(10I8)\n1\n%FLAG ATOMIC_NUMBER\n%FORMAT(10I8)\n6\n%FLAG MASS\n%FORMAT(5E16.8)\n  1.20000000E+01\n%FLAG CHARGE\n%FORMAT(5E16.8)\n  0.00000000E+00\n%FLAG ATOM_TYPE_INDEX\n%FORMAT(10I8)\n1\n%FLAG AMBER_ATOM_TYPE\n%FORMAT(20a4)\nCT\n%FLAG BOND_FORCE_CONSTANT\n%FORMAT(5E16.8)\n  3.00000000E+02\n",
    );
    let err = read_prmtop_topology(&prmtop_path).expect_err("pointer mismatch");
    assert!(err.to_string().contains("length mismatch"));
    let _ = fs::remove_file(&prmtop_path);
}

#[test]
fn restart_from_places_molecule() {
    let pdb_path = temp_path("restart_input.pdb");
    write_text(
        &pdb_path,
        "ATOM      1  A   MOL A   1       0.000   0.000   0.000           C\nATOM      2  B   MOL A   1       1.000   0.000   0.000           C\nEND\n",
    );
    let restart_path = temp_path("restart_from.dat");
    write_text(&restart_path, "5.0 0.0 0.0 0.0 0.0 0.0\n");
    let mut s1 = base_structure(&pdb_path);
    s1.center = false;
    s1.restart_from = Some(restart_path.to_string_lossy().to_string());
    let cfg = base_config(vec![s1]);
    let out = warp_pack::pack::run(&cfg).expect("pack with restart");
    assert!((out.atoms[0].position.x - 5.0).abs() < 1e-3);
    assert!((out.atoms[1].position.x - 6.0).abs() < 1e-3);
    let _ = fs::remove_file(&pdb_path);
    let _ = fs::remove_file(&restart_path);
}

#[test]
fn restart_to_writes_file() {
    let pdb_path = temp_path("restart_output.pdb");
    write_text(
        &pdb_path,
        "ATOM      1  A   MOL A   1       0.000   0.000   0.000           C\nEND\n",
    );
    let restart_path = temp_path("restart_to.dat");
    let mut s1 = base_structure(&pdb_path);
    s1.center = false;
    let mut cfg = base_config(vec![s1]);
    cfg.restart_to = Some(restart_path.to_string_lossy().to_string());
    warp_pack::pack::run(&cfg).expect("pack with restart_to");
    let contents = fs::read_to_string(&restart_path).expect("restart file");
    let first_line = contents.lines().next().unwrap_or("");
    assert!(first_line.split_whitespace().count() >= 6);
    let _ = fs::remove_file(&pdb_path);
    let _ = fs::remove_file(&restart_path);
}

#[test]
fn relax_increases_separation() {
    let pdb_path = temp_path("relax_input.pdb");
    write_text(
        &pdb_path,
        "ATOM      1  A   MOL A   1       0.000   0.000   0.000           C\nEND\n",
    );
    let restart_path = temp_path("relax_restart.dat");
    write_text(
        &restart_path,
        "0.0 0.0 0.0 0.0 0.0 0.0\n0.5 0.0 0.0 0.0 0.0 0.0\n",
    );
    let mut s1 = base_structure(&pdb_path);
    s1.count = 2;
    s1.center = false;
    s1.restart_from = Some(restart_path.to_string_lossy().to_string());
    let mut cfg = base_config(vec![s1]);
    cfg.avoid_overlap = false;
    cfg.relax_steps = Some(5);
    cfg.relax_step = Some(1.0);
    let out = warp_pack::pack::run(&cfg).expect("pack with relax");
    let dx = out.atoms[0].position.x - out.atoms[1].position.x;
    let dist = dx.abs();
    assert!(dist > 1.0);
    let _ = fs::remove_file(&pdb_path);
    let _ = fs::remove_file(&restart_path);
}
