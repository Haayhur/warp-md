use std::fs;

use warp_pack::io::read_amber_inpcrd;

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
        "%FLAG ATOM_NAME\n%FORMAT(20a4)\nH1 O1\n%FLAG RESIDUE_LABEL\n%FORMAT(20a4)\nWAT\n%FLAG RESIDUE_POINTER\n%FORMAT(10I8)\n1\n%FLAG ATOMIC_NUMBER\n%FORMAT(10I8)\n1 8\n%FLAG CHARGE\n%FORMAT(5E16.8)\n18.2223 -18.2223\n",
    );
    let mol = read_amber_inpcrd(&inpcrd_path, Some(&prmtop_path)).expect("parse inpcrd");
    assert_eq!(mol.atoms.len(), 2);
    assert_eq!(mol.atoms[0].name.trim(), "H1");
    assert_eq!(mol.atoms[1].element, "O");
    assert_eq!(mol.atoms[0].resname.trim(), "WAT");
    assert!((mol.atoms[0].charge - 1.0).abs() < 1e-4);
    let _ = fs::remove_file(&inpcrd_path);
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
