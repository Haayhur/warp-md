use std::fs;

mod common;
use common::{base_config, base_structure, temp_path, write_text};
use warp_pack::inp::parse_packmol_inp;

#[test]
fn normalized_sets_defaults() {
    let pdb_path = temp_path("defaults.pdb");
    write_text(
        &pdb_path,
        "ATOM      1  H   MOL A   1       0.000   0.000   0.000           H\nEND\n",
    );
    let cfg = base_config(vec![base_structure(&pdb_path)]);
    let norm = cfg.normalized().expect("normalized config");
    assert_eq!(norm.seed, Some(0));
    assert_eq!(norm.max_attempts, Some(1000));
    assert_eq!(norm.min_distance, Some(2.0));
    assert_eq!(norm.nloop0, Some(20));
    assert_eq!(norm.nloop, Some(200));
    assert!((norm.precision.unwrap() - 1.0e-2).abs() < 1e-8);
    assert!((norm.movefrac.unwrap() - 0.05).abs() < 1e-8);
    assert!((norm.fbins.unwrap() - 3.0f32.sqrt()).abs() < 1e-6);
    assert_eq!(norm.relax_steps, Some(0));
    assert_eq!(norm.relax_step, Some(0.5));
    let _ = fs::remove_file(&pdb_path);
}

#[test]
fn normalized_uses_maxit_as_gencan_alias() {
    let pdb_path = temp_path("maxit.pdb");
    write_text(
        &pdb_path,
        "ATOM      1  H   MOL A   1       0.000   0.000   0.000           H\nEND\n",
    );
    let mut cfg = base_config(vec![base_structure(&pdb_path)]);
    cfg.max_attempts = None;
    cfg.maxit = Some(7);
    let norm = cfg.normalized().expect("normalized config");
    assert_eq!(norm.max_attempts, Some(100000));
    assert_eq!(norm.gencan_maxit, Some(7));
    let _ = fs::remove_file(&pdb_path);
}

#[test]
fn normalized_uses_default_max_attempts_when_missing() {
    let pdb_path = temp_path("default_max_attempts.pdb");
    write_text(
        &pdb_path,
        "ATOM      1  H   MOL A   1       0.000   0.000   0.000           H\nEND\n",
    );
    let mut cfg = base_config(vec![base_structure(&pdb_path)]);
    cfg.max_attempts = None;
    cfg.maxit = None;
    let norm = cfg.normalized().expect("normalized config");
    assert_eq!(norm.max_attempts, Some(100000));
    let _ = fs::remove_file(&pdb_path);
}

#[test]
fn normalized_uses_packmol_default_gencan_maxit() {
    let pdb_path = temp_path("default_gencan_maxit.pdb");
    write_text(
        &pdb_path,
        "ATOM      1  H   MOL A   1       0.000   0.000   0.000           H\nEND\n",
    );
    let mut cfg = base_config(vec![base_structure(&pdb_path)]);
    cfg.gencan_maxit = None;
    cfg.maxit = None;
    let norm = cfg.normalized().expect("normalized config");
    assert_eq!(norm.gencan_maxit, Some(20));
    let _ = fs::remove_file(&pdb_path);
}

#[test]
fn json_default_uses_packmol_gencan_defaults() {
    use serde_json::json;

    let pdb_path = temp_path("json_default_gencan_defaults.pdb");
    write_text(
        &pdb_path,
        "ATOM      1  H   MOL A   1       0.000   0.000   0.000           H\nEND\n",
    );
    let cfg: warp_pack::config::PackConfig = serde_json::from_value(json!({
        "box": { "size": [10.0, 10.0, 10.0], "shape": "orthorhombic" },
        "structures": [{ "path": pdb_path.to_string_lossy(), "count": 1 }]
    }))
    .expect("deserialize config");
    let norm = cfg.normalized().expect("normalized config");
    assert_eq!(norm.gencan_maxit, Some(20));
    let _ = fs::remove_file(&pdb_path);
}

#[test]
fn validate_rejects_bad_box() {
    let pdb_path = temp_path("bad_box.pdb");
    write_text(
        &pdb_path,
        "ATOM      1  H   MOL A   1       0.000   0.000   0.000           H\nEND\n",
    );
    let mut cfg = base_config(vec![base_structure(&pdb_path)]);
    cfg.box_.size = [-1.0, 10.0, 10.0];
    assert!(cfg.validate().is_err());
    cfg.box_.size = [10.0, 10.0, 10.0];
    cfg.box_.shape = "triclinic".into();
    assert!(cfg.validate().is_err());
    let _ = fs::remove_file(&pdb_path);
}

#[test]
fn validate_structure_constraints() {
    let pdb_path = temp_path("structure_invalid.pdb");
    write_text(
        &pdb_path,
        "ATOM      1  H   MOL A   1       0.000   0.000   0.000           H\nEND\n",
    );
    let mut cfg = base_config(vec![base_structure(&pdb_path)]);
    cfg.sidemax = Some(5.0);
    assert!(cfg.validate().is_err());

    let mut bad = base_structure(&pdb_path);
    bad.fixed = true;
    bad.count = 2;
    cfg = base_config(vec![bad]);
    assert!(cfg.validate().is_err());

    let mut bad = base_structure(&pdb_path);
    bad.positions = Some(vec![[0.0, 0.0, 0.0]]);
    bad.restart_from = Some("restart.dat".into());
    cfg = base_config(vec![bad]);
    assert!(cfg.validate().is_err());

    let _ = fs::remove_file(&pdb_path);
}

#[test]
fn parse_constrain_rotation_maps_axes_and_degrees() {
    let inp = r#"
tolerance 2.0
output out.pdb
structure foo.pdb
  number 1
  constrain_rotation x 90 10
  constrain_rotation y 0 5
  constrain_rotation z -45 15
end structure
"#;
    let cfg = parse_packmol_inp(inp).expect("parse packmol inp");
    let bounds = cfg.structures[0].rot_bounds.expect("rot bounds");
    let deg = std::f32::consts::PI / 180.0;
    assert!((bounds[2][0] - (90.0 - 10.0) * deg).abs() < 1.0e-6);
    assert!((bounds[2][1] - (90.0 + 10.0) * deg).abs() < 1.0e-6);
    assert!((bounds[0][0] - (0.0 - 5.0) * deg).abs() < 1.0e-6);
    assert!((bounds[0][1] - (0.0 + 5.0) * deg).abs() < 1.0e-6);
    assert!((bounds[1][0] - (-45.0 - 15.0) * deg).abs() < 1.0e-6);
    assert!((bounds[1][1] - (-45.0 + 15.0) * deg).abs() < 1.0e-6);
}

#[test]
fn parse_atoms_block_constraints() {
    let inp = r#"
tolerance 2.0
output out.pdb
structure lip.pdb
  number 1
  atoms 1 2
    below plane 0. 0. 1. 2.
  end atoms
  atoms 3
    inside sphere 0. 0. 0. 5.
  end atoms
end structure
"#;
    let cfg = parse_packmol_inp(inp).expect("parse packmol inp");
    let spec = &cfg.structures[0];
    assert_eq!(spec.atom_constraints.len(), 2);
    assert_eq!(spec.atom_constraints[0].indices, vec![1, 2]);
    assert_eq!(spec.atom_constraints[1].indices, vec![3]);
}

#[test]
fn parse_atoms_range_expands_indices() {
    let inp = r#"
tolerance 2.0
output out.pdb
structure lip.pdb
  number 1
  atoms 2 4
    inside sphere 0. 0. 0. 5.
  end atoms
end structure
"#;
    let cfg = parse_packmol_inp(inp).expect("parse packmol inp");
    let spec = &cfg.structures[0];
    assert_eq!(spec.atom_constraints.len(), 1);
    assert_eq!(spec.atom_constraints[0].indices, vec![2, 3, 4]);
}

#[test]
fn parse_maxit_only_controls_gencan_iterations() {
    let inp = r#"
tolerance 2.0
maxit 20
output out.pdb
structure foo.pdb
  number 1
  inside box 0. 0. 0. 10. 10. 10.
end structure
"#;
    let cfg = parse_packmol_inp(inp).expect("parse packmol inp");
    let norm = cfg.normalized().expect("normalized config");
    assert_eq!(norm.maxit, Some(20));
    assert_eq!(norm.gencan_maxit, Some(20));
    assert_eq!(norm.max_attempts, Some(100000));
}
