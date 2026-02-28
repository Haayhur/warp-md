use std::fs;
use std::path::{Path, PathBuf};

use warp_pack::config::AtomConstraintSpec;
use warp_pack::constraints::{ConstraintMode, ConstraintSpec, ShapeSpec};
mod common;
use common::{base_config, base_structure, temp_path, write_text};

fn water_model_path(model: &str) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    root.join("../../python/warp_md/pack/data")
        .join(format!("{model}.pdb"))
}

fn count_atoms(path: &Path) -> usize {
    let contents = fs::read_to_string(path).expect("read water model");
    contents
        .lines()
        .filter(|line| line.starts_with("ATOM") || line.starts_with("HETATM"))
        .count()
}

#[test]
fn pack_places_with_min_distance() {
    let pdb_path = temp_path("min_dist.pdb");
    write_text(
        &pdb_path,
        "ATOM      1  H   MOL A   1       0.000   0.000   0.000           H\nEND\n",
    );
    let mut spec = base_structure(&pdb_path);
    spec.count = 2;
    let mut cfg = base_config(vec![spec]);
    cfg.min_distance = Some(2.5);
    let out = warp_pack::pack::run(&cfg).expect("pack");
    assert_eq!(out.atoms.len(), 2);
    let a = out.atoms[0].position;
    let b = out.atoms[1].position;
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    assert!(dist >= 2.5 - 1e-6);
    let _ = fs::remove_file(&pdb_path);
}

#[test]
fn pack_fixed_constraints_enforced() {
    let pdb_path = temp_path("constraint.pdb");
    write_text(
        &pdb_path,
        "ATOM      1  H   MOL A   1       0.000   0.000   0.000           H\nEND\n",
    );
    let mut spec_ok = base_structure(&pdb_path);
    spec_ok.fixed = true;
    spec_ok.positions = Some(vec![[0.0, 0.0, 0.0]]);
    spec_ok.constraints.push(ConstraintSpec {
        mode: ConstraintMode::Inside,
        shape: ShapeSpec::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        },
    });
    let cfg_ok = base_config(vec![spec_ok]);
    warp_pack::pack::run(&cfg_ok).expect("inside constraint");

    let mut spec_bad = base_structure(&pdb_path);
    spec_bad.fixed = true;
    spec_bad.positions = Some(vec![[2.0, 0.0, 0.0]]);
    spec_bad.constraints.push(ConstraintSpec {
        mode: ConstraintMode::Inside,
        shape: ShapeSpec::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        },
    });
    let cfg_bad = base_config(vec![spec_bad]);
    assert!(warp_pack::pack::run(&cfg_bad).is_err());
    let _ = fs::remove_file(&pdb_path);
}

#[test]
fn pack_fixed_atom_constraints_enforced() {
    let pdb_path = temp_path("atom_constraint.pdb");
    write_text(
        &pdb_path,
        "ATOM      1  A   MOL A   1       0.000   0.000   0.000           C\n\
ATOM      2  B   MOL A   1       0.000   0.000   2.000           C\n\
END\n",
    );
    let plane = ConstraintSpec {
        mode: ConstraintMode::Above,
        shape: ShapeSpec::Plane {
            point: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
        },
    };

    let mut spec_ok = base_structure(&pdb_path);
    spec_ok.fixed = true;
    spec_ok.positions = Some(vec![[0.0, 0.0, 0.0]]);
    spec_ok.atom_constraints.push(AtomConstraintSpec {
        indices: vec![2],
        constraint: plane.clone(),
    });
    let cfg_ok = base_config(vec![spec_ok]);
    warp_pack::pack::run(&cfg_ok).expect("atom constraint satisfied");

    let mut spec_bad = base_structure(&pdb_path);
    spec_bad.fixed = true;
    spec_bad.positions = Some(vec![[0.0, 0.0, 0.0]]);
    spec_bad.atom_constraints.push(AtomConstraintSpec {
        indices: vec![1],
        constraint: plane,
    });
    let cfg_bad = base_config(vec![spec_bad]);
    assert!(warp_pack::pack::run(&cfg_bad).is_err());
    let _ = fs::remove_file(&pdb_path);
}

#[test]
fn pack_chain_override_and_changechains() {
    let pdb_path = temp_path("chain.pdb");
    write_text(
        &pdb_path,
        "ATOM      1  H   MOL A   1       0.000   0.000   0.000           H\nEND\n",
    );
    let mut spec = base_structure(&pdb_path);
    spec.fixed = true;
    spec.positions = Some(vec![[1.0, 1.0, 1.0]]);
    spec.chain = Some("Z".into());
    let cfg = base_config(vec![spec]);
    let out = warp_pack::pack::run(&cfg).expect("chain override");
    assert!(out.atoms.iter().all(|a| a.chain == 'Z'));

    let mut spec = base_structure(&pdb_path);
    spec.count = 2;
    spec.fixed = true;
    spec.positions = Some(vec![[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]]);
    spec.changechains = true;
    let cfg = base_config(vec![spec]);
    let out = warp_pack::pack::run(&cfg).expect("changechains");
    let mut chains_by_mol =
        out.atoms
            .iter()
            .fold(std::collections::HashMap::new(), |mut acc, atom| {
                acc.entry(atom.mol_id).or_insert(atom.chain);
                acc
            });
    assert_eq!(chains_by_mol.remove(&1), Some('A'));
    assert_eq!(chains_by_mol.remove(&2), Some('B'));
    let _ = fs::remove_file(&pdb_path);
}

#[test]
fn pack_resnumbers_mode_global_mol() {
    let pdb_path = temp_path("resnumbers.pdb");
    write_text(
        &pdb_path,
        "ATOM      1  H   MOL A   5       0.000   0.000   0.000           H\nEND\n",
    );
    let mut spec = base_structure(&pdb_path);
    spec.count = 2;
    spec.fixed = true;
    spec.positions = Some(vec![[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]]);
    spec.resnumbers = Some(3);
    let cfg = base_config(vec![spec]);
    let out = warp_pack::pack::run(&cfg).expect("resnumbers");
    let mut resid_by_mol =
        out.atoms
            .iter()
            .fold(std::collections::HashMap::new(), |mut acc, atom| {
                acc.entry(atom.mol_id).or_insert(atom.resid);
                acc
            });
    assert_eq!(resid_by_mol.remove(&1), Some(1));
    assert_eq!(resid_by_mol.remove(&2), Some(2));
    let _ = fs::remove_file(&pdb_path);
}

#[test]
fn pack_check_detects_overlap() {
    let pdb_path = temp_path("check_overlap.pdb");
    write_text(
        &pdb_path,
        "ATOM      1  H   MOL A   1       0.000   0.000   0.000           H\nEND\n",
    );
    let mut spec = base_structure(&pdb_path);
    spec.count = 2;
    spec.fixed = true;
    spec.positions = Some(vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    let mut cfg = base_config(vec![spec]);
    cfg.avoid_overlap = false;
    cfg.check = true;
    assert!(warp_pack::pack::run(&cfg).is_err());
    let _ = fs::remove_file(&pdb_path);
}

#[test]
fn gencan_reduces_overlap() {
    let pdb_path = temp_path("gencan.pdb");
    write_text(
        &pdb_path,
        "ATOM      1  H   MOL A   1       0.000   0.000   0.000           H\nEND\n",
    );
    let restart_path = temp_path("gencan.restart");
    write_text(
        &restart_path,
        "1.0 1.0 1.0 0.0 0.0 0.0\n1.0 1.0 1.0 0.0 0.0 0.0\n",
    );
    let mut spec = base_structure(&pdb_path);
    spec.count = 2;
    spec.center = false;
    spec.restart_from = Some(restart_path.to_string_lossy().to_string());
    let mut cfg = base_config(vec![spec]);
    cfg.min_distance = Some(2.0);
    cfg.avoid_overlap = false;
    cfg.gencan_maxit = Some(50);
    cfg.gencan_step = Some(0.2);
    cfg.relax_steps = Some(0);

    let out = warp_pack::pack::run(&cfg).expect("gencan run");
    let a = out.atoms[0].position;
    let b = out.atoms[1].position;
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    assert!(dist > 0.5);
    let _ = fs::remove_file(&pdb_path);
    let _ = fs::remove_file(&restart_path);
}

#[test]
fn pack_water_models_single_residue() {
    for model in ["spce", "tip3p", "tip4pew", "tip5p"] {
        let pdb_path = water_model_path(model);
        let atoms_per_mol = count_atoms(&pdb_path);
        let mut spec = base_structure(&pdb_path);
        spec.count = 10;
        let mut cfg = base_config(vec![spec]);
        cfg.box_.size = [20.0, 20.0, 20.0];
        cfg.min_distance = Some(2.0);
        cfg.max_attempts = Some(5000);
        cfg.check = true;

        let out = warp_pack::pack::run(&cfg).expect("pack water model");
        assert_eq!(out.atoms.len(), atoms_per_mol * 10);
        let unique_mols = out
            .atoms
            .iter()
            .map(|a| a.mol_id)
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(unique_mols.len(), 10);
    }
}

#[test]
fn pack_realistic_water_box() {
    let pdb_path = water_model_path("spce");
    let atoms_per_mol = count_atoms(&pdb_path);
    let mut spec = base_structure(&pdb_path);
    spec.count = 50;
    let mut cfg = base_config(vec![spec]);
    cfg.box_.size = [25.0, 25.0, 25.0];
    cfg.min_distance = Some(2.0);
    cfg.max_attempts = Some(8000);
    cfg.check = true;

    let out = warp_pack::pack::run(&cfg).expect("pack water box");
    assert_eq!(out.atoms.len(), atoms_per_mol * 50);
    let unique_mols = out
        .atoms
        .iter()
        .map(|a| a.mol_id)
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(unique_mols.len(), 50);
}
