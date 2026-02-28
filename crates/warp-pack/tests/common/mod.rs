#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use warp_pack::config::{BoxSpec, PackConfig, StructureSpec};
use warp_pack::geom::Vec3;
use warp_pack::pack::{AtomRecord, AtomRecordKind, PackOutput};

pub fn temp_path(label: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let label_path = Path::new(label);
    let stem = label_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(label);
    let ext = label_path.extension().and_then(|s| s.to_str());
    let filename = if let Some(ext) = ext {
        format!(
            "warp_pack_test_{stem}_{}_{}.{}",
            std::process::id(),
            nanos,
            ext
        )
    } else {
        format!("warp_pack_test_{label}_{}_{}", std::process::id(), nanos)
    };
    path.push(filename);
    path
}

pub fn write_text(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write temp file");
}

pub fn base_structure(path: &Path) -> StructureSpec {
    StructureSpec {
        path: path.to_string_lossy().to_string(),
        count: 1,
        name: None,
        topology: None,
        restart_from: None,
        restart_to: None,
        fixed_eulers: None,
        chain: None,
        changechains: false,
        segid: None,
        connect: true,
        format: None,
        rotate: true,
        fixed: false,
        positions: None,
        translate: None,
        center: true,
        min_distance: None,
        resnumbers: None,
        maxmove: None,
        nloop: None,
        nloop0: None,
        constraints: Vec::new(),
        radius: None,
        fscale: None,
        short_radius: None,
        short_radius_scale: None,
        atom_overrides: Vec::new(),
        atom_constraints: Vec::new(),
        rot_bounds: None,
    }
}

pub fn base_config(structures: Vec<StructureSpec>) -> PackConfig {
    PackConfig {
        box_: BoxSpec {
            size: [10.0, 10.0, 10.0],
            shape: "orthorhombic".into(),
        },
        structures,
        seed: Some(0),
        max_attempts: Some(1000),
        min_distance: Some(2.0),
        filetype: None,
        add_box_sides: false,
        add_box_sides_fix: None,
        add_amber_ter: false,
        amber_ter_preserve: false,
        hexadecimal_indices: false,
        ignore_conect: false,
        non_standard_conect: true,
        pbc: false,
        pbc_min: None,
        pbc_max: None,
        maxit: None,
        nloop: None,
        nloop0: None,
        avoid_overlap: true,
        packall: false,
        check: false,
        sidemax: None,
        discale: None,
        precision: None,
        chkgrad: false,
        iprint1: None,
        iprint2: None,
        use_short_tol: false,
        short_tol_dist: None,
        short_tol_scale: None,
        movefrac: None,
        movebadrandom: false,
        disable_movebad: true,
        maxmove: None,
        randominitialpoint: false,
        fbins: None,
        writeout: None,
        writebad: false,
        restart_from: None,
        restart_to: None,
        relax_steps: None,
        relax_step: None,
        gencan_maxit: None,
        gencan_step: None,
        write_crd: None,
        output: None,
    }
}

pub fn sample_output() -> PackOutput {
    PackOutput {
        atoms: vec![
            AtomRecord {
                record_kind: AtomRecordKind::Atom,
                name: "C".into(),
                element: "C".into(),
                resname: "MOL".into(),
                resid: 1,
                chain: 'A',
                segid: String::new(),
                charge: 0.1,
                position: Vec3::new(1.0, 2.0, 3.0),
                mol_id: 1,
            },
            AtomRecord {
                record_kind: AtomRecordKind::Atom,
                name: "O".into(),
                element: "O".into(),
                resname: "MOL".into(),
                resid: 1,
                chain: 'A',
                segid: String::new(),
                charge: -0.1,
                position: Vec3::new(2.0, 2.0, 3.0),
                mol_id: 1,
            },
        ],
        bonds: vec![(0, 1)],
        box_size: [10.0, 10.0, 10.0],
        ter_after: vec![1],
    }
}
