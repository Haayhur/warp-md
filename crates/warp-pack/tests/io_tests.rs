use std::fs;
use std::path::Path;

use warp_pack::config::OutputSpec;
use warp_pack::io::{read_molecule, write_output};

mod common;
use common::{sample_output, temp_path, write_text};

#[test]
fn read_molecule_parses_pdb_with_conect() {
    let pdb_path = temp_path("sample.pdb");
    write_text(
        &pdb_path,
        "ATOM      1  C   MOL A   1       0.000   0.000   0.000           C\n\
ATOM      2  O   MOL A   1       1.000   0.000   0.000           O\n\
CONECT    1    2\n\
END\n",
    );
    let mol = read_molecule(Path::new(&pdb_path), None, false, false, None).expect("pdb");
    assert_eq!(mol.atoms.len(), 2);
    assert_eq!(mol.bonds.len(), 1);
    let mol_no_conect = read_molecule(Path::new(&pdb_path), None, true, false, None).expect("pdb");
    assert_eq!(mol_no_conect.bonds.len(), 0);
    let _ = fs::remove_file(&pdb_path);
}

#[test]
fn read_molecule_parses_all_models_from_pdb() {
    let pdb_path = temp_path("sample_multimodel.pdb");
    write_text(
        &pdb_path,
        "MODEL        1\n\
ATOM      1  C   MOL A   1       1.000   0.000   0.000           C\n\
ENDMDL\n\
MODEL        2\n\
ATOM      1  O   MOL A   1       2.000   0.000   0.000           O\n\
ENDMDL\n\
END\n",
    );
    let mol = read_molecule(Path::new(&pdb_path), None, false, false, None).expect("pdb");
    assert_eq!(mol.atoms.len(), 2);
    assert!((mol.atoms[0].position.x - 1.0).abs() < 1e-6);
    assert!((mol.atoms[1].position.x - 2.0).abs() < 1e-6);
    let _ = fs::remove_file(&pdb_path);
}

#[test]
fn read_molecule_parses_xyz() {
    let xyz_path = temp_path("sample.xyz");
    write_text(&xyz_path, "2\ncomment\nC 0.0 0.0 0.0\nO 1.0 0.0 0.0\n");
    let mol = read_molecule(Path::new(&xyz_path), None, false, false, None).expect("xyz");
    assert_eq!(mol.atoms.len(), 2);
    let _ = fs::remove_file(&xyz_path);
}

#[test]
fn read_molecule_parses_mol2() {
    let mol2_path = temp_path("sample.mol2");
    write_text(
        &mol2_path,
        "@<TRIPOS>MOLECULE\nwarp\n2 0 0 0 0\nSMALL\nUSER_CHARGES\n@<TRIPOS>ATOM\n\
1 C1 0.0 0.0 0.0 C 1 MOL 0.0\n\
2 O1 1.0 0.0 0.0 O 1 MOL 0.0\n",
    );
    let mol = read_molecule(Path::new(&mol2_path), None, false, false, None).expect("mol2");
    assert_eq!(mol.atoms.len(), 2);
    let _ = fs::remove_file(&mol2_path);
}

#[test]
fn read_molecule_parses_tinker_xyz() {
    let txyz_path = temp_path("sample.txyz");
    write_text(
        &txyz_path,
        "2 sample\n1 C 0.0 0.0 0.0 1\n2 O 1.0 0.0 0.0 1\n",
    );
    let mol =
        read_molecule(Path::new(&txyz_path), Some("tinker"), false, false, None).expect("tinker");
    assert_eq!(mol.atoms.len(), 2);
    let _ = fs::remove_file(&txyz_path);
}

#[test]
fn read_molecule_parses_gro() {
    let gro_path = temp_path("sample.gro");
    write_text(
        &gro_path,
        "test\n    2\n    1WAT     O    1   0.000   0.000   0.000\n    1WAT    H1    2   0.100   0.000   0.000\n   1.0 1.0 1.0\n",
    );
    let mol = read_molecule(Path::new(&gro_path), Some("gro"), false, false, None).expect("gro");
    assert_eq!(mol.atoms.len(), 2);
    let _ = fs::remove_file(&gro_path);
}

#[test]
fn read_molecule_parses_pdbx() {
    let pdbx_path = temp_path("sample.cif");
    write_text(
        &pdbx_path,
        "data_test\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n_atom_site.label_atom_id\n_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.label_seq_id\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\nATOM 1 O OW WAT A 1 0.0 0.0 0.0\nATOM 2 H HW1 WAT A 1 0.1 0.0 0.0\n",
    );
    let mol = read_molecule(Path::new(&pdbx_path), Some("cif"), false, false, None).expect("pdbx");
    assert_eq!(mol.atoms.len(), 2);
    let _ = fs::remove_file(&pdbx_path);
}

#[test]
fn read_molecule_parses_lammps() {
    let lmp_path = temp_path("sample.lmp");
    write_text(
        &lmp_path,
        "LAMMPS data\n\n2 atoms\n2 atom types\n1 bonds\n\n0.0 10.0 xlo xhi\n0.0 10.0 ylo yhi\n0.0 10.0 zlo zhi\n\nMasses\n\n1 15.999 # O\n2 1.008 # H\n\nAtoms # full\n\n1 1 1 0.0 0.0 0.0 0.0\n2 1 2 -0.1 1.0 0.0 0.0\n\nBonds\n\n1 1 1 2\n",
    );
    let mol =
        read_molecule(Path::new(&lmp_path), Some("lammps"), false, false, None).expect("lammps");
    assert_eq!(mol.atoms.len(), 2);
    assert_eq!(mol.bonds.len(), 1);
    assert_eq!(mol.atoms[0].element, "O");
    assert_eq!(mol.atoms[1].element, "H");
    let _ = fs::remove_file(&lmp_path);
}

#[test]
fn read_molecule_parses_crd() {
    let crd_path = temp_path("sample.crd");
    write_text(
        &crd_path,
        "* TITLE\n* test\n*\n    2 EXT\n         1         1  WAT      O     0.0000000000     0.0000000000     0.0000000000  WAT      1     0.0\n         2         1  WAT      H1    0.1000000000     0.0000000000     0.0000000000  WAT      1     0.0\n",
    );
    let mol = read_molecule(Path::new(&crd_path), Some("crd"), false, false, None).expect("crd");
    assert_eq!(mol.atoms.len(), 2);
    let _ = fs::remove_file(&crd_path);
}

#[test]
fn write_output_formats() {
    let out = sample_output();

    let pdb_path = temp_path("out.pdb");
    let pdb_spec = OutputSpec {
        path: pdb_path.to_string_lossy().to_string(),
        format: "pdb".into(),
        scale: Some(1.0),
    };
    write_output(&out, &pdb_spec, true, 0.0, true, false).expect("pdb write");
    let pdb_text = fs::read_to_string(&pdb_path).expect("pdb read");
    assert!(pdb_text.contains("CRYST1"));
    assert!(pdb_text.contains("CONECT"));
    assert!(pdb_text.contains("TER"));
    let _ = fs::remove_file(&pdb_path);

    let xyz_path = temp_path("out.xyz");
    let xyz_spec = OutputSpec {
        path: xyz_path.to_string_lossy().to_string(),
        format: "xyz".into(),
        scale: Some(1.0),
    };
    write_output(&out, &xyz_spec, false, 0.0, false, false).expect("xyz write");
    let xyz_text = fs::read_to_string(&xyz_path).expect("xyz read");
    assert!(xyz_text.lines().next().unwrap_or("").starts_with('2'));
    let _ = fs::remove_file(&xyz_path);

    let pdbx_path = temp_path("out.pdbx");
    let pdbx_spec = OutputSpec {
        path: pdbx_path.to_string_lossy().to_string(),
        format: "pdbx".into(),
        scale: Some(1.0),
    };
    write_output(&out, &pdbx_spec, false, 0.0, false, false).expect("pdbx write");
    let pdbx_text = fs::read_to_string(&pdbx_path).expect("pdbx read");
    assert!(pdbx_text.contains("data_warp_pack"));
    let _ = fs::remove_file(&pdbx_path);

    let gro_path = temp_path("out.gro");
    let gro_spec = OutputSpec {
        path: gro_path.to_string_lossy().to_string(),
        format: "gro".into(),
        scale: Some(1.0),
    };
    write_output(&out, &gro_spec, false, 0.0, false, false).expect("gro write");
    let gro_text = fs::read_to_string(&gro_path).expect("gro read");
    assert!(gro_text
        .lines()
        .next()
        .unwrap_or("")
        .starts_with("warp_pack"));
    let _ = fs::remove_file(&gro_path);

    let lmp_path = temp_path("out.lammps");
    let lmp_spec = OutputSpec {
        path: lmp_path.to_string_lossy().to_string(),
        format: "lammps".into(),
        scale: Some(1.0),
    };
    write_output(&out, &lmp_spec, false, 0.0, false, false).expect("lammps write");
    let lmp_text = fs::read_to_string(&lmp_path).expect("lammps read");
    assert!(lmp_text.contains("lammps data"));
    assert!(lmp_text.contains("Masses"));
    let _ = fs::remove_file(&lmp_path);

    let mol2_path = temp_path("out.mol2");
    let mol2_spec = OutputSpec {
        path: mol2_path.to_string_lossy().to_string(),
        format: "mol2".into(),
        scale: Some(1.0),
    };
    write_output(&out, &mol2_spec, false, 0.0, false, false).expect("mol2 write");
    let mol2_text = fs::read_to_string(&mol2_path).expect("mol2 read");
    assert!(mol2_text.contains("@<TRIPOS>MOLECULE"));
    let _ = fs::remove_file(&mol2_path);

    let crd_path = temp_path("out.crd");
    let crd_spec = OutputSpec {
        path: crd_path.to_string_lossy().to_string(),
        format: "crd".into(),
        scale: Some(1.0),
    };
    write_output(&out, &crd_spec, false, 0.0, false, false).expect("crd write");
    let crd_text = fs::read_to_string(&crd_path).expect("crd read");
    assert!(crd_text.contains("* TITLE"));
    let _ = fs::remove_file(&crd_path);
}
