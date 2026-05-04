use std::fs;
use std::path::Path;

use warp_pack::config::OutputSpec;
use warp_pack::io::{read_molecule, write_output};
use warp_structure::PdbAtomMetadata;

mod test_support;
use test_support::{sample_output, temp_path, write_text};

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
        "@<TRIPOS>MOLECULE\nwarp\n2 1 0 0 0\nSMALL\nUSER_CHARGES\n@<TRIPOS>ATOM\n\
1 C1 0.0 0.0 0.0 C 1 MOL 0.0\n\
2 O1 1.0 0.0 0.0 O 1 MOL 0.0\n\
@<TRIPOS>BOND\n\
1 1 2 1\n",
    );
    let mol = read_molecule(Path::new(&mol2_path), None, false, false, None).expect("mol2");
    assert_eq!(mol.atoms.len(), 2);
    assert_eq!(mol.bonds, vec![(0, 1)]);
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
    assert!((mol.atoms[1].position.x - 1.0).abs() < 1.0e-6);
    assert_eq!(
        mol.box_vectors,
        Some([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    );
    let _ = fs::remove_file(&gro_path);
}

#[test]
fn read_molecule_parses_g96() {
    let g96_path = temp_path("sample.g96");
    write_text(
        &g96_path,
        "TITLE\nsample\nEND\nPOSITION\n    1 WAT   O          1    0.000000000    0.000000000    0.000000000\n    1 WAT   H1         2    0.100000000    0.000000000    0.000000000\nEND\nBOX\n    1.000000000    2.000000000    3.000000000    0.100000000    0.200000000    0.300000000    0.400000000    0.500000000    0.600000000\nEND\n",
    );
    let mol = read_molecule(Path::new(&g96_path), Some("g96"), false, false, None).expect("g96");
    assert_eq!(mol.atoms.len(), 2);
    assert!((mol.atoms[1].position.x - 1.0).abs() < 1.0e-6);
    assert_eq!(
        mol.box_vectors,
        Some([[10.0, 1.0, 2.0], [3.0, 20.0, 4.0], [5.0, 6.0, 30.0]])
    );
    let _ = fs::remove_file(&g96_path);
}

#[test]
fn read_molecule_parses_pqr() {
    let pqr_path = temp_path("sample.pqr");
    write_text(
        &pqr_path,
        "CRYST1   10.000   20.000   30.000  90.00  90.00 120.00 P 1           1\n\
ATOM      1 O    WAT A   1    0.000    0.000    0.000  -0.83   1.52\n\
ATOM      2 H1   WAT A   1    0.957    0.000    0.000   0.42   1.20\n\
TER\n\
END\n",
    );
    let mol = read_molecule(Path::new(&pqr_path), Some("pqr"), false, false, None).expect("pqr");
    assert_eq!(mol.atoms.len(), 2);
    assert!((mol.atoms[0].charge + 0.83).abs() < 1.0e-6);
    assert_eq!(
        mol.atoms[1]
            .pdb_metadata
            .as_ref()
            .and_then(|metadata| metadata.pqr_radius),
        Some(1.2)
    );
    assert_eq!(mol.ter_after, vec![1]);
    let box_vectors = mol.box_vectors.expect("pqr box");
    assert!((box_vectors[0][0] - 10.0).abs() < 1.0e-5);
    assert!((box_vectors[1][0] + 10.0).abs() < 1.0e-4);
    assert!((box_vectors[1][1] - 17.320507).abs() < 1.0e-4);
    assert!((box_vectors[2][2] - 30.0).abs() < 1.0e-5);
    let _ = fs::remove_file(&pqr_path);
}

#[test]
fn write_output_uses_gro_default_scale() {
    let out = sample_output();
    let gro_path = temp_path("out_default_scale.gro");
    let gro_spec = OutputSpec {
        path: gro_path.to_string_lossy().to_string(),
        format: "gro".into(),
        scale: None,
    };
    write_output(&out, &gro_spec, false, 0.0, false, false).expect("gro write");
    let gro_text = fs::read_to_string(&gro_path).expect("gro read");
    let atom_line = gro_text.lines().nth(2).expect("first atom line");
    assert!(atom_line.contains("   0.100   0.200   0.300"));
    let box_line = gro_text.lines().last().expect("gro box line");
    assert_eq!(box_line.trim(), "1.00000 1.00000 1.00000");
    let _ = fs::remove_file(&gro_path);
}

#[test]
fn write_output_uses_g96_default_scale() {
    let out = sample_output();
    let g96_path = temp_path("out_default_scale.g96");
    let g96_spec = OutputSpec {
        path: g96_path.to_string_lossy().to_string(),
        format: "g96".into(),
        scale: None,
    };
    write_output(&out, &g96_spec, false, 0.0, false, false).expect("g96 write");
    let g96_text = fs::read_to_string(&g96_path).expect("g96 read");
    assert!(g96_text.contains("POSITION"));
    let first_position = g96_text
        .lines()
        .skip_while(|line| *line != "POSITION")
        .nth(1)
        .expect("g96 first position line");
    let position_fields: Vec<f32> = first_position
        .split_whitespace()
        .rev()
        .take(3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|value| value.parse::<f32>().expect("g96 position value"))
        .collect();
    assert!((position_fields[0] - 0.1).abs() < 1.0e-6);
    assert!((position_fields[1] - 0.2).abs() < 1.0e-6);
    assert!((position_fields[2] - 0.3).abs() < 1.0e-6);
    let box_line = g96_text
        .lines()
        .skip_while(|line| *line != "BOX")
        .nth(1)
        .expect("g96 box line");
    assert_eq!(
        box_line.split_whitespace().collect::<Vec<_>>(),
        vec!["1.000000000", "1.000000000", "1.000000000"]
    );
    let _ = fs::remove_file(&g96_path);
}

#[test]
fn write_output_emits_triclinic_gro_box_and_wraps_atom_ids() {
    let mut out = sample_output();
    out.box_vectors = Some([[10.0, 1.0, 2.0], [3.0, 20.0, 4.0], [5.0, 6.0, 30.0]]);
    out.atoms = (0..100_001)
        .map(|i| {
            let mut atom = out.atoms[i % 2].clone();
            atom.resid = 100_000 + i as i32;
            atom
        })
        .collect();
    let gro_path = temp_path("out_triclinic_wrap.gro");
    let gro_spec = OutputSpec {
        path: gro_path.to_string_lossy().to_string(),
        format: "gro".into(),
        scale: None,
    };
    write_output(&out, &gro_spec, false, 0.0, false, false).expect("gro write");
    let gro_text = fs::read_to_string(&gro_path).expect("gro read");
    let last_atom_line = gro_text.lines().nth(100_001 + 1).expect("last atom line");
    assert!(last_atom_line.starts_with("    0"));
    assert!(last_atom_line[15..20].trim() == "1");
    let box_line = gro_text.lines().last().expect("gro box line");
    let fields: Vec<&str> = box_line.split_whitespace().collect();
    assert_eq!(fields.len(), 9);
    assert_eq!(
        fields,
        vec![
            "1.00000", "2.00000", "3.00000", "0.10000", "0.20000", "0.30000", "0.40000", "0.50000",
            "0.60000"
        ]
    );
    let _ = fs::remove_file(&gro_path);
}

#[test]
fn write_output_emits_triclinic_g96_box() {
    let mut out = sample_output();
    out.box_vectors = Some([[10.0, 1.0, 2.0], [3.0, 20.0, 4.0], [5.0, 6.0, 30.0]]);
    let g96_path = temp_path("out_triclinic.g96");
    let g96_spec = OutputSpec {
        path: g96_path.to_string_lossy().to_string(),
        format: "g96".into(),
        scale: None,
    };
    write_output(&out, &g96_spec, false, 0.0, false, false).expect("g96 write");
    let g96_text = fs::read_to_string(&g96_path).expect("g96 read");
    let box_line = g96_text
        .lines()
        .skip_while(|line| *line != "BOX")
        .nth(1)
        .expect("g96 box line");
    let values = box_line
        .split_whitespace()
        .map(|value| value.parse::<f32>().expect("g96 box value"))
        .collect::<Vec<_>>();
    assert_eq!(values.len(), 9);
    let expected = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    for (value, expected) in values.iter().zip(expected) {
        assert!((*value - expected).abs() < 1.0e-6);
    }
    let _ = fs::remove_file(&g96_path);
}

#[test]
fn pdb_metadata_roundtrips() {
    let pdb_path = temp_path("metadata_input.pdb");
    let mut input = sample_output();
    input.atoms.truncate(1);
    input.atoms[0].name = "CA".into();
    input.atoms[0].element = "C".into();
    input.atoms[0].resname = "GLY".into();
    input.atoms[0].resid = 7;
    input.atoms[0].segid = "SEG1".into();
    input.atoms[0].pdb_metadata = Some(PdbAtomMetadata {
        occupancy: Some(0.55),
        temp_factor: Some(12.34),
        altloc: Some('A'),
        insertion_code: Some('B'),
        formal_charge: Some("1+".into()),
        pqr_radius: None,
    });
    let input_spec = OutputSpec {
        path: pdb_path.to_string_lossy().to_string(),
        format: "pdb".into(),
        scale: Some(1.0),
    };
    write_output(&input, &input_spec, false, 0.0, false, false).expect("pdb seed write");
    let mol = read_molecule(Path::new(&pdb_path), Some("pdb"), false, false, None).expect("pdb");
    let metadata = mol.atoms[0].pdb_metadata.as_ref().expect("metadata");
    assert_eq!(metadata.altloc, Some('A'));
    assert_eq!(metadata.insertion_code, Some('B'));
    assert_eq!(metadata.formal_charge.as_deref(), Some("1+"));
    assert_eq!(mol.atoms[0].segid, "SEG1");
    assert!((metadata.occupancy.unwrap_or_default() - 0.55).abs() < 1.0e-6);
    assert!((metadata.temp_factor.unwrap_or_default() - 12.34).abs() < 1.0e-6);

    let out_path = temp_path("metadata_output.pdb");
    let out = warp_pack::pack::PackOutput {
        atoms: mol.atoms,
        bonds: vec![],
        box_size: [0.0, 0.0, 0.0],
        ter_after: vec![],
        box_vectors: None,
    };
    let spec = OutputSpec {
        path: out_path.to_string_lossy().to_string(),
        format: "pdb".into(),
        scale: Some(1.0),
    };
    write_output(&out, &spec, false, 0.0, false, false).expect("pdb write");
    let out_text = fs::read_to_string(&out_path).expect("pdb reread");
    let atom_line = out_text
        .lines()
        .find(|line| line.starts_with("ATOM"))
        .unwrap();
    assert_eq!(&atom_line[16..17], "A");
    assert_eq!(&atom_line[26..27], "B");
    assert_eq!(&atom_line[54..60], "  0.55");
    assert_eq!(&atom_line[60..66], " 12.34");
    assert_eq!(&atom_line[72..76], "SEG1");
    assert_eq!(&atom_line[78..80], "1+");
    let _ = fs::remove_file(&pdb_path);
    let _ = fs::remove_file(&out_path);
}

#[test]
fn pqr_roundtrips_charge_radius_and_box() {
    let mut out = sample_output();
    out.box_vectors = Some([[10.0, 0.0, 0.0], [-10.0, 17.320507, 0.0], [0.0, 0.0, 30.0]]);
    out.atoms[0].name = "OW".into();
    out.atoms[0].resname = "WAT".into();
    out.atoms[0].charge = -0.83;
    out.atoms[0].pdb_metadata = Some(PdbAtomMetadata {
        occupancy: None,
        temp_factor: None,
        altloc: None,
        insertion_code: None,
        formal_charge: None,
        pqr_radius: Some(1.52),
    });
    out.atoms[1].name = "HW1".into();
    out.atoms[1].resname = "WAT".into();
    out.atoms[1].charge = 0.42;
    out.atoms[1].pdb_metadata = Some(PdbAtomMetadata {
        occupancy: None,
        temp_factor: None,
        altloc: None,
        insertion_code: None,
        formal_charge: None,
        pqr_radius: Some(1.2),
    });
    let pqr_path = temp_path("roundtrip.pqr");
    let pqr_spec = OutputSpec {
        path: pqr_path.to_string_lossy().to_string(),
        format: "pqr".into(),
        scale: Some(1.0),
    };
    write_output(&out, &pqr_spec, true, 0.0, false, false).expect("pqr write");
    let pqr_text = fs::read_to_string(&pqr_path).expect("pqr read");
    assert!(pqr_text.contains("CRYST1"));
    assert!(pqr_text.contains("-0.83"));
    assert!(pqr_text.contains("  1.52"));
    let mol = read_molecule(Path::new(&pqr_path), Some("pqr"), false, false, None).expect("pqr");
    assert_eq!(mol.atoms.len(), 2);
    assert!((mol.atoms[0].charge + 0.83).abs() < 1.0e-6);
    assert_eq!(
        mol.atoms[0]
            .pdb_metadata
            .as_ref()
            .and_then(|metadata| metadata.pqr_radius),
        Some(1.52)
    );
    assert_eq!(mol.ter_after, vec![1]);
    assert!(mol.box_vectors.is_some());
    let _ = fs::remove_file(&pqr_path);
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
    let atom_line = pdb_text
        .lines()
        .find(|line| line.starts_with("ATOM") || line.starts_with("HETATM"))
        .expect("strict pdb atom line");
    assert!(atom_line.len() >= 78);
    let _ = fs::remove_file(&pdb_path);

    let pdb_strict_path = temp_path("out_strict.pdb");
    let pdb_strict_spec = OutputSpec {
        path: pdb_strict_path.to_string_lossy().to_string(),
        format: "pdb-strict".into(),
        scale: Some(1.0),
    };
    write_output(&out, &pdb_strict_spec, true, 0.0, true, false).expect("pdb strict write");
    let pdb_strict_text = fs::read_to_string(&pdb_strict_path).expect("pdb strict read");
    assert!(pdb_strict_text.contains("CONECT"));
    let _ = fs::remove_file(&pdb_strict_path);

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

    let g96_path = temp_path("out.g96");
    let g96_spec = OutputSpec {
        path: g96_path.to_string_lossy().to_string(),
        format: "g96".into(),
        scale: Some(1.0),
    };
    write_output(&out, &g96_spec, false, 0.0, false, false).expect("g96 write");
    let g96_text = fs::read_to_string(&g96_path).expect("g96 read");
    assert!(g96_text.contains("TITLE"));
    assert!(g96_text.contains("POSITION"));
    assert!(g96_text.contains("BOX"));
    let _ = fs::remove_file(&g96_path);

    let pqr_path = temp_path("out.pqr");
    let mut pqr_out = out.clone();
    pqr_out.atoms[0].pdb_metadata = Some(PdbAtomMetadata {
        occupancy: None,
        temp_factor: None,
        altloc: None,
        insertion_code: None,
        formal_charge: None,
        pqr_radius: Some(1.52),
    });
    let pqr_spec = OutputSpec {
        path: pqr_path.to_string_lossy().to_string(),
        format: "pqr".into(),
        scale: Some(1.0),
    };
    write_output(&pqr_out, &pqr_spec, true, 0.0, false, false).expect("pqr write");
    let pqr_text = fs::read_to_string(&pqr_path).expect("pqr read");
    assert!(pqr_text.contains("CRYST1"));
    assert!(pqr_text.contains("ATOM"));
    assert!(pqr_text.contains("  1.52"));
    let _ = fs::remove_file(&pqr_path);

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
    assert!(mol2_text.contains("2 1 0 0 0"));
    assert!(mol2_text.contains("@<TRIPOS>BOND"));
    assert!(mol2_text.contains("\n     1      1      2 1\n"));
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

#[test]
fn pdb_falls_back_to_mmcif_when_residue_id_exceeds_fixed_width() {
    let mut out = sample_output();
    out.atoms[0].resid = 10_000;
    out.atoms[0].pdb_metadata = Some(PdbAtomMetadata::default());
    let pdb_path = temp_path("out_invalid_resid.pdb");
    let pdb_spec = OutputSpec {
        path: pdb_path.to_string_lossy().to_string(),
        format: "pdb".into(),
        scale: Some(1.0),
    };
    let written = write_output(&out, &pdb_spec, false, 0.0, false, false)
        .expect("pdb should fall back to mmcif");
    assert!(written.fallback_applied);
    assert_eq!(written.format, "mmcif");
    assert!(written.path.ends_with(".cif"));
    let cif_text = fs::read_to_string(&written.path).expect("fallback mmcif read");
    assert!(cif_text.contains("data_warp_pack"));
    let _ = fs::remove_file(&written.path);
}

#[test]
fn pdb_strict_falls_back_to_mmcif_when_needed() {
    let mut out = sample_output();
    out.atoms[0].resname = "pes_8mer_029".into();
    let pdb_path = temp_path("out_strict_invalid_resname.pdb");
    let pdb_spec = OutputSpec {
        path: pdb_path.to_string_lossy().to_string(),
        format: "pdb-strict".into(),
        scale: Some(1.0),
    };
    let written = write_output(&out, &pdb_spec, false, 0.0, false, false)
        .expect("strict pdb should fall back to mmcif");
    assert!(written.fallback_applied);
    assert_eq!(written.format, "mmcif");
    assert!(written.path.ends_with(".cif"));
    let cif_text = fs::read_to_string(&written.path).expect("fallback mmcif read");
    assert!(cif_text.contains("data_warp_pack"));
    let _ = fs::remove_file(&written.path);
}

#[test]
fn write_output_infers_format_from_path_when_missing() {
    let out = sample_output();
    let cif_path = temp_path("out_inferred.cif");
    let cif_spec = OutputSpec {
        path: cif_path.to_string_lossy().to_string(),
        format: String::new(),
        scale: Some(1.0),
    };
    let written =
        write_output(&out, &cif_spec, false, 0.0, false, false).expect("inferred cif write");
    assert_eq!(written.format, "cif");
    let cif_text = fs::read_to_string(&cif_path).expect("cif read");
    assert!(cif_text.contains("data_warp_pack"));
    assert!(cif_text.contains("_cell.length_a 10.000"));
    let _ = fs::remove_file(&cif_path);

    let ent_path = temp_path("out_inferred.ent");
    let ent_spec = OutputSpec {
        path: ent_path.to_string_lossy().to_string(),
        format: String::new(),
        scale: Some(1.0),
    };
    let written =
        write_output(&out, &ent_spec, false, 0.0, false, false).expect("inferred ent write");
    assert_eq!(written.format, "ent");
    let ent_text = fs::read_to_string(&ent_path).expect("ent read");
    assert!(ent_text.contains("ATOM"));
    let _ = fs::remove_file(&ent_path);
}

#[test]
fn cif_output_writes_cell_parameters() {
    let mut out = sample_output();
    out.box_size = [300.0, 300.0, 300.0];
    let cif_path = temp_path("out_cell.cif");
    let cif_spec = OutputSpec {
        path: cif_path.to_string_lossy().to_string(),
        format: "cif".into(),
        scale: Some(1.0),
    };

    write_output(&out, &cif_spec, true, 0.0, false, false).expect("cif write");

    let cif_text = fs::read_to_string(&cif_path).expect("cif read");
    assert!(cif_text.contains("_cell.length_a 300.000"));
    assert!(cif_text.contains("_cell.length_b 300.000"));
    assert!(cif_text.contains("_cell.length_c 300.000"));
    assert!(cif_text.contains("_cell.angle_alpha 90.00"));
    assert!(cif_text.contains("_cell.angle_beta 90.00"));
    assert!(cif_text.contains("_cell.angle_gamma 90.00"));
    let _ = fs::remove_file(&cif_path);
}
