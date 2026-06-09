use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::{json, Value};
use warp_structure::io::{read_prmtop_topology, write_minimal_prmtop, AmberTopology};

fn temp_path(label: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!(
        "warp_build_test_{}_{}_{}",
        label,
        std::process::id(),
        nanos
    ));
    path
}

fn write_text(path: &Path, text: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, text).expect("write file");
}

fn write_training_oligomer(path: &Path) {
    write_text(
        path,
        "ATOM      1  C1  HDA A   1       0.000   0.000   0.000  1.00  0.00           C\n\
ATOM      2  C2  RPT A   2       3.000   0.000   0.000  1.00  0.00           C\n\
ATOM      3  C3  TLA A   3       6.000   0.000   0.000  1.00  0.00           C\n\
END\n",
    );
}

fn write_prmtop(path: &Path) {
    let topology = AmberTopology {
        atom_names: vec!["C1".into(), "C2".into(), "C3".into()],
        residue_labels: vec!["HDA".into(), "RPT".into(), "TLA".into()],
        residue_pointers: vec![1, 2, 3],
        atomic_numbers: vec![6, 6, 6],
        masses: vec![12.01, 12.01, 12.01],
        charges: vec![0.5, 1.0, 0.5],
        atom_type_indices: vec![1, 1, 1],
        amber_atom_types: vec!["CT".into(), "CT".into(), "CT".into()],
        radii: vec![1.7, 1.7, 1.7],
        screen: vec![0.8, 0.8, 0.8],
        bonds: vec![(0, 1), (1, 2)],
        bond_type_indices: vec![1, 1],
        bond_force_constants: vec![310.0],
        bond_equil_values: vec![1.53],
        angles: vec![[0, 1, 2]],
        angle_type_indices: vec![1],
        angle_force_constants: vec![55.0],
        angle_equil_values: vec![109.5],
        dihedrals: Vec::new(),
        dihedral_type_indices: Vec::new(),
        dihedral_force_constants: Vec::new(),
        dihedral_periodicities: Vec::new(),
        dihedral_phases: Vec::new(),
        scee_scale_factors: Vec::new(),
        scnb_scale_factors: Vec::new(),
        solty: vec![0.0],
        impropers: Vec::new(),
        improper_type_indices: Vec::new(),
        excluded_atoms: vec![vec![2], vec![1, 3], vec![2]],
        nonbonded_parm_index: vec![1],
        lennard_jones_acoef: vec![1.0],
        lennard_jones_bcoef: vec![0.5],
        lennard_jones_14_acoef: vec![0.8],
        lennard_jones_14_bcoef: vec![0.4],
        hbond_acoef: vec![0.0],
        hbond_bcoef: vec![0.0],
        hbcut: vec![0.0],
        tree_chain_classification: vec!["M".into(), "M".into(), "E".into()],
        join_array: vec![0, 0, 0],
        irotat: vec![0, 0, 0],
        solvent_pointers: Vec::new(),
        atoms_per_molecule: Vec::new(),
        box_dimensions: Vec::new(),
        radius_set: Some("modified Bondi radii".into()),
        ipol: 0,
    };
    write_minimal_prmtop(path.to_string_lossy().as_ref(), &topology).expect("write prmtop");
}

fn write_json(path: &Path, value: &Value) {
    write_text(
        path,
        &(serde_json::to_string_pretty(value).expect("json") + "\n"),
    );
}

fn make_bundle_dir(label: &str) -> (PathBuf, PathBuf) {
    let dir = temp_path(label);
    fs::create_dir_all(&dir).expect("create dir");
    let training = dir.join("training.pdb");
    let prmtop = dir.join("training.prmtop");
    let charges = dir.join("training_charge.json");
    let forcefield = dir.join("training.ffxml");
    let bundle = dir.join("bundle.json");
    write_training_oligomer(&training);
    write_prmtop(&prmtop);
    write_text(&forcefield, "<ForceField/>\n");
    write_json(
        &charges,
        &json!({
            "version": "warp-build.charge-manifest.v1",
            "head_charge_e": 0.5,
            "repeat_charge_e": 1.0,
            "tail_charge_e": 0.5,
        }),
    );
    write_json(
        &bundle,
        &json!({
            "schema_version": "polymer-param-source.bundle.v1",
            "bundle_id": "pmma_param_bundle_v1",
            "training_context": {
                "mode": "oligomer_training",
                "training_oligomer_n": 3,
                "notes": "test"
            },
            "provenance": {},
            "unit_library": {
                "H": {
                    "display_name": "head_cap",
                    "junctions": {"head": "pmma_head_cap", "tail": "pmma_head_cap"},
                    "template_resname": "HDA"
                },
                "A": {
                    "display_name": "repeat",
                    "junctions": {"head": "pmma_head", "tail": "pmma_tail"},
                    "template_resname": "RPT"
                },
                "B": {
                    "display_name": "alternate",
                    "junctions": {"head": "pmma_head", "tail": "pmma_tail"},
                    "template_resname": "RPT"
                },
                "T": {
                    "display_name": "tail_cap",
                    "junctions": {"head": "pmma_tail_cap", "tail": "pmma_tail_cap"},
                    "template_resname": "TLA"
                }
            },
            "motif_library": {
                "M2": {
                    "display_name": "dimer_motif",
                    "root_node_id": "m1",
                    "nodes": [
                        {"id": "m1", "token": "A"},
                        {"id": "m2", "token": "B"}
                    ],
                    "edges": [
                        {"from": "m1", "to": "m2", "from_junction": "tail", "to_junction": "head", "bond_order": 1}
                    ],
                    "exposed_ports": {
                        "head": {"node_id": "m1", "junction": "head"},
                        "tail": {"node_id": "m2", "junction": "tail"}
                    }
                }
            },
            "junction_library": {
                "pmma_head_cap": {
                    "attach_atom": {"scope": "unit", "selector": "name C1"},
                    "leaving_atoms": [],
                    "bond_order": 1,
                    "anchor_atoms": [{"scope": "unit", "selector": "name C1"}]
                },
                "pmma_head": {
                    "attach_atom": {"scope": "unit", "selector": "name C2"},
                    "leaving_atoms": [],
                    "bond_order": 1,
                    "anchor_atoms": [{"scope": "unit", "selector": "name C2"}]
                },
                "pmma_tail": {
                    "attach_atom": {"scope": "unit", "selector": "name C2"},
                    "leaving_atoms": [],
                    "bond_order": 1,
                    "anchor_atoms": [{"scope": "unit", "selector": "name C2"}]
                },
                "pmma_tail_cap": {
                    "attach_atom": {"scope": "unit", "selector": "name C3"},
                    "leaving_atoms": [],
                    "bond_order": 1,
                    "anchor_atoms": [{"scope": "unit", "selector": "name C3"}]
                }
            },
            "capabilities": {
                "supported_target_modes": ["linear_homopolymer", "linear_sequence_polymer", "block_copolymer", "random_copolymer", "star_polymer", "branched_polymer", "polymer_graph"],
                "supported_conformation_modes": ["extended", "random_walk"],
                "supported_tacticity_modes": ["inherit", "isotactic", "syndiotactic", "atactic"],
                "supported_termini_policies": ["default", "source_default"],
                "sequence_token_support": {
                    "tokens": ["H", "A", "B", "T", "M2"],
                    "allowed_adjacencies": [["H", "A"], ["H", "B"], ["A", "A"], ["A", "B"], ["B", "A"], ["B", "B"], ["A", "T"], ["B", "T"], ["H", "T"]]
                },
                "charge_transfer_supported": true
            },
            "artifacts": {
                "source_coordinates": "training.pdb",
                "source_topology_ref": "training.prmtop",
                "forcefield_ref": "training.ffxml",
                "source_charge_manifest": "training_charge.json"
            },
            "charge_model": {}
        }),
    );
    (dir, bundle)
}

fn make_structure_only_bundle_dir(label: &str) -> (PathBuf, PathBuf) {
    let (dir, bundle) = make_bundle_dir(label);
    let mut payload: Value =
        serde_json::from_str(&fs::read_to_string(&bundle).expect("read bundle"))
            .expect("bundle json");
    let artifacts = payload["artifacts"]
        .as_object_mut()
        .expect("bundle artifacts object");
    artifacts.remove("source_topology_ref");
    artifacts.remove("source_charge_manifest");
    write_json(&bundle, &payload);
    fs::remove_file(dir.join("training.prmtop")).ok();
    fs::remove_file(dir.join("training_charge.json")).ok();
    (dir, bundle)
}

fn make_multiport_star_bundle_dir(label: &str) -> (PathBuf, PathBuf) {
    let dir = temp_path(label);
    fs::create_dir_all(&dir).expect("create dir");
    let training = dir.join("training.pdb");
    let bundle = dir.join("bundle.json");
    write_text(
        &training,
        "ATOM      1  C1  COR A   1       0.000   0.000   0.000  1.00  0.00           C\n\
ATOM      2  C2  COR A   1       1.530   0.000   0.000  1.00  0.00           C\n\
ATOM      3  C3  COR A   1       0.000   1.530   0.000  1.00  0.00           C\n\
ATOM      4  C4  COR A   1       0.000   0.000   1.530  1.00  0.00           C\n\
ATOM      5  A1  ARM A   2       4.500   0.000   0.000  1.00  0.00           C\n\
ATOM      6  D1  DMY A   3       8.000   0.000   0.000  1.00  0.00           C\n\
END\n",
    );
    write_json(
        &bundle,
        &json!({
            "schema_version": "polymer-param-source.bundle.v1",
            "bundle_id": "multiport_star_bundle_v1",
            "training_context": {
                "mode": "oligomer_training",
                "training_oligomer_n": 3,
                "notes": "four-port core test fixture"
            },
            "provenance": {},
            "unit_library": {
                "C": {
                    "display_name": "four_port_core",
                    "junctions": {
                        "j1": "core_j1",
                        "j2": "core_j2",
                        "j3": "core_j3",
                        "j4": "core_j4"
                    },
                    "template_resname": "COR"
                },
                "A": {
                    "display_name": "arm_repeat",
                    "junctions": {"head": "arm_head", "tail": "arm_tail"},
                    "template_resname": "ARM"
                }
            },
            "motif_library": {},
            "junction_library": {
                "core_j1": {
                    "attach_atom": {"scope": "unit", "selector": "name C1"},
                    "leaving_atoms": [],
                    "bond_order": 1,
                    "anchor_atoms": [{"scope": "unit", "selector": "name C1"}]
                },
                "core_j2": {
                    "attach_atom": {"scope": "unit", "selector": "name C2"},
                    "leaving_atoms": [],
                    "bond_order": 1,
                    "anchor_atoms": [{"scope": "unit", "selector": "name C2"}]
                },
                "core_j3": {
                    "attach_atom": {"scope": "unit", "selector": "name C3"},
                    "leaving_atoms": [],
                    "bond_order": 1,
                    "anchor_atoms": [{"scope": "unit", "selector": "name C3"}]
                },
                "core_j4": {
                    "attach_atom": {"scope": "unit", "selector": "name C4"},
                    "leaving_atoms": [],
                    "bond_order": 1,
                    "anchor_atoms": [{"scope": "unit", "selector": "name C4"}]
                },
                "arm_head": {
                    "attach_atom": {"scope": "unit", "selector": "name A1"},
                    "leaving_atoms": [],
                    "bond_order": 1,
                    "anchor_atoms": [{"scope": "unit", "selector": "name A1"}]
                },
                "arm_tail": {
                    "attach_atom": {"scope": "unit", "selector": "name A1"},
                    "leaving_atoms": [],
                    "bond_order": 1,
                    "anchor_atoms": [{"scope": "unit", "selector": "name A1"}]
                }
            },
            "capabilities": {
                "supported_target_modes": ["star_polymer", "polymer_graph"],
                "supported_conformation_modes": ["aligned", "random_walk"],
                "supported_tacticity_modes": ["inherit"],
                "supported_termini_policies": ["default", "source_default"],
                "sequence_token_support": {
                    "tokens": ["C", "A"],
                    "allowed_adjacencies": [["C", "A"], ["A", "A"]]
                },
                "charge_transfer_supported": false
            },
            "artifacts": {
                "source_coordinates": "training.pdb"
            },
            "charge_model": {}
        }),
    );
    (dir, bundle)
}

fn write_bad_valence_training_oligomer(path: &Path) {
    write_text(
        path,
        "ATOM      1  C1  HDA A   1       0.000   0.000   0.000  1.00  0.00           C\n\
ATOM      2 H11  HDA A   1       0.000   1.090   0.000  1.00  0.00           H\n\
ATOM      3 H12  HDA A   1       0.000  -1.090   0.000  1.00  0.00           H\n\
ATOM      4 H13  HDA A   1       0.000   0.000   1.090  1.00  0.00           H\n\
ATOM      5 H14  HDA A   1       0.000   0.000  -1.090  1.00  0.00           H\n\
ATOM      6 CL1  HDA A   1      -1.800   0.000   0.000  1.00  0.00          CL\n\
ATOM      7  C2  RPT A   2       4.000   0.000   0.000  1.00  0.00           C\n\
ATOM      8 H21  RPT A   2       4.000   1.090   0.000  1.00  0.00           H\n\
ATOM      9 H22  RPT A   2       4.000  -1.090   0.000  1.00  0.00           H\n\
ATOM     10 H23  RPT A   2       4.000   0.000   1.090  1.00  0.00           H\n\
ATOM     11 H24  RPT A   2       4.000   0.000  -1.090  1.00  0.00           H\n\
ATOM     12 CL2  RPT A   2       2.200   0.000   0.000  1.00  0.00          CL\n\
ATOM     13  C3  TLA A   3       8.000   0.000   0.000  1.00  0.00           C\n\
ATOM     14 H31  TLA A   3       8.000   1.090   0.000  1.00  0.00           H\n\
ATOM     15 H32  TLA A   3       8.000  -1.090   0.000  1.00  0.00           H\n\
ATOM     16 H33  TLA A   3       8.000   0.000   1.090  1.00  0.00           H\n\
ATOM     17 H34  TLA A   3       8.000   0.000  -1.090  1.00  0.00           H\n\
ATOM     18 CL3  TLA A   3       6.200   0.000   0.000  1.00  0.00          CL\n\
END\n",
    );
}

fn write_bad_valence_prmtop(path: &Path) {
    let atom_names = vec![
        "C1", "H11", "H12", "H13", "H14", "CL1", "C2", "H21", "H22", "H23", "H24", "CL2", "C3",
        "H31", "H32", "H33", "H34", "CL3",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    let atomic_numbers = vec![6, 1, 1, 1, 1, 17, 6, 1, 1, 1, 1, 17, 6, 1, 1, 1, 1, 17];
    let masses = atomic_numbers
        .iter()
        .map(|z| match *z {
            6 => 12.01,
            17 => 35.45,
            _ => 1.008,
        })
        .collect::<Vec<_>>();
    let bonds = vec![
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (6, 7),
        (6, 8),
        (6, 9),
        (6, 10),
        (6, 11),
        (12, 13),
        (12, 14),
        (12, 15),
        (12, 16),
        (12, 17),
    ];
    let topology = AmberTopology {
        atom_names,
        residue_labels: vec!["HDA".into(), "RPT".into(), "TLA".into()],
        residue_pointers: vec![1, 7, 13],
        atomic_numbers,
        masses,
        charges: vec![0.0; 18],
        atom_type_indices: vec![1; 18],
        amber_atom_types: vec!["DU".into(); 18],
        radii: vec![1.5; 18],
        screen: vec![0.8; 18],
        bonds,
        bond_type_indices: vec![1; 15],
        bond_force_constants: vec![310.0],
        bond_equil_values: vec![1.09],
        angles: Vec::new(),
        angle_type_indices: Vec::new(),
        angle_force_constants: Vec::new(),
        angle_equil_values: Vec::new(),
        dihedrals: Vec::new(),
        dihedral_type_indices: Vec::new(),
        dihedral_force_constants: Vec::new(),
        dihedral_periodicities: Vec::new(),
        dihedral_phases: Vec::new(),
        scee_scale_factors: Vec::new(),
        scnb_scale_factors: Vec::new(),
        solty: vec![0.0],
        impropers: Vec::new(),
        improper_type_indices: Vec::new(),
        excluded_atoms: vec![Vec::new(); 18],
        nonbonded_parm_index: vec![1],
        lennard_jones_acoef: vec![1.0],
        lennard_jones_bcoef: vec![0.5],
        lennard_jones_14_acoef: vec![0.8],
        lennard_jones_14_bcoef: vec![0.4],
        hbond_acoef: vec![0.0],
        hbond_bcoef: vec![0.0],
        hbcut: vec![0.0],
        tree_chain_classification: vec!["M".into(); 18],
        join_array: vec![0; 18],
        irotat: vec![0; 18],
        solvent_pointers: Vec::new(),
        atoms_per_molecule: vec![18],
        box_dimensions: Vec::new(),
        radius_set: Some("modified Bondi radii".into()),
        ipol: 0,
    };
    write_minimal_prmtop(path.to_string_lossy().as_ref(), &topology).expect("write bad prmtop");
}

fn write_generic_forcefield_ffxml(path: &Path) {
    write_text(
        path,
        "<ForceField>\n\
  <AtomTypes>\n\
    <Type name=\"CT\" class=\"C\" element=\"C\" mass=\"12.011\"/>\n\
    <Type name=\"HC\" class=\"H\" element=\"H\" mass=\"1.008\"/>\n\
    <Type name=\"CL\" class=\"Cl\" element=\"Cl\" mass=\"35.45\"/>\n\
  </AtomTypes>\n\
  <Residues>\n\
    <Residue name=\"HDA\">\n\
      <Atom name=\"C1\" type=\"CT\" charge=\"0.25\"/>\n\
      <Atom name=\"H11\" type=\"HC\" charge=\"0.05\"/>\n\
      <Atom name=\"H12\" type=\"HC\" charge=\"0.05\"/>\n\
      <Atom name=\"H13\" type=\"HC\" charge=\"0.05\"/>\n\
      <Atom name=\"H14\" type=\"HC\" charge=\"0.05\"/>\n\
      <Atom name=\"CL1\" type=\"CL\" charge=\"-0.45\"/>\n\
      <AllowPatch name=\"GENERIC_POLYMER_PATCH\"/>\n\
    </Residue>\n\
    <Residue name=\"RPT\">\n\
      <Atom name=\"C2\" type=\"CT\" charge=\"0.25\"/>\n\
      <Atom name=\"H21\" type=\"HC\" charge=\"0.05\"/>\n\
      <Atom name=\"H22\" type=\"HC\" charge=\"0.05\"/>\n\
      <Atom name=\"H23\" type=\"HC\" charge=\"0.05\"/>\n\
      <Atom name=\"H24\" type=\"HC\" charge=\"0.05\"/>\n\
      <Atom name=\"CL2\" type=\"CL\" charge=\"-0.45\"/>\n\
      <AllowPatch name=\"GENERIC_POLYMER_PATCH\"/>\n\
    </Residue>\n\
    <Residue name=\"TLA\">\n\
      <Atom name=\"C3\" type=\"CT\" charge=\"0.25\"/>\n\
      <Atom name=\"H31\" type=\"HC\" charge=\"0.05\"/>\n\
      <Atom name=\"H32\" type=\"HC\" charge=\"0.05\"/>\n\
      <Atom name=\"H33\" type=\"HC\" charge=\"0.05\"/>\n\
      <Atom name=\"H34\" type=\"HC\" charge=\"0.05\"/>\n\
      <Atom name=\"CL3\" type=\"CL\" charge=\"-0.45\"/>\n\
      <AllowPatch name=\"GENERIC_POLYMER_PATCH\"/>\n\
    </Residue>\n\
  </Residues>\n\
  <Patches>\n\
    <Patch name=\"GENERIC_POLYMER_PATCH\"/>\n\
  </Patches>\n\
  <HarmonicBondForce>\n\
    <Bond length=\"0.109\" k=\"265000.0\"/>\n\
  </HarmonicBondForce>\n\
  <HarmonicAngleForce>\n\
    <Angle angle=\"1.9111355\" k=\"520.0\"/>\n\
  </HarmonicAngleForce>\n\
  <PeriodicTorsionForce>\n\
    <Proper periodicity1=\"3\" phase1=\"0.0\" k1=\"1.0\"/>\n\
    <Improper periodicity1=\"2\" phase1=\"3.14159265\" k1=\"1.0\"/>\n\
  </PeriodicTorsionForce>\n\
  <NonbondedForce coulomb14scale=\"0.8333333333\" lj14scale=\"0.5\">\n\
    <UseAttributeFromResidue name=\"charge\"/>\n\
    <Atom type=\"CT\" sigma=\"0.339967\" epsilon=\"0.4577296\"/>\n\
    <Atom type=\"HC\" sigma=\"0.247135\" epsilon=\"0.0656888\"/>\n\
    <Atom type=\"CL\" sigma=\"0.399000\" epsilon=\"1.108800\"/>\n\
  </NonbondedForce>\n\
</ForceField>\n",
    );
}

fn make_bad_training_bundle_dir(
    label: &str,
    include_prmtop: bool,
    include_ffxml: bool,
) -> (PathBuf, PathBuf) {
    let dir = temp_path(label);
    fs::create_dir_all(&dir).expect("create bad training dir");
    let training = dir.join("training.pdb");
    let prmtop = dir.join("training.prmtop");
    let forcefield = dir.join("training.ffxml");
    let bundle = dir.join("bundle.json");
    write_bad_valence_training_oligomer(&training);
    if include_prmtop {
        write_bad_valence_prmtop(&prmtop);
    }
    if include_ffxml {
        write_generic_forcefield_ffxml(&forcefield);
    }
    write_json(
        &bundle,
        &json!({
            "schema_version": "polymer-param-source.bundle.v1",
            "bundle_id": "pmma_param_bundle_v1",
            "training_context": {
                "mode": "oligomer_training",
                "training_oligomer_n": 3,
                "notes": "bad training"
            },
            "provenance": {},
            "unit_library": {
                "H": {
                    "display_name": "head_cap",
                    "junctions": {"head": "pmma_head_cap", "tail": "pmma_head_cap"},
                    "template_resname": "HDA"
                },
                "A": {
                    "display_name": "repeat",
                    "junctions": {"head": "pmma_head", "tail": "pmma_tail"},
                    "template_resname": "RPT"
                },
                "T": {
                    "display_name": "tail_cap",
                    "junctions": {"head": "pmma_tail_cap", "tail": "pmma_tail_cap"},
                    "template_resname": "TLA"
                }
            },
            "motif_library": {},
            "junction_library": {
                "pmma_head_cap": {
                    "attach_atom": {"scope": "unit", "selector": "name C1"},
                    "leaving_atoms": [],
                    "bond_order": 1,
                    "anchor_atoms": [{"scope": "unit", "selector": "name C1"}]
                },
                "pmma_head": {
                    "attach_atom": {"scope": "unit", "selector": "name C2"},
                    "leaving_atoms": [],
                    "bond_order": 1,
                    "anchor_atoms": [{"scope": "unit", "selector": "name C2"}]
                },
                "pmma_tail": {
                    "attach_atom": {"scope": "unit", "selector": "name C2"},
                    "leaving_atoms": [],
                    "bond_order": 1,
                    "anchor_atoms": [{"scope": "unit", "selector": "name C2"}]
                },
                "pmma_tail_cap": {
                    "attach_atom": {"scope": "unit", "selector": "name C3"},
                    "leaving_atoms": [],
                    "bond_order": 1,
                    "anchor_atoms": [{"scope": "unit", "selector": "name C3"}]
                }
            },
            "capabilities": {
                "supported_target_modes": ["linear_homopolymer", "linear_sequence_polymer"],
                "supported_conformation_modes": ["extended", "random_walk"],
                "supported_tacticity_modes": ["inherit", "isotactic", "syndiotactic", "atactic"],
                "supported_termini_policies": ["default", "source_default"],
                "sequence_token_support": {
                    "tokens": ["H", "A", "T"],
                    "allowed_adjacencies": [["H", "A"], ["A", "A"], ["A", "T"]]
                },
                "charge_transfer_supported": include_prmtop
            },
            "artifacts": {
                "source_coordinates": "training.pdb",
                "source_topology_ref": if include_prmtop { Value::String("training.prmtop".into()) } else { Value::Null },
                "forcefield_ref": if include_ffxml { Value::String("training.ffxml".into()) } else { Value::Null }
            },
            "charge_model": {}
        }),
    );
    (dir, bundle)
}

fn copy_fixture_dir(name: &str, label: &str) -> (PathBuf, PathBuf) {
    let dir = temp_path(label);
    fs::create_dir_all(&dir).expect("create fixture dir");
    let fixture_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name);
    for entry in fs::read_dir(&fixture_root).expect("read fixture dir") {
        let entry = entry.expect("fixture entry");
        let target = dir.join(entry.file_name());
        fs::copy(entry.path(), &target).expect("copy fixture");
    }
    let bundle = dir.join("bundle.json");
    let payload: Value =
        serde_json::from_str(&fs::read_to_string(&bundle).expect("read fixture bundle"))
            .expect("fixture bundle json");
    if let Some(path) = payload
        .pointer("/artifacts/forcefield_ref")
        .and_then(Value::as_str)
    {
        let forcefield = dir.join(path);
        if !forcefield.exists() {
            write_text(&forcefield, "<ForceField/>\n");
        }
    }
    (dir.clone(), dir.join("bundle.json"))
}

fn max_inter_residue_bond_distance(coords: &Path, topology_graph: &Path) -> f64 {
    let atoms = fs::read_to_string(coords)
        .expect("read coords")
        .lines()
        .filter(|line| line.starts_with("ATOM  ") || line.starts_with("HETATM"))
        .map(|line| {
            (
                line[12..16].trim().to_string(),
                [
                    line[30..38].trim().parse::<f64>().expect("x"),
                    line[38..46].trim().parse::<f64>().expect("y"),
                    line[46..54].trim().parse::<f64>().expect("z"),
                ],
            )
        })
        .collect::<Vec<_>>();
    let graph: Value =
        serde_json::from_str(&fs::read_to_string(topology_graph).expect("read graph"))
            .expect("parse graph");
    graph["inter_residue_bonds"]
        .as_array()
        .expect("inter_residue_bonds")
        .iter()
        .map(|bond| {
            let a = bond["a"].as_u64().expect("bond a") as usize - 1;
            let b = bond["b"].as_u64().expect("bond b") as usize - 1;
            let pa = atoms[a].1;
            let pb = atoms[b].1;
            let dx = pa[0] - pb[0];
            let dy = pa[1] - pb[1];
            let dz = pa[2] - pb[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .fold(0.0, f64::max)
}

fn pdb_atom_positions(coords: &Path) -> Vec<[f64; 3]> {
    fs::read_to_string(coords)
        .expect("read coords")
        .lines()
        .filter(|line| line.starts_with("ATOM  ") || line.starts_with("HETATM"))
        .map(|line| {
            [
                line[30..38].trim().parse::<f64>().expect("x"),
                line[38..46].trim().parse::<f64>().expect("y"),
                line[46..54].trim().parse::<f64>().expect("z"),
            ]
        })
        .collect()
}

fn pdb_record_count(coords: &Path, prefix: &str) -> usize {
    fs::read_to_string(coords)
        .expect("read coords")
        .lines()
        .filter(|line| line.starts_with(prefix))
        .count()
}

#[test]
fn schema_and_inspect_source_work() {
    let schema = warp_build::schema_json("request").expect("schema");
    let parsed: Value = serde_json::from_str(&schema).expect("parse schema");
    assert!(parsed["properties"].get("schema_version").is_some());
    let source_schema = warp_build::schema_json("source_bundle").expect("source schema");
    let source_schema_value: Value =
        serde_json::from_str(&source_schema).expect("parse source schema");
    assert!(source_schema_value["properties"]
        .get("schema_version")
        .is_some());
    let build_manifest_schema =
        warp_build::schema_json("build_manifest").expect("build manifest schema");
    assert!(build_manifest_schema.contains("overlap_status"));
    let build_manifest_schema_value: Value =
        serde_json::from_str(&build_manifest_schema).expect("parse build manifest schema");
    assert!(build_manifest_schema_value["properties"]
        .get("schema_version")
        .is_some());
    let charge_manifest_schema =
        warp_build::schema_json("charge_manifest").expect("charge manifest schema");
    let charge_manifest_schema_value: Value =
        serde_json::from_str(&charge_manifest_schema).expect("parse charge manifest schema");
    assert!(charge_manifest_schema_value["properties"]
        .get("schema_version")
        .is_some());
    let graph_schema = warp_build::schema_json("topology_graph").expect("graph schema");
    assert!(graph_schema.contains("final_overlap_pairs"));
    let graph_schema_value: Value =
        serde_json::from_str(&graph_schema).expect("parse graph schema");
    assert!(graph_schema_value["properties"].get("build_plan").is_some());
    assert_eq!(
        warp_build::example_request("random_walk")["schema_version"],
        "warp-build.agent.v1"
    );

    let (_dir, bundle) = make_bundle_dir("inspect");
    let (code, payload) = warp_build::inspect_source_json(&bundle.to_string_lossy());
    assert_eq!(code, 0);
    assert_eq!(payload["status"], "ok");
    assert_eq!(payload["bundle_id"], "pmma_param_bundle_v1");
    assert_eq!(payload["unit_tokens"], json!(["A", "B", "H", "T"]));
    assert_eq!(payload["motif_tokens"], json!(["M2"]));
    assert_eq!(payload["topology_transfer_supported"], json!(true));
    assert_eq!(
        payload["sequence_token_support"]["tokens"],
        json!(["H", "A", "B", "T", "M2"])
    );
    assert_eq!(
        payload["target_count_semantics"]["linear_homopolymer"]["n_repeat"],
        "total_final_residues_when_terminal_aware"
    );
    assert_eq!(
        payload["conformation_semantics"]["extended"],
        "deterministic_collinear_end_to_end_initial_chain"
    );
    let caps = warp_build::capabilities();
    assert_eq!(
        caps["target_count_semantics"]["linear_homopolymer"]["n_repeat"],
        "total_final_residues_when_terminal_aware"
    );
    assert_eq!(
        caps["conformation_semantics"]["extended"],
        "deterministic_collinear_end_to_end_initial_chain"
    );
    assert_eq!(
        caps["executable_target_modes"],
        json!([
            "linear_homopolymer",
            "linear_sequence_polymer",
            "block_copolymer",
            "random_copolymer",
            "star_polymer",
            "branched_polymer",
            "polymer_graph"
        ])
    );
    assert_eq!(
        caps["supported_realization_modes"],
        json!(["extended", "random_walk", "aligned", "ensemble"])
    );
    assert_eq!(
        caps["supported_validation_depths"],
        json!(["shallow", "deep"])
    );
    assert_eq!(
        caps["supports_relax_modes"],
        json!(["graph_spring", "targeted_steric"])
    );
    assert_eq!(caps["default_validation_depth"], "deep");
    assert_eq!(caps["supported_qc_policies"], json!(["strict", "salvage"]));
    assert_eq!(caps["supports_named_termini_tokens"], json!(true));
    assert_eq!(
        warp_build::example_request("block")["target"]["mode"],
        "block_copolymer"
    );
    assert_eq!(
        warp_build::example_request("ensemble")["realization"]["ensemble_size"],
        4
    );
    assert_eq!(
        warp_build::example_request("sequence")["target"]["termini"],
        json!({"head": "H", "tail": "T"})
    );
    assert_eq!(
        warp_build::example_request("graph")["target"]["mode"],
        "polymer_graph"
    );
}

#[test]
fn inspect_source_rejects_template_selector_mismatch() {
    let (_dir, bundle) = make_bundle_dir("inspect_selector_mismatch");
    let mut payload: Value =
        serde_json::from_str(&fs::read_to_string(&bundle).expect("read bundle"))
            .expect("bundle json");
    payload["unit_library"]["A"]["template_resname"] = Value::String("HDA".into());
    write_json(&bundle, &payload);

    let (code, payload) = warp_build::inspect_source_json(&bundle.to_string_lossy());
    assert_eq!(code, 2);
    assert_eq!(payload["status"], "error");
    assert!(payload["errors"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["message"]
            .as_str()
            .unwrap_or("")
            .contains("attach atom 'C2' missing from template 'HDA'")));
}

#[test]
fn example_request_uses_bundle_id_from_existing_bundle_path() {
    let (_dir, bundle) = make_bundle_dir("example_request_bundle_id");
    let mut payload: Value =
        serde_json::from_str(&fs::read_to_string(&bundle).expect("read bundle"))
            .expect("bundle json");
    payload["bundle_id"] = Value::String("custom_paa_bundle_v1".into());
    write_json(&bundle, &payload);

    let request = warp_build::example_request_for_bundle("random_walk", &bundle.to_string_lossy());
    assert_eq!(
        request["source_ref"]["bundle_id"],
        Value::String("custom_paa_bundle_v1".into())
    );
    assert_eq!(
        request["request_id"],
        Value::String("custom_paa_50mer-build-001".into())
    );
    assert_eq!(
        request["artifacts"]["coordinates"],
        Value::String("custom_paa_50mer.pdb".into())
    );
    assert_eq!(
        request["artifacts"]["topology"],
        Value::String("custom_paa_50mer.prmtop".into())
    );
}

#[test]
fn validate_rejects_unknown_repeat_unit() {
    let (_dir, bundle) = make_bundle_dir("validate");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "build-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "linear_homopolymer",
            "repeat_unit": "C",
            "n_repeat": 4,
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"},
        },
        "realization": {
            "conformation_mode": "extended"
        },
        "artifacts": {
            "coordinates": temp_path("coords").to_string_lossy(),
            "build_manifest": temp_path("manifest").to_string_lossy(),
            "charge_manifest": temp_path("charge").to_string_lossy(),
        }
    });
    let (code, payload) =
        warp_build::validate_request_json(&serde_json::to_string(&request).expect("serialize"));
    assert_eq!(code, 2);
    assert_eq!(payload["errors"][0]["code"], "E_UNKNOWN_TOKEN");
}

#[test]
fn validate_rejects_random_copolymer_total_units_mismatch() {
    let (_dir, bundle) = make_bundle_dir("random_validate");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "build-002",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "random_copolymer",
            "composition": {"A": 2, "B": 1},
            "total_units": 99,
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"},
        },
        "realization": {
            "conformation_mode": "random_walk",
            "seed": 12345
        },
        "artifacts": {
            "coordinates": temp_path("coords").to_string_lossy(),
            "build_manifest": temp_path("manifest").to_string_lossy(),
            "charge_manifest": temp_path("charge").to_string_lossy(),
        }
    });
    let (code, payload) =
        warp_build::validate_request_json(&serde_json::to_string(&request).expect("serialize"));
    assert_eq!(code, 2);
    assert_eq!(payload["errors"][0]["path"], "/target/total_units");
}

#[test]
fn validate_rejects_missing_seed_for_stochastic_modes() {
    let (_dir, bundle) = make_bundle_dir("random_seed_validate");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "build-seed-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "linear_homopolymer",
            "repeat_unit": "A",
            "n_repeat": 4,
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"},
        },
        "realization": {
            "conformation_mode": "random_walk"
        },
        "artifacts": {
            "coordinates": temp_path("seed_coords").to_string_lossy(),
            "build_manifest": temp_path("seed_manifest").to_string_lossy(),
            "charge_manifest": temp_path("seed_charge").to_string_lossy(),
        }
    });
    let (code, payload) =
        warp_build::validate_request_json(&serde_json::to_string(&request).expect("serialize"));
    assert_eq!(code, 2);
    assert_eq!(payload["errors"][0]["code"], "E_MISSING_SEED");
    assert_eq!(payload["errors"][0]["path"], "/realization/seed");
}

#[test]
fn extended_linear_build_is_collinear_end_to_end() {
    let (_dir, bundle) = make_bundle_dir("extended_collinear");
    let coords = temp_path("extended_collinear_coords.pdb");
    let build_manifest = temp_path("extended_collinear_manifest.json");
    let charge_manifest = temp_path("extended_collinear_charge.json");
    let topology_graph = temp_path("extended_collinear_graph.json");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "extended-collinear-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "linear_homopolymer",
            "repeat_unit": "A",
            "n_repeat": 8,
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"},
        },
        "realization": {
            "conformation_mode": "extended"
        },
        "artifacts": {
            "coordinates": coords.to_string_lossy(),
            "build_manifest": build_manifest.to_string_lossy(),
            "charge_manifest": charge_manifest.to_string_lossy(),
            "topology_graph": topology_graph.to_string_lossy(),
        }
    });
    let (code, payload) =
        warp_build::run_request_json(&serde_json::to_string(&request).expect("serialize"), false);
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );

    let positions = pdb_atom_positions(&coords);
    assert_eq!(positions.len(), 8);
    let max_off_axis = positions
        .iter()
        .map(|pos| pos[1].abs().max(pos[2].abs()))
        .fold(0.0f64, f64::max);
    assert!(
        max_off_axis <= 1.0e-3,
        "extended build should place the chain on one axis; max off-axis={max_off_axis}"
    );
    let contour = positions
        .windows(2)
        .map(|pair| {
            let dx = pair[1][0] - pair[0][0];
            let dy = pair[1][1] - pair[0][1];
            let dz = pair[1][2] - pair[0][2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .sum::<f64>();
    let end_to_end = {
        let first = positions.first().expect("first atom");
        let last = positions.last().expect("last atom");
        let dx = last[0] - first[0];
        let dy = last[1] - first[1];
        let dz = last[2] - first[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    };
    assert!(
        end_to_end / contour >= 0.999,
        "extended build should be end-to-end stretched; end_to_end={end_to_end} contour={contour}"
    );

    let manifest: Value =
        serde_json::from_str(&fs::read_to_string(&build_manifest).expect("read manifest"))
            .expect("parse manifest");
    assert_eq!(manifest["summary"]["total_residues"], 8);
    assert_eq!(
        pdb_record_count(&coords, "TER"),
        0,
        "continuous polymer PDB output must not split every residue into a separate chain"
    );
    let graph: Value =
        serde_json::from_str(&fs::read_to_string(&topology_graph).expect("read graph"))
            .expect("parse topology graph");
    assert_eq!(graph["build_plan"]["max_branch_depth"], 0);
    let residues = graph["residues"].as_array().expect("graph residues");
    assert_eq!(residues.len(), 8);
    assert!(residues.iter().all(|residue| residue["branch_depth"] == 0));
    assert!(residues.iter().all(|residue| residue["branch_path"]
        .as_str()
        .map(|path| !path.contains('>'))
        .unwrap_or(false)));
    let max_bond_distance = max_inter_residue_bond_distance(&coords, &topology_graph);
    assert!(
        (max_bond_distance - 1.53).abs() <= 0.05,
        "extended build should preserve inter-residue bond length; max_bond_distance={max_bond_distance}"
    );
}

#[test]
fn run_build_writes_polymer_artifacts() {
    let (_dir, bundle) = make_bundle_dir("run");
    let coords = temp_path("coords.pdb");
    let build_manifest = temp_path("build_manifest.json");
    let charge_manifest = temp_path("charge_manifest.json");
    let topology = temp_path("topology.prmtop");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "build-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "linear_homopolymer",
            "repeat_unit": "A",
            "n_repeat": 4,
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "syndiotactic"},
        },
        "realization": {
            "conformation_mode": "random_walk",
            "seed": 12345
        },
        "artifacts": {
            "coordinates": coords.to_string_lossy(),
            "build_manifest": build_manifest.to_string_lossy(),
            "charge_manifest": charge_manifest.to_string_lossy(),
            "topology": topology.to_string_lossy(),
        }
    });
    let (code, payload) =
        warp_build::run_request_json(&serde_json::to_string(&request).expect("serialize"), false);
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    assert!(coords.exists());
    assert!(build_manifest.exists());
    assert!(charge_manifest.exists());
    assert!(topology.exists());

    let manifest: Value =
        serde_json::from_str(&fs::read_to_string(&build_manifest).expect("read manifest"))
            .expect("parse manifest");
    assert_eq!(manifest["schema_version"], "warp-build.manifest.v1");
    assert_eq!(payload["summary"]["n_repeat_requested"], 4);
    assert_eq!(payload["summary"]["requested_total_residues"], 4);
    assert_eq!(payload["summary"]["middle_repeat_units"], 2);
    assert_eq!(payload["summary"]["terminal_units"], 2);
    assert_eq!(manifest["summary"]["n_repeat_requested"], 4);
    assert_eq!(manifest["summary"]["requested_total_residues"], 4);
    assert_eq!(manifest["summary"]["middle_repeat_units"], 2);
    assert_eq!(manifest["summary"]["terminal_units"], 2);
    assert_eq!(manifest["summary"]["total_repeat_units"], 2);
    assert_eq!(manifest["summary"]["total_residues"], 4);
    assert_eq!(manifest["realization"]["seed"], 12345);
    assert_eq!(manifest["realization"]["seed_policy"], "explicit");
    assert_eq!(manifest["summary"]["bond_count"], 3);
    assert_eq!(
        manifest["summary"]["resolved_sequence"],
        json!(["H", "A", "A", "T"])
    );
    assert_eq!(
        manifest["provenance"]["target_normalization"]["applied"],
        "linear_homopolymer_to_linear_sequence_polymer"
    );
    assert_eq!(
        manifest["md_ready_handoff"]["coordinates"]["format"],
        "pdb-strict"
    );
    assert_eq!(
        manifest["summary"]["qc"]["severe_bond_violations"],
        json!([])
    );
    assert_eq!(manifest["summary"]["solver"]["mode"], "torsion_solve");
    assert_eq!(
        manifest["summary"]["solver_cleanup"]["mode"],
        "graph_spring"
    );
    assert_eq!(manifest["summary"]["overlap_status"]["status"], "clear");
    assert_eq!(
        manifest["summary"]["overlap_status"]["may_report_no_overlaps"],
        true
    );
    assert_eq!(
        manifest["summary"]["overlap_status"]["report_source"],
        "solver_cleanup"
    );
    assert_eq!(manifest["summary"]["overlap_status"]["overlap_pairs"], 0);
    assert_eq!(payload["summary"]["overlap_status"]["status"], "clear");
    assert!(manifest["summary"]["timings_ms"]["compile_plan"]
        .as_u64()
        .is_some());
    assert!(payload["summary"]["timings_ms"]["build_graph"]
        .as_u64()
        .is_some());
    assert_eq!(
        manifest["artifacts"]["topology"],
        topology.to_string_lossy().to_string()
    );
    assert!(manifest["artifact_digests"]["coordinates"]
        .as_str()
        .expect("coordinates digest")
        .starts_with("sha256:"));
    assert!(manifest["artifact_digests"]["topology"]
        .as_str()
        .expect("topology digest")
        .starts_with("sha256:"));
    assert!(manifest["artifact_digests"]["charge_manifest"]
        .as_str()
        .expect("charge digest")
        .starts_with("sha256:"));

    let charge: Value =
        serde_json::from_str(&fs::read_to_string(&charge_manifest).expect("read charge"))
            .expect("parse charge");
    assert_eq!(charge["schema_version"], "warp-build.charge-manifest.v1");
    assert_eq!(charge["net_charge_e"], 3.0);
    assert_eq!(
        charge["target_topology_ref"],
        topology.to_string_lossy().to_string()
    );
    let topology_text = fs::read_to_string(&topology).expect("read topology");
    assert!(topology_text.contains("%FLAG BONDS_WITHOUT_HYDROGEN"));
    let topology_graph_path = manifest["artifacts"]["topology_graph"]
        .as_str()
        .expect("topology graph path");
    let topology_graph: Value = serde_json::from_str(
        &fs::read_to_string(topology_graph_path).expect("read topology graph"),
    )
    .expect("parse topology graph");
    assert_eq!(topology_graph["relax_metadata"]["final_overlap_pairs"], 0);
    assert_eq!(
        topology_graph["relax_metadata"]["overlap_metric"],
        "vdw_overlap_pairs_excluding_1_2_and_1_3"
    );
}

#[test]
fn terminal_aware_homopolymer_n_repeat_means_total_residues() {
    let (_dir, bundle) = make_bundle_dir("terminal_n_repeat_total");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "terminal-total-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "linear_homopolymer",
            "repeat_unit": "A",
            "n_repeat": 50,
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"},
        },
        "realization": {
            "conformation_mode": "extended"
        },
        "artifacts": {
            "coordinates": temp_path("terminal_total_coords.pdb").to_string_lossy(),
            "build_manifest": temp_path("terminal_total_manifest.json").to_string_lossy(),
            "charge_manifest": temp_path("terminal_total_charge.json").to_string_lossy(),
        }
    });
    let (code, payload) =
        warp_build::validate_request_json(&serde_json::to_string(&request).expect("serialize"));
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    let sequence = payload["normalized_request"]["target"]["sequence"]
        .as_array()
        .expect("normalized sequence");
    assert_eq!(sequence.len(), 50);
    assert_eq!(sequence.first().expect("head"), "H");
    assert_eq!(sequence.last().expect("tail"), "T");
    assert_eq!(
        sequence
            .iter()
            .filter(|item| item.as_str() == Some("A"))
            .count(),
        48
    );
    assert_eq!(
        payload["resolved_inputs"]["target_normalization"]["resolved_repeat_units"],
        48
    );
}

#[test]
fn terminal_aware_homopolymer_total_units_matches_n_repeat_semantics() {
    let (_dir, bundle) = make_bundle_dir("terminal_total_units");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "terminal-total-units-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "linear_homopolymer",
            "repeat_unit": "A",
            "total_units": 50,
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"},
        },
        "realization": {
            "conformation_mode": "extended"
        },
        "artifacts": {
            "coordinates": temp_path("terminal_total_units_coords.pdb").to_string_lossy(),
            "build_manifest": temp_path("terminal_total_units_manifest.json").to_string_lossy(),
            "charge_manifest": temp_path("terminal_total_units_charge.json").to_string_lossy(),
        }
    });
    let (code, payload) =
        warp_build::validate_request_json(&serde_json::to_string(&request).expect("serialize"));
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    assert_eq!(
        payload["resolved_inputs"]["target_normalization"]["requested_total_units"],
        50
    );
    assert_eq!(
        payload["resolved_inputs"]["target_normalization"]["resolved_repeat_units"],
        48
    );
    assert_eq!(
        payload["normalized_request"]["target"]["sequence"]
            .as_array()
            .expect("sequence")
            .len(),
        50
    );
}

#[test]
fn run_build_with_explicit_relax_records_cleanup_and_user_relax() {
    let (_dir, bundle) = make_bundle_dir("run_relax");
    let coords = temp_path("coords_relax.pdb");
    let raw_coords = temp_path("coords_relax_raw.pdb");
    let build_manifest = temp_path("build_manifest_relax.json");
    let charge_manifest = temp_path("charge_manifest_relax.json");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "build-relax-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "linear_homopolymer",
            "repeat_unit": "A",
            "n_repeat": 4,
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "syndiotactic"},
        },
        "realization": {
            "conformation_mode": "random_walk",
            "seed": 12346,
            "relax": {
                "mode": "graph_spring",
                "steps": 8,
                "step_scale": 0.2,
                "clash_scale": 0.9
            }
        },
        "artifacts": {
            "coordinates": coords.to_string_lossy(),
            "raw_coordinates": raw_coords.to_string_lossy(),
            "build_manifest": build_manifest.to_string_lossy(),
            "charge_manifest": charge_manifest.to_string_lossy(),
        }
    });
    let (code, payload) =
        warp_build::run_request_json(&serde_json::to_string(&request).expect("serialize"), false);
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    assert!(raw_coords.exists());
    let manifest: Value =
        serde_json::from_str(&fs::read_to_string(&build_manifest).expect("read manifest"))
            .expect("parse manifest");
    assert_eq!(
        manifest["summary"]["solver_cleanup"]["mode"],
        "graph_spring"
    );
    assert_eq!(manifest["summary"]["relax"]["mode"], "graph_spring");
    assert_eq!(
        manifest["summary"]["solver_cleanup"]["final_overlap_pairs"],
        0
    );
    assert_eq!(manifest["summary"]["relax"]["final_overlap_pairs"], 0);
    assert_eq!(manifest["summary"]["overlap_status"]["status"], "clear");
    assert_eq!(
        manifest["summary"]["overlap_status"]["may_report_no_overlaps"],
        true
    );
    assert_eq!(
        manifest["summary"]["overlap_status"]["report_source"],
        "relax"
    );
    assert_eq!(manifest["summary"]["overlap_status"]["overlap_pairs"], 0);
    assert_eq!(payload["summary"]["overlap_status"]["status"], "clear");
    assert_eq!(
        payload["summary"]["overlap_status"]["report_source"],
        "relax"
    );
    assert!(
        manifest["summary"]["relax"]["fallback_mode"].is_null()
            || manifest["summary"]["relax"]["fallback_mode"] == "targeted_steric"
    );
    assert_eq!(
        manifest["summary"]["relax"]["raw_coordinates"],
        raw_coords.to_string_lossy().to_string()
    );
}

#[test]
fn run_build_accepts_targeted_stearic_alias_and_reports_canonical_mode() {
    let (_dir, bundle) = make_bundle_dir("run_targeted_stearic_alias");
    let coords = temp_path("coords_targeted_stearic_alias.pdb");
    let raw_coords = temp_path("coords_targeted_stearic_alias_raw.pdb");
    let build_manifest = temp_path("build_manifest_targeted_stearic_alias.json");
    let charge_manifest = temp_path("charge_manifest_targeted_stearic_alias.json");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "build-targeted-stearic-alias-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "linear_homopolymer",
            "repeat_unit": "A",
            "n_repeat": 4,
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "syndiotactic"},
        },
        "realization": {
            "conformation_mode": "random_walk",
            "seed": 12347,
            "relax": {
                "mode": "targeted_stearic",
                "steps": 8,
                "step_scale": 0.2,
                "clash_scale": 0.9
            }
        },
        "artifacts": {
            "coordinates": coords.to_string_lossy(),
            "raw_coordinates": raw_coords.to_string_lossy(),
            "build_manifest": build_manifest.to_string_lossy(),
            "charge_manifest": charge_manifest.to_string_lossy(),
        }
    });
    let (code, payload) =
        warp_build::run_request_json(&serde_json::to_string(&request).expect("serialize"), false);
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    let manifest: Value =
        serde_json::from_str(&fs::read_to_string(&build_manifest).expect("read manifest"))
            .expect("parse manifest");
    assert_eq!(manifest["summary"]["relax"]["mode"], "targeted_steric");
    assert_eq!(manifest["summary"]["overlap_status"]["status"], "clear");
    assert_eq!(
        manifest["summary"]["overlap_status"]["report_source"],
        "relax"
    );
    assert!(manifest["summary"]["relax"]["initial_overlap_pairs"]
        .as_u64()
        .is_some());
    assert!(manifest["summary"]["relax"]["final_overlap_pairs"]
        .as_u64()
        .is_some());
}

fn overlap_report_snapshot(report: &Value) -> Value {
    if report.is_null() {
        return Value::Null;
    }
    json!({
        "mode": report["mode"],
        "steps_requested": report["steps_requested"],
        "steps_executed": report["steps_executed"],
        "initial_max_clash": report["initial_max_clash"],
        "final_max_clash": report["final_max_clash"],
        "initial_overlap_pairs": report["initial_overlap_pairs"],
        "final_overlap_pairs": report["final_overlap_pairs"],
        "fallback_mode": report["fallback_mode"],
        "fallback_steps_requested": report["fallback_steps_requested"],
        "fallback_steps_executed": report["fallback_steps_executed"],
        "pre_fallback_max_clash": report["pre_fallback_max_clash"],
        "pre_fallback_overlap_pairs": report["pre_fallback_overlap_pairs"],
        "movable_atom_count": report["movable_atom_count"],
        "max_atom_displacement_angstrom": report["max_atom_displacement_angstrom"],
    })
}

fn run_overlap_corpus_case(case_name: &str, request: Value) -> Value {
    let (code, payload) =
        warp_build::run_request_json(&serde_json::to_string(&request).expect("serialize"), false);
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    let summary = &payload["summary"];
    json!({
        "case": case_name,
        "build_mode": summary["build_mode"],
        "conformation_mode": summary["conformation_mode"],
        "atom_count": summary["atom_count"],
        "solver_cleanup": overlap_report_snapshot(&summary["solver_cleanup"]),
        "relax": overlap_report_snapshot(&summary["relax"]),
    })
}

#[derive(Clone, Debug, Serialize)]
struct OverlapSweepRecord {
    case: String,
    seed: u64,
    atom_count: u64,
    fallback_triggered: bool,
    worst_pre_fallback_overlap_pairs: u64,
    worst_initial_overlap_pairs: u64,
    worst_final_overlap_pairs: u64,
    worst_max_displacement_angstrom: f64,
    solver_cleanup: Value,
    relax: Value,
}

#[derive(Clone, Debug, Serialize)]
struct OverlapSweepCaseSummary {
    case: String,
    seeds_scanned: usize,
    fallback_trigger_count: usize,
    max_pre_fallback_overlap_pairs: u64,
    max_initial_overlap_pairs: u64,
    max_final_overlap_pairs: u64,
    max_atom_displacement_angstrom: f64,
    top_offenders: Vec<OverlapSweepRecord>,
}

fn overlap_report_u64(report: &Value, key: &str) -> u64 {
    report.get(key).and_then(Value::as_u64).unwrap_or_default()
}

fn overlap_report_f64(report: &Value, key: &str) -> f64 {
    report.get(key).and_then(Value::as_f64).unwrap_or_default()
}

fn overlap_report_has_fallback(report: &Value) -> bool {
    report
        .get("fallback_mode")
        .map(|value| !value.is_null())
        .unwrap_or(false)
}

fn graph_spring_relax_request(steps: u64, step_scale: f64) -> Value {
    json!({
        "mode": "graph_spring",
        "steps": steps,
        "step_scale": step_scale,
        "clash_scale": 0.9
    })
}

fn run_overlap_sweep_case(case_name: &str, seed: u64, request: Value) -> OverlapSweepRecord {
    let (code, payload) =
        warp_build::run_request_json(&serde_json::to_string(&request).expect("serialize"), false);
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    let summary = &payload["summary"];
    let solver_cleanup = overlap_report_snapshot(&summary["solver_cleanup"]);
    let relax = overlap_report_snapshot(&summary["relax"]);
    OverlapSweepRecord {
        case: case_name.into(),
        seed,
        atom_count: summary["atom_count"].as_u64().unwrap_or_default(),
        fallback_triggered: overlap_report_has_fallback(&solver_cleanup)
            || overlap_report_has_fallback(&relax),
        worst_pre_fallback_overlap_pairs: overlap_report_u64(
            &solver_cleanup,
            "pre_fallback_overlap_pairs",
        )
        .max(overlap_report_u64(&relax, "pre_fallback_overlap_pairs")),
        worst_initial_overlap_pairs: overlap_report_u64(&solver_cleanup, "initial_overlap_pairs")
            .max(overlap_report_u64(&relax, "initial_overlap_pairs")),
        worst_final_overlap_pairs: overlap_report_u64(&solver_cleanup, "final_overlap_pairs")
            .max(overlap_report_u64(&relax, "final_overlap_pairs")),
        worst_max_displacement_angstrom: overlap_report_f64(
            &solver_cleanup,
            "max_atom_displacement_angstrom",
        )
        .max(overlap_report_f64(&relax, "max_atom_displacement_angstrom")),
        solver_cleanup,
        relax,
    }
}

fn overlap_seed_sweep_star_request(bundle: &Path, seed: u64) -> Value {
    json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": format!("overlap-sweep-star-{seed:04}"),
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "star_polymer",
            "core_token": "A",
            "core_junctions": ["head", "tail"],
            "arm_sequence": ["A", "B", "A", "B"],
            "arm_repeat_count": 6,
            "termini": {"head": "H", "tail": "T"},
            "stereochemistry": {"mode": "syndiotactic"}
        },
        "realization": {
            "conformation_mode": "random_walk",
            "seed": seed,
            "relax": graph_spring_relax_request(24, 0.3)
        },
        "artifacts": {
            "coordinates": temp_path(&format!("overlap_sweep_star_coords_{seed}.pdb")).to_string_lossy(),
            "build_manifest": temp_path(&format!("overlap_sweep_star_manifest_{seed}.json")).to_string_lossy(),
            "charge_manifest": temp_path(&format!("overlap_sweep_star_charge_{seed}.json")).to_string_lossy(),
            "topology_graph": temp_path(&format!("overlap_sweep_star_graph_{seed}.json")).to_string_lossy(),
        }
    })
}

fn overlap_seed_sweep_branched_request(bundle: &Path, seed: u64) -> Value {
    json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": format!("overlap-sweep-branched-{seed:04}"),
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "branched_polymer",
            "branch_tree": {
                "token": "A",
                "children": [
                    {
                        "parent_junction": "head",
                        "child_junction": "head",
                        "sequence": ["B", "A", "B", "A"],
                        "repeat_count": 4,
                        "child": {
                            "token": "B",
                            "children": [
                                {
                                    "parent_junction": "head",
                                    "child_junction": "head",
                                    "sequence": ["A", "B", "A"],
                                    "repeat_count": 3,
                                    "child": {
                                        "token": "A",
                                        "children": [
                                            {
                                                "parent_junction": "head",
                                                "child_junction": "head",
                                                "sequence": ["B", "A"],
                                                "repeat_count": 2
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            "termini": {"head": "H", "tail": "T"},
            "stereochemistry": {"mode": "syndiotactic"}
        },
        "realization": {
            "conformation_mode": "random_walk",
            "seed": seed,
            "relax": graph_spring_relax_request(24, 0.3)
        },
        "artifacts": {
            "coordinates": temp_path(&format!("overlap_sweep_branched_coords_{seed}.pdb")).to_string_lossy(),
            "build_manifest": temp_path(&format!("overlap_sweep_branched_manifest_{seed}.json")).to_string_lossy(),
            "charge_manifest": temp_path(&format!("overlap_sweep_branched_charge_{seed}.json")).to_string_lossy(),
            "topology_graph": temp_path(&format!("overlap_sweep_branched_graph_{seed}.json")).to_string_lossy(),
        }
    })
}

fn overlap_seed_sweep_graph_request(bundle: &Path, seed: u64) -> Value {
    json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": format!("overlap-sweep-graph-{seed:04}"),
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "polymer_graph",
            "graph_root": "g1",
            "graph_nodes": [
                {"id": "g1", "token": "M2"},
                {"id": "g2", "token": "M2"},
                {"id": "g3", "token": "M2"},
                {"id": "g4", "token": "A"},
                {"id": "g5", "token": "B"},
                {"id": "g6", "token": "M2"},
                {"id": "g7", "token": "A"},
                {"id": "g8", "token": "B"}
            ],
            "graph_edges": [
                {"id": "e1", "from": "g1", "to": "g2", "from_junction": "tail", "to_junction": "head"},
                {"id": "e2", "from": "g1", "to": "g3", "from_junction": "head", "to_junction": "head"},
                {"id": "e3", "from": "g2", "to": "g4", "from_junction": "tail", "to_junction": "head"},
                {"id": "e4", "from": "g3", "to": "g5", "from_junction": "tail", "to_junction": "head"},
                {"id": "e5", "from": "g4", "to": "g6", "from_junction": "tail", "to_junction": "head"},
                {"id": "e6", "from": "g5", "to": "g7", "from_junction": "tail", "to_junction": "head"},
                {"id": "e7", "from": "g6", "to": "g8", "from_junction": "tail", "to_junction": "head"}
            ],
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "random_walk",
            "seed": seed,
            "relax": graph_spring_relax_request(24, 0.3)
        },
        "conformer_policy": {
            "layout_mode": "mixed",
            "default_torsion": "trans",
            "branch_spread": "staggered",
            "ring_mode": "planar"
        },
        "artifacts": {
            "coordinates": temp_path(&format!("overlap_sweep_graph_coords_{seed}.pdb")).to_string_lossy(),
            "build_manifest": temp_path(&format!("overlap_sweep_graph_manifest_{seed}.json")).to_string_lossy(),
            "charge_manifest": temp_path(&format!("overlap_sweep_graph_charge_{seed}.json")).to_string_lossy(),
            "topology_graph": temp_path(&format!("overlap_sweep_graph_topology_{seed}.json")).to_string_lossy()
        }
    })
}

fn sort_overlap_sweep_records(records: &mut [OverlapSweepRecord]) {
    records.sort_by(|left, right| {
        right
            .fallback_triggered
            .cmp(&left.fallback_triggered)
            .then_with(|| {
                right
                    .worst_pre_fallback_overlap_pairs
                    .cmp(&left.worst_pre_fallback_overlap_pairs)
            })
            .then_with(|| {
                right
                    .worst_initial_overlap_pairs
                    .cmp(&left.worst_initial_overlap_pairs)
            })
            .then_with(|| {
                right
                    .worst_final_overlap_pairs
                    .cmp(&left.worst_final_overlap_pairs)
            })
            .then_with(|| {
                right
                    .worst_max_displacement_angstrom
                    .partial_cmp(&left.worst_max_displacement_angstrom)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
}

fn overlap_sweep_summary(case: &str, records: &[OverlapSweepRecord]) -> OverlapSweepCaseSummary {
    let mut offenders = records.to_vec();
    sort_overlap_sweep_records(&mut offenders);
    OverlapSweepCaseSummary {
        case: case.into(),
        seeds_scanned: records.len(),
        fallback_trigger_count: records
            .iter()
            .filter(|record| record.fallback_triggered)
            .count(),
        max_pre_fallback_overlap_pairs: records
            .iter()
            .map(|record| record.worst_pre_fallback_overlap_pairs)
            .max()
            .unwrap_or_default(),
        max_initial_overlap_pairs: records
            .iter()
            .map(|record| record.worst_initial_overlap_pairs)
            .max()
            .unwrap_or_default(),
        max_final_overlap_pairs: records
            .iter()
            .map(|record| record.worst_final_overlap_pairs)
            .max()
            .unwrap_or_default(),
        max_atom_displacement_angstrom: records
            .iter()
            .map(|record| record.worst_max_displacement_angstrom)
            .fold(0.0, f64::max),
        top_offenders: offenders.into_iter().take(5).collect(),
    }
}

fn assert_overlap_sweep_record_is_clear(record: &OverlapSweepRecord) {
    assert_eq!(
        record.worst_final_overlap_pairs,
        0,
        "{}",
        serde_json::to_string_pretty(record).unwrap()
    );
    assert_eq!(
        record.solver_cleanup["final_overlap_pairs"].as_u64(),
        Some(0),
        "{}",
        serde_json::to_string_pretty(record).unwrap()
    );
    if !record.relax.is_null() {
        assert_eq!(
            record.relax["final_overlap_pairs"].as_u64(),
            Some(0),
            "{}",
            serde_json::to_string_pretty(record).unwrap()
        );
    }
}

#[test]
#[ignore = "manual overlap tuning corpus"]
fn overlap_regression_corpus_reports_metrics() {
    let mut results = Vec::new();

    let (_dir, linear_bundle) = make_bundle_dir("overlap_corpus_linear");
    results.push(run_overlap_corpus_case(
        "linear_random_walk",
        json!({
            "schema_version": "warp-build.agent.v1",
            "request_id": "overlap-corpus-linear-001",
            "source_ref": {
                "bundle_id": "pmma_param_bundle_v1",
                "bundle_path": linear_bundle.to_string_lossy(),
            },
            "target": {
                "mode": "linear_homopolymer",
                "repeat_unit": "A",
                "n_repeat": 6,
                "termini": {"head": "default", "tail": "default"},
                "stereochemistry": {"mode": "syndiotactic"},
            },
            "realization": {
                "conformation_mode": "random_walk",
                "seed": 3101,
                "relax": {
                    "mode": "graph_spring",
                    "steps": 16,
                    "step_scale": 0.25,
                    "clash_scale": 0.9
                }
            },
            "artifacts": {
                "coordinates": temp_path("overlap_corpus_linear_coords.pdb").to_string_lossy(),
                "build_manifest": temp_path("overlap_corpus_linear_manifest.json").to_string_lossy(),
                "charge_manifest": temp_path("overlap_corpus_linear_charge.json").to_string_lossy(),
            }
        }),
    ));

    let (_dir, star_bundle) = make_bundle_dir("overlap_corpus_star");
    results.push(run_overlap_corpus_case(
        "star_random_walk",
        json!({
            "schema_version": "warp-build.agent.v1",
            "request_id": "overlap-corpus-star-001",
            "source_ref": {
                "bundle_id": "pmma_param_bundle_v1",
                "bundle_path": star_bundle.to_string_lossy(),
            },
            "target": {
                "mode": "star_polymer",
                "core_token": "A",
                "core_junctions": ["head", "tail"],
                "arm_sequence": ["B", "A"],
                "arm_repeat_count": 3,
                "termini": {"head": "H", "tail": "T"},
                "stereochemistry": {"mode": "inherit"}
            },
            "realization": {
                "conformation_mode": "random_walk",
                "seed": 3102,
                "relax": {
                    "mode": "graph_spring",
                    "steps": 20,
                    "step_scale": 0.3,
                    "clash_scale": 0.9
                }
            },
            "artifacts": {
                "coordinates": temp_path("overlap_corpus_star_coords.pdb").to_string_lossy(),
                "build_manifest": temp_path("overlap_corpus_star_manifest.json").to_string_lossy(),
                "charge_manifest": temp_path("overlap_corpus_star_charge.json").to_string_lossy(),
                "topology_graph": temp_path("overlap_corpus_star_graph.json").to_string_lossy(),
            }
        }),
    ));

    let (_dir, branched_bundle) = copy_fixture_dir("pmma_branched_mix", "overlap_corpus_branched");
    results.push(run_overlap_corpus_case(
        "branched_mix_random_walk",
        json!({
            "schema_version": "warp-build.agent.v1",
            "request_id": "overlap-corpus-branched-001",
            "source_ref": {
                "bundle_id": "pmma_branched_mix_bundle_v1",
                "bundle_path": branched_bundle.to_string_lossy(),
            },
            "target": {
                "mode": "branched_polymer",
                "branch_tree": {
                    "token": "A",
                    "children": [
                        {
                            "parent_junction": "head",
                            "child_junction": "head",
                            "sequence": ["ARM2"],
                            "repeat_count": 1,
                            "child": {
                                "token": "B",
                                "children": [
                                    {
                                        "parent_junction": "head",
                                        "child_junction": "head",
                                        "sequence": ["A"],
                                        "repeat_count": 1
                                    }
                                ]
                            }
                        }
                    ]
                },
                "termini": {"head": "default", "tail": "default"},
                "stereochemistry": {"mode": "inherit"}
            },
            "realization": {
                "conformation_mode": "random_walk",
                "seed": 3103,
                "relax": {
                    "mode": "graph_spring",
                    "steps": 20,
                    "step_scale": 0.3,
                    "clash_scale": 0.9
                }
            },
            "artifacts": {
                "coordinates": temp_path("overlap_corpus_branched_coords.pdb").to_string_lossy(),
                "build_manifest": temp_path("overlap_corpus_branched_manifest.json").to_string_lossy(),
                "charge_manifest": temp_path("overlap_corpus_branched_charge.json").to_string_lossy(),
                "topology_graph": temp_path("overlap_corpus_branched_graph.json").to_string_lossy()
            }
        }),
    ));

    let (_dir, graph_bundle) = copy_fixture_dir("pmma_motif", "overlap_corpus_graph");
    results.push(run_overlap_corpus_case(
        "motif_graph_random_walk",
        json!({
            "schema_version": "warp-build.agent.v1",
            "request_id": "overlap-corpus-graph-001",
            "source_ref": {
                "bundle_id": "pmma_fixture_bundle_v1",
                "bundle_path": graph_bundle.to_string_lossy(),
            },
            "target": {
                "mode": "polymer_graph",
                "graph_root": "g1",
                "graph_nodes": [
                    {"id": "g1", "token": "M2"},
                    {"id": "g2", "token": "A"}
                ],
                "graph_edges": [
                    {"id": "e1", "from": "g1", "to": "g2", "from_junction": "tail", "to_junction": "head"}
                ],
                "termini": {"head": "default", "tail": "default"},
                "stereochemistry": {"mode": "inherit"}
            },
            "realization": {
                "conformation_mode": "random_walk",
                "seed": 3104,
                "relax": {
                    "mode": "graph_spring",
                    "steps": 16,
                    "step_scale": 0.25,
                    "clash_scale": 0.9
                }
            },
            "conformer_policy": {
                "layout_mode": "mixed",
                "default_torsion": "trans",
                "branch_spread": "staggered",
                "ring_mode": "planar",
                "edge_overrides": [
                    {"edge_id": "e1", "torsion_mode": "fixed_deg", "torsion_deg": 60.0}
                ]
            },
            "artifacts": {
                "coordinates": temp_path("overlap_corpus_graph_coords.pdb").to_string_lossy(),
                "build_manifest": temp_path("overlap_corpus_graph_manifest.json").to_string_lossy(),
                "charge_manifest": temp_path("overlap_corpus_graph_charge.json").to_string_lossy(),
                "topology_graph": temp_path("overlap_corpus_graph_topology.json").to_string_lossy()
            }
        }),
    ));

    println!(
        "{}",
        serde_json::to_string_pretty(&Value::Array(results.clone())).expect("serialize corpus")
    );

    for result in &results {
        let solver_cleanup = &result["solver_cleanup"];
        if !solver_cleanup.is_null() {
            assert_eq!(
                solver_cleanup["final_overlap_pairs"].as_u64(),
                Some(0),
                "{}",
                serde_json::to_string_pretty(result).unwrap()
            );
        }
        let relax = &result["relax"];
        if !relax.is_null() {
            assert_eq!(
                relax["final_overlap_pairs"].as_u64(),
                Some(0),
                "{}",
                serde_json::to_string_pretty(result).unwrap()
            );
        }
    }
}

#[test]
#[ignore = "manual overlap seed sweep"]
fn overlap_seed_sweep_surfaces_worst_random_walk_offenders() {
    let seed_range = 3000u64..3064u64;

    let (_dir, star_bundle) = make_bundle_dir("overlap_seed_sweep_star");
    let mut star_records = Vec::new();
    for seed in seed_range.clone() {
        star_records.push(run_overlap_sweep_case(
            "star_random_walk_dense",
            seed,
            overlap_seed_sweep_star_request(&star_bundle, seed),
        ));
    }

    let (_dir, branched_bundle) = make_bundle_dir("overlap_seed_sweep_branched");
    let mut branched_records = Vec::new();
    for seed in seed_range.clone() {
        branched_records.push(run_overlap_sweep_case(
            "branched_random_walk_dense",
            seed,
            overlap_seed_sweep_branched_request(&branched_bundle, seed),
        ));
    }

    let (_dir, graph_bundle) = make_bundle_dir("overlap_seed_sweep_graph");
    let mut graph_records = Vec::new();
    for seed in seed_range {
        graph_records.push(run_overlap_sweep_case(
            "graph_random_walk_dense",
            seed,
            overlap_seed_sweep_graph_request(&graph_bundle, seed),
        ));
    }

    let summaries = vec![
        overlap_sweep_summary("star_random_walk_dense", &star_records),
        overlap_sweep_summary("branched_random_walk_dense", &branched_records),
        overlap_sweep_summary("graph_random_walk_dense", &graph_records),
    ];
    println!(
        "{}",
        serde_json::to_string_pretty(&summaries).expect("serialize sweep")
    );

    for records in [&star_records, &branched_records, &graph_records] {
        for record in records {
            assert_overlap_sweep_record_is_clear(record);
        }
    }
}

#[test]
fn overlap_dense_star_seed_3012_clears_after_solver_cleanup_and_relax() {
    let (_dir, bundle) = make_bundle_dir("overlap_regression_star_3012");
    let record = run_overlap_sweep_case(
        "star_random_walk_dense",
        3012,
        overlap_seed_sweep_star_request(&bundle, 3012),
    );
    assert_overlap_sweep_record_is_clear(&record);
}

#[test]
fn overlap_dense_branched_seed_3001_clears_after_solver_cleanup_and_relax() {
    let (_dir, bundle) = make_bundle_dir("overlap_regression_branched_3001");
    let record = run_overlap_sweep_case(
        "branched_random_walk_dense",
        3001,
        overlap_seed_sweep_branched_request(&bundle, 3001),
    );
    assert_overlap_sweep_record_is_clear(&record);
}

#[test]
fn overlap_dense_graph_seed_3018_clears_after_solver_cleanup_and_relax() {
    let (_dir, bundle) = make_bundle_dir("overlap_regression_graph_3018");
    let record = run_overlap_sweep_case(
        "graph_random_walk_dense",
        3018,
        overlap_seed_sweep_graph_request(&bundle, 3018),
    );
    assert_overlap_sweep_record_is_clear(&record);
}

#[test]
fn pes_repro_build_keeps_inter_residue_bonds_below_three_angstrom() {
    let (_dir, bundle) = copy_fixture_dir("pes_repro", "fixture_pes_repro");
    let coords = temp_path("pes_coords.pdb");
    let raw_coords = temp_path("pes_coords_raw.pdb");
    let build_manifest = temp_path("pes_manifest.json");
    let charge_manifest = temp_path("pes_charge.json");
    let topology_graph = temp_path("pes_topology_graph.json");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "pes-build-8mer-001",
        "source_ref": {
            "bundle_id": "Polyethersulfone_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "linear_homopolymer",
            "repeat_unit": "A",
            "n_repeat": 8,
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"},
        },
        "realization": {
            "conformation_mode": "random_walk",
            "seed": 12345,
            "relax": {
                "mode": "graph_spring",
                "steps": 64,
                "step_scale": 0.25,
                "clash_scale": 0.9
            }
        },
        "artifacts": {
            "coordinates": coords.to_string_lossy(),
            "raw_coordinates": raw_coords.to_string_lossy(),
            "build_manifest": build_manifest.to_string_lossy(),
            "charge_manifest": charge_manifest.to_string_lossy(),
            "topology_graph": topology_graph.to_string_lossy(),
        }
    });
    let (code, payload) =
        warp_build::run_request_json(&serde_json::to_string(&request).expect("serialize"), false);
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    let manifest: Value =
        serde_json::from_str(&fs::read_to_string(&build_manifest).expect("read manifest"))
            .expect("parse manifest");
    assert_eq!(
        manifest["summary"]["resolved_sequence"],
        json!(["H", "A", "A", "A", "A", "A", "A", "T"])
    );
    assert_eq!(
        manifest["summary"]["qc"]["severe_bond_violations"],
        json!([])
    );
    let max_distance = max_inter_residue_bond_distance(&coords, &topology_graph);
    assert!(
        max_distance < 3.0,
        "expected PES inter-residue bond distances below 3.0 A, got {max_distance:.3}"
    );
}

#[test]
fn run_sequence_build_resolves_named_termini_tokens() {
    let (_dir, bundle) = make_bundle_dir("named_termini");
    let coords = temp_path("named_termini_coords.pdb");
    let build_manifest = temp_path("named_termini_manifest.json");
    let charge_manifest = temp_path("named_termini_charge.json");
    let topology = temp_path("named_termini_topology.prmtop");
    let topology_graph = temp_path("named_termini_topology_graph.json");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "named-termini-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "linear_sequence_polymer",
            "sequence": ["A", "B"],
            "repeat_count": 1,
            "termini": {"head": "H", "tail": "T"},
            "stereochemistry": {"mode": "inherit"},
        },
        "realization": {
            "conformation_mode": "extended",
            "seed": 9
        },
        "artifacts": {
            "coordinates": coords.to_string_lossy(),
            "build_manifest": build_manifest.to_string_lossy(),
            "charge_manifest": charge_manifest.to_string_lossy(),
            "topology": topology.to_string_lossy(),
            "topology_graph": topology_graph.to_string_lossy(),
        }
    });
    let (code, payload) =
        warp_build::run_request_json(&serde_json::to_string(&request).expect("serialize"), false);
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );

    let manifest: Value =
        serde_json::from_str(&fs::read_to_string(&build_manifest).expect("read manifest"))
            .expect("parse manifest");
    assert_eq!(manifest["summary"]["resolved_sequence"], json!(["H", "T"]));
    assert_eq!(
        manifest["summary"]["applied_termini"]["head"]["resolved_token"],
        "H"
    );
    assert_eq!(
        manifest["summary"]["applied_termini"]["tail"]["resolved_token"],
        "T"
    );
    assert_eq!(
        manifest["summary"]["applied_termini"]["head"]["template_resname"],
        "HDA"
    );
    assert_eq!(
        manifest["summary"]["applied_termini"]["tail"]["template_resname"],
        "TLA"
    );
    assert!(topology.exists());

    let coords_text = fs::read_to_string(&coords).expect("read generated coordinates");
    let residue_names = coords_text
        .lines()
        .filter(|line| line.starts_with("ATOM  ") || line.starts_with("HETATM"))
        .map(|line| line[17..20].trim().to_string())
        .collect::<Vec<_>>();
    assert_eq!(
        residue_names,
        vec!["HDA".to_string(), "TLA".to_string()],
        "generated PDB should keep source template residue names for forcefield matching"
    );
    assert!(
        !residue_names.iter().any(|name| name == "H" || name == "T"),
        "generated PDB must not use sequence token labels as residue names"
    );

    let graph: Value =
        serde_json::from_str(&fs::read_to_string(&topology_graph).expect("read graph"))
            .expect("parse graph");
    assert_eq!(graph["schema_version"], "warp-build.topology-graph.v5");
    assert_eq!(
        graph["build_plan"]["requested_termini"],
        json!({"head": "H", "tail": "T"})
    );
    assert_eq!(graph["sequence"], json!(["H", "T"]));
    assert_eq!(
        graph["connection_definitions"][0]["parent_attach_atom"],
        "C1"
    );
    assert_eq!(graph["branch_points"], json!([]));
}

#[test]
fn run_block_build_writes_topology_graph() {
    let (_dir, bundle) = make_bundle_dir("block_run");
    let coords = temp_path("block_coords.pdb");
    let build_manifest = temp_path("block_build_manifest.json");
    let charge_manifest = temp_path("block_charge_manifest.json");
    let topology = temp_path("block_topology.prmtop");
    let topology_graph = temp_path("block_topology_graph.json");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "block-build-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "block_copolymer",
            "blocks": [
                {"token": "A", "count": 2},
                {"token": "B", "count": 1}
            ],
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"},
        },
        "realization": {
            "conformation_mode": "aligned",
            "alignment_axis": "x",
            "seed": 321
        },
        "artifacts": {
            "coordinates": coords.to_string_lossy(),
            "build_manifest": build_manifest.to_string_lossy(),
            "charge_manifest": charge_manifest.to_string_lossy(),
            "topology": topology.to_string_lossy(),
            "topology_graph": topology_graph.to_string_lossy(),
        }
    });
    let (code, payload) =
        warp_build::run_request_json(&serde_json::to_string(&request).expect("serialize"), false);
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    assert!(topology_graph.exists());
    let graph: Value =
        serde_json::from_str(&fs::read_to_string(&topology_graph).expect("read graph"))
            .expect("parse graph");
    assert_eq!(graph["build_plan"]["target_mode"], "block_copolymer");
    assert_eq!(graph["build_plan"]["realization_mode"], "aligned");
    assert_eq!(
        graph["build_plan"]["resolved_sequence"],
        json!(["A", "A", "B"])
    );
    assert_eq!(graph["bonds"].as_array().map(|items| items.len()), Some(2));
    assert_eq!(graph["angles"].as_array().map(|items| items.len()), Some(1));
    assert_eq!(
        graph["dihedrals"].as_array().map(|items| items.len()),
        Some(0)
    );
}

#[test]
fn run_ensemble_build_writes_member_manifest() {
    let (_dir, bundle) = make_bundle_dir("ensemble_run");
    let coords = temp_path("ensemble_coords.pdb");
    let build_manifest = temp_path("ensemble_build_manifest.json");
    let charge_manifest = temp_path("ensemble_charge_manifest.json");
    let topology = temp_path("ensemble_topology.prmtop");
    let topology_graph = temp_path("ensemble_topology_graph.json");
    let ensemble_manifest = temp_path("ensemble_manifest.json");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "ensemble-build-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "linear_sequence_polymer",
            "sequence": ["A", "B"],
            "repeat_count": 2,
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"},
        },
        "realization": {
            "conformation_mode": "ensemble",
            "ensemble_size": 3,
            "seed": 77
        },
        "artifacts": {
            "coordinates": coords.to_string_lossy(),
            "build_manifest": build_manifest.to_string_lossy(),
            "charge_manifest": charge_manifest.to_string_lossy(),
            "topology": topology.to_string_lossy(),
            "topology_graph": topology_graph.to_string_lossy(),
            "ensemble_manifest": ensemble_manifest.to_string_lossy(),
        }
    });
    let (code, payload) =
        warp_build::run_request_json(&serde_json::to_string(&request).expect("serialize"), false);
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    let ensemble: Value =
        serde_json::from_str(&fs::read_to_string(&ensemble_manifest).expect("read ensemble"))
            .expect("parse ensemble");
    assert_eq!(ensemble["member_count"], 3);
    assert_eq!(
        ensemble["shared_artifacts"]["topology_graph"],
        topology_graph.to_string_lossy().to_string()
    );
    assert_eq!(
        ensemble["shared_artifacts"]["topology"],
        topology.to_string_lossy().to_string()
    );
    assert_eq!(
        ensemble["shared_artifacts"]["charge_manifest"],
        charge_manifest.to_string_lossy().to_string()
    );
    assert_eq!(
        ensemble["member_paths"].as_array().map(|items| items.len()),
        Some(3)
    );
}

#[test]
fn validate_accepts_branched_aligned_realization() {
    let (_dir, bundle) = make_bundle_dir("branched_validate");
    let coords = temp_path("branched_validate_coords.pdb");
    let build_manifest = temp_path("branched_validate_manifest.json");
    let charge_manifest = temp_path("branched_validate_charge.json");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "branched-validate-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "branched_polymer",
            "branch_tree": {
                "token": "A",
                "children": [
                    {
                        "parent_junction": "head",
                        "child_junction": "head",
                        "sequence": ["B"],
                        "repeat_count": 1
                    }
                ]
            },
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "aligned",
            "alignment_axis": "z"
        },
        "artifacts": {
            "coordinates": coords.to_string_lossy(),
            "build_manifest": build_manifest.to_string_lossy(),
            "charge_manifest": charge_manifest.to_string_lossy(),
        }
    });
    let (code, payload) =
        warp_build::validate_request_json(&serde_json::to_string(&request).expect("serialize"));
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    assert_eq!(payload["schema_version"], "warp-build.agent.v1");
    assert_eq!(
        payload["resolved_inputs"]["seed_policy"],
        "deterministic_default"
    );
    assert_eq!(
        payload["resolved_inputs"]["resolved_termini_policy"],
        json!({"head": "source_default", "tail": "source_default"})
    );
    assert_eq!(
        payload["resolved_inputs"]["validation"]["requested_depth"],
        "deep"
    );
    assert_eq!(payload["preflight"]["executed"], true);
    assert_eq!(
        payload["preflight"]["qc"]["sequence_token_template_consistent"],
        true
    );
    assert!(payload["preflight"]["timings_ms"]["build_graph"]
        .as_u64()
        .is_some());
    assert_eq!(
        payload["preflight"]["overlap_status"]["report_source"],
        "solver_cleanup"
    );
    assert_eq!(
        payload["normalized_request"]["artifacts"]["inpcrd"],
        coords
            .parent()
            .expect("coords parent")
            .join(format!(
                "{}.inpcrd",
                coords
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .expect("coords stem")
            ))
            .to_string_lossy()
            .to_string()
    );
    assert_eq!(
        payload["normalized_request"]["artifacts"]["forcefield_ref"],
        Value::Null
    );
    let warnings = payload["warnings"].as_array().expect("warnings array");
    assert!(!warnings.is_empty());
    assert_eq!(warnings[0]["severity"], "warning");
    assert_eq!(warnings[0]["path"], "/target/termini/head");
}

#[test]
fn validate_deep_runs_geometry_preflight() {
    let (_dir, bundle) = make_bundle_dir("branched_validate_deep");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "branched-validate-deep-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "branched_polymer",
            "branch_tree": {
                "token": "A",
                "children": [
                    {
                        "parent_junction": "head",
                        "child_junction": "head",
                        "sequence": ["B"],
                        "repeat_count": 1
                    }
                ]
            },
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "aligned",
            "alignment_axis": "z"
        },
        "validation": {
            "depth": "deep"
        },
        "artifacts": {
            "coordinates": temp_path("branched_validate_deep_coords.pdb").to_string_lossy(),
            "build_manifest": temp_path("branched_validate_deep_manifest.json").to_string_lossy(),
            "charge_manifest": temp_path("branched_validate_deep_charge.json").to_string_lossy(),
        }
    });
    let (code, payload) =
        warp_build::validate_request_json(&serde_json::to_string(&request).expect("serialize"));
    assert_eq!(code, 0);
    assert_eq!(
        payload["resolved_inputs"]["validation"]["requested_depth"],
        "deep"
    );
    assert_eq!(payload["preflight"]["executed"], true);
    assert_eq!(
        payload["preflight"]["qc"]["sequence_token_template_consistent"],
        true
    );
    assert_eq!(
        payload["preflight"]["overlap_status"]["report_source"],
        "solver_cleanup"
    );
    assert!(payload["preflight"]["overlap_status"]["overlap_pairs"]
        .as_u64()
        .is_some());
    assert!(payload["preflight"]["timings_ms"]["build_graph"]
        .as_u64()
        .is_some());
}

#[test]
fn run_star_and_branched_builds_write_branch_metadata() {
    let (_dir, bundle) = make_bundle_dir("branch_run");
    let star_coords = temp_path("star_coords.pdb");
    let star_manifest = temp_path("star_manifest.json");
    let star_charge = temp_path("star_charge.json");
    let star_topology = temp_path("star_topology.prmtop");
    let star_graph = temp_path("star_graph.json");
    let star_request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "star-build-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "star_polymer",
            "core_token": "A",
            "core_junctions": ["head", "tail"],
            "arm_sequence": ["B", "A"],
            "arm_repeat_count": 2,
            "termini": {"head": "H", "tail": "T"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "aligned",
            "alignment_axis": "z",
            "seed": 77
        },
        "artifacts": {
            "coordinates": star_coords.to_string_lossy(),
            "build_manifest": star_manifest.to_string_lossy(),
            "charge_manifest": star_charge.to_string_lossy(),
            "topology": star_topology.to_string_lossy(),
            "topology_graph": star_graph.to_string_lossy(),
        }
    });
    let (code, payload) = warp_build::run_request_json(
        &serde_json::to_string(&star_request).expect("serialize"),
        false,
    );
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    let star_graph_payload: Value =
        serde_json::from_str(&fs::read_to_string(&star_graph).expect("read graph")).expect("graph");
    assert_eq!(
        star_graph_payload["build_plan"]["target_mode"],
        "star_polymer"
    );
    assert_eq!(star_graph_payload["build_plan"]["root_token"], "A");
    assert_eq!(star_graph_payload["build_plan"]["arm_count"], 2);
    assert_eq!(star_graph_payload["build_plan"]["max_branch_depth"], 5);
    assert!(star_topology.exists());
    assert!(star_graph_payload["connection_definitions"]
        .as_array()
        .map(|items| !items.is_empty())
        .unwrap_or(false));
    assert!(star_graph_payload["sequence"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item == "T"));

    let branched_coords = temp_path("branched_coords.pdb");
    let branched_manifest = temp_path("branched_manifest.json");
    let branched_charge = temp_path("branched_charge.json");
    let branched_topology = temp_path("branched_topology.prmtop");
    let branched_graph = temp_path("branched_graph.json");
    let branched_request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "branched-build-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "branched_polymer",
            "branch_tree": {
                "token": "A",
                "children": [
                    {
                        "parent_junction": "head",
                        "child_junction": "head",
                        "sequence": ["B", "A"],
                        "repeat_count": 1,
                        "child": {
                            "token": "B",
                            "children": [
                                {
                                    "parent_junction": "head",
                                    "child_junction": "head",
                                    "sequence": ["A"],
                                    "repeat_count": 1
                                }
                            ]
                        }
                    }
                ]
            },
            "termini": {"head": "H", "tail": "T"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "aligned",
            "alignment_axis": "z",
            "seed": 88
        },
        "artifacts": {
            "coordinates": branched_coords.to_string_lossy(),
            "build_manifest": branched_manifest.to_string_lossy(),
            "charge_manifest": branched_charge.to_string_lossy(),
            "topology": branched_topology.to_string_lossy(),
            "topology_graph": branched_graph.to_string_lossy(),
        }
    });
    let (code, payload) = warp_build::run_request_json(
        &serde_json::to_string(&branched_request).expect("serialize"),
        false,
    );
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    let branched_graph_payload: Value =
        serde_json::from_str(&fs::read_to_string(&branched_graph).expect("read graph"))
            .expect("graph");
    assert_eq!(
        branched_graph_payload["build_plan"]["target_mode"],
        "branched_polymer"
    );
    assert_eq!(branched_graph_payload["build_plan"]["max_branch_depth"], 5);
    assert!(branched_topology.exists());
    assert_eq!(branched_graph_payload["residues"][0]["branch_depth"], 0);
    assert_eq!(branched_graph_payload["residues"][1]["branch_depth"], 1);
    assert!(branched_graph_payload["residues"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["branch_depth"] == 5));
    assert!(branched_graph_payload["sequence"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item == "T"));
}

#[test]
fn run_multiport_star_build_uses_all_core_junctions_without_clashes() {
    let (_dir, bundle) = make_multiport_star_bundle_dir("multiport_star_run");
    let coords = temp_path("multiport_star_coords.pdb");
    let manifest = temp_path("multiport_star_manifest.json");
    let charge = temp_path("multiport_star_charge.json");
    let topology = temp_path("multiport_star_topology.prmtop");
    let graph = temp_path("multiport_star_graph.json");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "multiport-star-build-001",
        "source_ref": {
            "bundle_id": "multiport_star_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "star_polymer",
            "core_token": "C",
            "core_junctions": ["j1", "j2", "j3", "j4"],
            "arm_sequence": ["A"],
            "arm_repeat_count": 8,
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "random_walk",
            "seed": 20260424
        },
        "artifacts": {
            "coordinates": coords.to_string_lossy(),
            "build_manifest": manifest.to_string_lossy(),
            "charge_manifest": charge.to_string_lossy(),
            "topology": topology.to_string_lossy(),
            "topology_graph": graph.to_string_lossy()
        }
    });
    let (code, payload) =
        warp_build::run_request_json(&serde_json::to_string(&request).expect("serialize"), false);
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    assert_eq!(payload["summary"]["acceptance_state"], "accepted");
    assert_eq!(payload["summary"]["handoff_level"], "minimizable_synthetic");
    assert_eq!(payload["summary"]["qc"]["severe_nonbonded_clash_count"], 0);
    assert!(
        payload["summary"]["qc"]["min_nonbonded_distance_angstrom"]
            .as_f64()
            .unwrap_or(0.0)
            >= 1.2
    );

    let graph_payload: Value =
        serde_json::from_str(&fs::read_to_string(&graph).expect("read graph")).expect("graph");
    assert_eq!(graph_payload["build_plan"]["target_mode"], "star_polymer");
    assert_eq!(graph_payload["build_plan"]["root_token"], "C");
    assert_eq!(graph_payload["build_plan"]["arm_count"], 4);
    assert!(graph_payload["build_plan"]["max_branch_depth"]
        .as_u64()
        .map(|value| value >= 8)
        .unwrap_or(false));
    let mut core_ports = graph_payload["connection_definitions"]
        .as_array()
        .expect("connection definitions")
        .iter()
        .filter(|item| item["parent_resid"] == 1 && item["child_port"] == "head")
        .filter_map(|item| item["parent_port"].as_str())
        .collect::<Vec<_>>();
    core_ports.sort_unstable();
    assert_eq!(core_ports, vec!["j1", "j2", "j3", "j4"]);

    let topo = read_prmtop_topology(&topology).expect("read multiport star prmtop");
    assert_eq!(topo.atom_names.len(), 36);
    assert_eq!(topo.bonds.len(), 35);
}

#[test]
fn run_polymer_graph_build_rejects_unresolved_cycle_geometry() {
    let (_dir, bundle) = make_bundle_dir("graph_run");
    let coords = temp_path("graph_coords.pdb");
    let manifest = temp_path("graph_manifest.json");
    let charge = temp_path("graph_charge.json");
    let topology = temp_path("graph_topology.prmtop");
    let graph = temp_path("graph_topology.json");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "graph-build-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "polymer_graph",
            "graph_root": "n1",
            "graph_nodes": [
                {"id": "n1", "token": "A"},
                {"id": "n2", "token": "B"},
                {"id": "n3", "token": "A"}
            ],
            "graph_edges": [
                {"from": "n1", "to": "n2", "from_junction": "head", "to_junction": "head"},
                {"from": "n2", "to": "n3", "from_junction": "tail", "to_junction": "head"},
                {"from": "n3", "to": "n1", "from_junction": "tail", "to_junction": "tail"}
            ],
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "random_walk",
            "seed": 91
        },
        "artifacts": {
            "coordinates": coords.to_string_lossy(),
            "build_manifest": manifest.to_string_lossy(),
            "charge_manifest": charge.to_string_lossy(),
            "topology": topology.to_string_lossy(),
            "topology_graph": graph.to_string_lossy(),
        }
    });
    let (code, payload) =
        warp_build::run_request_json(&serde_json::to_string(&request).expect("serialize"), false);
    assert_eq!(
        code,
        4,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    assert_eq!(payload["errors"][0]["code"], "E_RUNTIME_BUILD");
    assert!(payload["errors"][0]["message"]
        .as_str()
        .expect("error message")
        .contains("build QC failed"));
    assert!(payload["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["message"]
            .as_str()
            .unwrap_or("")
            .contains("_built_solute.pdb")));
    assert!(!coords.exists());
    assert!(!manifest.exists());
    assert!(!charge.exists());
    assert!(!topology.exists());
    assert!(!graph.exists());
}

#[test]
fn run_graph_build_qc_failure_attempts_relax_and_writes_raw_coordinates() {
    let (_dir, bundle) = make_bundle_dir("graph_qc_raw_failure");
    let coords = temp_path("graph_qc_raw_failure_coords.pdb");
    let raw_coords = temp_path("graph_qc_raw_failure.raw.pdb");
    let manifest = temp_path("graph_qc_raw_failure_manifest.json");
    let charge = temp_path("graph_qc_raw_failure_charge.json");
    let topology = temp_path("graph_qc_raw_failure.prmtop");
    let graph = temp_path("graph_qc_raw_failure_graph.json");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "graph-qc-raw-failure-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "polymer_graph",
            "graph_root": "n1",
            "graph_nodes": [
                {"id": "n1", "token": "A"},
                {"id": "n2", "token": "B"},
                {"id": "n3", "token": "A"}
            ],
            "graph_edges": [
                {"from": "n1", "to": "n2", "from_junction": "head", "to_junction": "head"},
                {"from": "n2", "to": "n3", "from_junction": "tail", "to_junction": "head"},
                {"from": "n3", "to": "n1", "from_junction": "tail", "to_junction": "tail"}
            ],
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "extended",
            "seed": 91,
            "qc_policy": "salvage",
            "relax": {
                "mode": "graph_spring",
                "steps": 8,
                "step_scale": 0.2,
                "clash_scale": 0.9
            }
        },
        "artifacts": {
            "coordinates": coords.to_string_lossy(),
            "raw_coordinates": raw_coords.to_string_lossy(),
            "build_manifest": manifest.to_string_lossy(),
            "charge_manifest": charge.to_string_lossy(),
            "topology": topology.to_string_lossy(),
            "topology_graph": graph.to_string_lossy()
        }
    });
    let (code, payload) =
        warp_build::run_request_json(&serde_json::to_string(&request).expect("serialize"), false);
    assert_eq!(
        code,
        3,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    assert_eq!(payload["status"], "salvaged");
    assert_eq!(payload["summary"]["acceptance_state"], "salvaged");
    assert_eq!(payload["summary"]["relax"]["mode"], "graph_spring");
    assert_eq!(
        payload["summary"]["relax"]["raw_coordinates"],
        raw_coords.to_string_lossy().to_string()
    );
    assert_eq!(payload["summary"]["qc"]["severe_nonbonded_clash_count"], 0);
    assert!(
        payload["summary"]["qc"]["severe_bond_violations"]
            .as_array()
            .expect("severe bond violations")
            .len()
            > 0
    );
    assert!(raw_coords.exists());
    assert!(fs::read_to_string(&raw_coords)
        .expect("read raw coordinates")
        .contains("ATOM"));
    assert!(coords.exists());
    assert!(manifest.exists());
    assert!(charge.exists());
    assert!(topology.exists());
    assert!(graph.exists());
}

#[test]
fn validate_preflight_rejects_qc_failing_graph_build() {
    let (_dir, bundle) = make_bundle_dir("validate_qc_fail");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "graph-qc-preflight-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "polymer_graph",
            "graph_root": "n1",
            "graph_nodes": [
                {"id": "n1", "token": "A"},
                {"id": "n2", "token": "B"},
                {"id": "n3", "token": "A"}
            ],
            "graph_edges": [
                {"from": "n1", "to": "n2", "from_junction": "head", "to_junction": "head"},
                {"from": "n2", "to": "n3", "from_junction": "tail", "to_junction": "head"},
                {"from": "n3", "to": "n1", "from_junction": "tail", "to_junction": "tail"}
            ],
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "random_walk",
            "seed": 91
        },
        "validation": {
            "depth": "deep"
        },
        "artifacts": {
            "coordinates": temp_path("preflight_qc_fail_coords.pdb").to_string_lossy(),
            "build_manifest": temp_path("preflight_qc_fail_manifest.json").to_string_lossy(),
            "charge_manifest": temp_path("preflight_qc_fail_charge.json").to_string_lossy()
        }
    });
    let (code, payload) =
        warp_build::validate_request_json(&serde_json::to_string(&request).expect("serialize"));
    assert_eq!(code, 2);
    assert!(payload["errors"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["code"] == "E_SOURCE_GEOMETRY"));
    assert!(payload["errors"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["message"]
            .as_str()
            .unwrap_or("")
            .contains("preflight QC failed")));
}

#[test]
fn run_structure_only_bundle_writes_minimizable_synthetic_handoff() {
    let (_dir, bundle) = make_structure_only_bundle_dir("structure_only_run");
    let coords = temp_path("structure_only_coords.pdb");
    let manifest = temp_path("structure_only_manifest.json");
    let charge = temp_path("structure_only_charge.json");
    let graph = temp_path("structure_only_graph.json");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "structure-only-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "linear_homopolymer",
            "repeat_unit": "A",
            "n_repeat": 4,
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "extended"
        },
        "artifacts": {
            "coordinates": coords.to_string_lossy(),
            "build_manifest": manifest.to_string_lossy(),
            "charge_manifest": charge.to_string_lossy(),
            "topology_graph": graph.to_string_lossy()
        }
    });
    let (code, payload) =
        warp_build::run_request_json(&serde_json::to_string(&request).expect("serialize"), false);
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    assert_eq!(payload["summary"]["acceptance_state"], "accepted");
    assert_eq!(payload["summary"]["handoff_level"], "minimizable_synthetic");
    assert_eq!(
        payload["summary"]["limitations"],
        json!(["synthetic_topology", "charge_unavailable"])
    );
    let topology_path = payload["artifacts"]["topology"]
        .as_str()
        .expect("synthetic topology path");
    assert!(coords.exists());
    assert!(manifest.exists());
    assert!(charge.exists());
    assert!(graph.exists());
    assert!(Path::new(topology_path).exists());
    assert_eq!(payload["artifacts"]["forcefield_ref"], Value::Null);
    let manifest_value: Value =
        serde_json::from_str(&fs::read_to_string(&manifest).expect("read manifest"))
            .expect("parse manifest");
    assert_eq!(
        manifest_value["summary"]["handoff_level"],
        "minimizable_synthetic"
    );
    assert_eq!(
        manifest_value["artifacts"]["topology"],
        Value::String(topology_path.to_string())
    );
    assert_eq!(
        manifest_value["md_ready_handoff"]["topology"],
        Value::String(topology_path.to_string())
    );
    assert_eq!(
        manifest_value["md_ready_handoff"]["forcefield_ref"],
        Value::Null
    );
    assert_eq!(
        manifest_value["summary"]["limitations"],
        json!(["synthetic_topology", "charge_unavailable"])
    );
    let charge_value: Value =
        serde_json::from_str(&fs::read_to_string(&charge).expect("read charge manifest"))
            .expect("parse charge manifest");
    assert_eq!(charge_value["net_charge_e"], Value::Null);
    assert_eq!(charge_value["charge_derivation"], "unavailable");
    assert_eq!(
        charge_value["target_topology_ref"],
        Value::String(topology_path.to_string())
    );
}

#[test]
fn synthetic_structure_only_topology_writes_explicit_bonded_terms() {
    let (_dir, bundle) = make_structure_only_bundle_dir("structure_only_synthetic_terms");
    let coords = temp_path("synthetic_terms_coords.pdb");
    let manifest = temp_path("synthetic_terms_manifest.json");
    let charge = temp_path("synthetic_terms_charge.json");
    let graph = temp_path("synthetic_terms_graph.json");
    let topology = temp_path("synthetic_terms_topology.prmtop");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "structure-only-terms-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "linear_homopolymer",
            "repeat_unit": "A",
            "n_repeat": 4,
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "extended"
        },
        "artifacts": {
            "coordinates": coords.to_string_lossy(),
            "build_manifest": manifest.to_string_lossy(),
            "charge_manifest": charge.to_string_lossy(),
            "topology": topology.to_string_lossy(),
            "topology_graph": graph.to_string_lossy()
        }
    });
    let (code, payload) =
        warp_build::run_request_json(&serde_json::to_string(&request).expect("serialize"), false);
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    let topo = read_prmtop_topology(&topology).expect("read synthetic prmtop");
    assert!(topo.atom_names.len() >= 4);
    assert_eq!(topo.bonds.len() + 1, topo.atom_names.len());
    assert!(!topo.angles.is_empty());
    assert!(!topo.dihedrals.is_empty());
    assert!(topo.bond_force_constants.iter().all(|value| *value > 0.0));
    assert!(topo.angle_force_constants.iter().all(|value| *value > 0.0));
    assert!(topo
        .angle_equil_values
        .iter()
        .all(|value| *value > 0.0 && *value <= std::f32::consts::PI + 1.0e-5));
    assert!(topo
        .dihedral_phases
        .iter()
        .all(|value| *value >= 0.0 && *value <= std::f32::consts::PI));
    assert!(topo
        .masses
        .iter()
        .all(|value| (*value - 12.011).abs() < 0.05));
}

#[test]
fn validate_rejects_unreliable_training_without_strong_source() {
    let (_dir, bundle) = make_bad_training_bundle_dir("bad_training_reject", false, false);
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "bad-training-reject-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "linear_homopolymer",
            "repeat_unit": "A",
            "n_repeat": 4,
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "extended"
        },
        "artifacts": {
            "coordinates": temp_path("bad_training_reject_coords.pdb").to_string_lossy(),
            "build_manifest": temp_path("bad_training_reject_manifest.json").to_string_lossy(),
            "charge_manifest": temp_path("bad_training_reject_charge.json").to_string_lossy()
        }
    });
    let (code, payload) =
        warp_build::validate_request_json(&serde_json::to_string(&request).expect("serialize"));
    assert_eq!(
        code,
        2,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    assert!(payload["errors"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["code"] == "E_SOURCE_UNRELIABLE"));
}

#[test]
fn validate_unreliable_training_prefers_source_topology_fallback() {
    let (_dir, bundle) = make_bad_training_bundle_dir("bad_training_prmtop", true, true);
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "bad-training-prmtop-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "linear_homopolymer",
            "repeat_unit": "A",
            "n_repeat": 4,
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "extended"
        },
        "artifacts": {
            "coordinates": temp_path("bad_training_prmtop_coords.pdb").to_string_lossy(),
            "build_manifest": temp_path("bad_training_prmtop_manifest.json").to_string_lossy(),
            "charge_manifest": temp_path("bad_training_prmtop_charge.json").to_string_lossy()
        }
    });
    let (code, payload) =
        warp_build::validate_request_json(&serde_json::to_string(&request).expect("serialize"));
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    assert_eq!(
        payload["preflight"]["parameter_source_decision"]["quality"],
        "unreliable"
    );
    assert_eq!(
        payload["preflight"]["parameter_source_decision"]["parameter_source"],
        "source_topology_ref"
    );
}

#[test]
fn clean_random_walk_synthetic_handoff_is_minimizable() {
    let (_dir, bundle) = make_structure_only_bundle_dir("structure_only_random_walk_handoff");
    let coords = temp_path("structure_only_random_walk_coords.pdb");
    let manifest = temp_path("structure_only_random_walk_manifest.json");
    let charge = temp_path("structure_only_random_walk_charge.json");
    let graph = temp_path("structure_only_random_walk_graph.json");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "structure-only-random-walk-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "linear_homopolymer",
            "repeat_unit": "A",
            "n_repeat": 4,
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "random_walk",
            "seed": 12345
        },
        "artifacts": {
            "coordinates": coords.to_string_lossy(),
            "build_manifest": manifest.to_string_lossy(),
            "charge_manifest": charge.to_string_lossy(),
            "topology_graph": graph.to_string_lossy()
        }
    });
    let (code, payload) =
        warp_build::run_request_json(&serde_json::to_string(&request).expect("serialize"), false);
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    assert_eq!(payload["summary"]["acceptance_state"], "accepted");
    assert_eq!(payload["summary"]["handoff_level"], "minimizable_synthetic");
    assert!(!payload["summary"]["limitations"]
        .as_array()
        .expect("limitations")
        .iter()
        .any(|item| item == "conformer_not_production_minimized"));
    assert!(payload["summary"]["qc"]["severe_bond_violations"]
        .as_array()
        .expect("severe bond violations")
        .is_empty());
    assert_eq!(payload["summary"]["qc"]["severe_nonbonded_clash_count"], 0);
    assert!(
        payload["summary"]["qc"]["min_nonbonded_distance_angstrom"]
            .as_f64()
            .unwrap_or(0.0)
            >= 1.2
    );
    let manifest_value: Value =
        serde_json::from_str(&fs::read_to_string(&manifest).expect("read manifest"))
            .expect("parse manifest");
    assert_eq!(
        manifest_value["summary"]["handoff_level"],
        "minimizable_synthetic"
    );
}

#[test]
fn validate_records_preflight_cache_and_run_requires_cache() {
    let (_dir, bundle) = make_structure_only_bundle_dir("preflight_cache_bundle");
    let cache_dir = temp_path("preflight_cache_store");
    let original_topology = temp_path("preflight_cache_original_topology.prmtop");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "preflight-cache-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "linear_homopolymer",
            "repeat_unit": "A",
            "n_repeat": 4,
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "extended"
        },
        "validation": {
            "depth": "deep",
            "cache_mode": "record",
            "cache_dir": cache_dir.to_string_lossy()
        },
        "artifacts": {
            "coordinates": temp_path("preflight_cache_original_coords.pdb").to_string_lossy(),
            "build_manifest": temp_path("preflight_cache_original_manifest.json").to_string_lossy(),
            "charge_manifest": temp_path("preflight_cache_original_charge.json").to_string_lossy(),
            "topology": original_topology.to_string_lossy(),
            "topology_graph": temp_path("preflight_cache_original_graph.json").to_string_lossy()
        }
    });
    let (validate_code, validate_payload) =
        warp_build::validate_request_json(&serde_json::to_string(&request).expect("serialize"));
    assert_eq!(
        validate_code,
        0,
        "{}",
        serde_json::to_string_pretty(&validate_payload).unwrap()
    );
    assert_eq!(validate_payload["preflight_cache"]["state"], "written");
    let record_path = validate_payload["preflight_cache"]["record_path"]
        .as_str()
        .expect("record path");
    assert!(Path::new(record_path).exists());

    let mut relocated_request = request.clone();
    relocated_request["validation"] = json!({
        "depth": "deep",
        "cache_mode": "off"
    });
    relocated_request["artifacts"] = json!({
        "coordinates": temp_path("preflight_cache_relocated_coords.pdb").to_string_lossy(),
        "build_manifest": temp_path("preflight_cache_relocated_manifest.json").to_string_lossy(),
        "charge_manifest": temp_path("preflight_cache_relocated_charge.json").to_string_lossy(),
        "topology": temp_path("preflight_cache_relocated_topology.prmtop").to_string_lossy(),
        "topology_graph": temp_path("preflight_cache_relocated_graph.json").to_string_lossy()
    });
    let (relocated_code, relocated_payload) = warp_build::validate_request_json(
        &serde_json::to_string(&relocated_request).expect("serialize"),
    );
    assert_eq!(
        relocated_code,
        0,
        "{}",
        serde_json::to_string_pretty(&relocated_payload).unwrap()
    );
    assert_eq!(
        relocated_payload["preflight_cache"]["input_digest"],
        validate_payload["preflight_cache"]["input_digest"]
    );
    assert_eq!(
        relocated_payload["preflight_cache"]["cache_key"],
        validate_payload["preflight_cache"]["cache_key"]
    );
    assert_ne!(
        relocated_payload["preflight_cache"]["request_digest"],
        validate_payload["preflight_cache"]["request_digest"]
    );

    let mut changed_input_request = relocated_request.clone();
    changed_input_request["target"]["n_repeat"] = json!(5);
    let (changed_input_code, changed_input_payload) = warp_build::validate_request_json(
        &serde_json::to_string(&changed_input_request).expect("serialize"),
    );
    assert_eq!(
        changed_input_code,
        0,
        "{}",
        serde_json::to_string_pretty(&changed_input_payload).unwrap()
    );
    assert_ne!(
        changed_input_payload["preflight_cache"]["input_digest"],
        validate_payload["preflight_cache"]["input_digest"]
    );

    let final_coords = temp_path("preflight_cache_final_coords.pdb");
    let final_manifest = temp_path("preflight_cache_final_manifest.json");
    let final_charge = temp_path("preflight_cache_final_charge.json");
    let final_topology = temp_path("preflight_cache_final_topology.prmtop");
    let final_graph = temp_path("preflight_cache_final_graph.json");
    let mut run_request = request.clone();
    run_request["validation"]["cache_mode"] = json!("require");
    run_request["artifacts"] = json!({
        "coordinates": final_coords.to_string_lossy(),
        "build_manifest": final_manifest.to_string_lossy(),
        "charge_manifest": final_charge.to_string_lossy(),
        "topology": final_topology.to_string_lossy(),
        "topology_graph": final_graph.to_string_lossy()
    });
    let (run_code, run_payload) = warp_build::run_request_json(
        &serde_json::to_string(&run_request).expect("serialize"),
        false,
    );
    assert_eq!(
        run_code,
        0,
        "{}",
        serde_json::to_string_pretty(&run_payload).unwrap()
    );
    assert!(run_payload["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["code"] == "W_PREFLIGHT_CACHE_HIT"));
    assert!(final_coords.exists());
    assert!(final_manifest.exists());
    assert!(final_charge.exists());
    assert!(final_topology.exists());
    assert!(final_graph.exists());
    let manifest_value: Value =
        serde_json::from_str(&fs::read_to_string(&final_manifest).expect("read manifest"))
            .expect("parse manifest");
    assert_eq!(
        manifest_value["artifacts"]["coordinates"],
        final_coords.to_string_lossy().as_ref()
    );
    assert!(manifest_value["artifact_digests"]["coordinates"]
        .as_str()
        .unwrap()
        .starts_with("sha256:"));
}

#[test]
fn run_unreliable_training_uses_forcefield_fallback() {
    let (_dir, bundle) = make_bad_training_bundle_dir("bad_training_ffxml", false, true);
    let coords = temp_path("bad_training_ffxml_coords.pdb");
    let manifest = temp_path("bad_training_ffxml_manifest.json");
    let charge = temp_path("bad_training_ffxml_charge.json");
    let graph = temp_path("bad_training_ffxml_graph.json");
    let topology = temp_path("bad_training_ffxml_topology.prmtop");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "bad-training-ffxml-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "linear_homopolymer",
            "repeat_unit": "A",
            "n_repeat": 4,
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "extended"
        },
        "artifacts": {
            "coordinates": coords.to_string_lossy(),
            "build_manifest": manifest.to_string_lossy(),
            "charge_manifest": charge.to_string_lossy(),
            "topology": topology.to_string_lossy(),
            "topology_graph": graph.to_string_lossy()
        }
    });
    let (code, payload) =
        warp_build::run_request_json(&serde_json::to_string(&request).expect("serialize"), false);
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    assert_eq!(payload["summary"]["handoff_level"], "forcefield_backed");
    assert_eq!(
        payload["summary"]["parameter_source_decision"]["parameter_source"],
        "forcefield_ref"
    );
    assert_eq!(
        payload["summary"]["parameter_source_decision"]["quality"],
        "unreliable"
    );
    assert!(payload["summary"]["limitations"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item == "training_source_unreliable"));
    let topo = read_prmtop_topology(&topology).expect("read ffxml-backed prmtop");
    assert_eq!(topo.atom_names.len(), 24);
    assert!(topo.atomic_numbers.iter().any(|value| *value == 1));
    assert!(topo.atomic_numbers.iter().any(|value| *value == 17));
    assert!(!topo.bonds.is_empty());
    assert!(!topo.angles.is_empty());
    assert!(!topo.dihedrals.is_empty());
    assert!(!topo.bond_force_constants.is_empty());
    assert!(!topo.angle_force_constants.is_empty());
    assert!(!topo.dihedral_force_constants.is_empty());
    assert!(topo.charges.iter().any(|charge| charge.abs() > 1.0e-3));
    assert!(topo.charges.iter().sum::<f32>().abs() < 1.0e-3);
    let manifest_value: Value =
        serde_json::from_str(&fs::read_to_string(&manifest).expect("read manifest"))
            .expect("manifest json");
    assert_eq!(
        manifest_value["summary"]["parameter_source_decision"]["parameter_source"],
        "forcefield_ref"
    );
    assert_eq!(
        manifest_value["summary"]["handoff_level"],
        "forcefield_backed"
    );
}

#[test]
fn run_qc_salvage_mode_writes_non_final_outputs() {
    let (_dir, bundle) = make_bundle_dir("graph_qc_salvage");
    let coords = temp_path("graph_qc_salvage_coords.pdb");
    let manifest = temp_path("graph_qc_salvage_manifest.json");
    let charge = temp_path("graph_qc_salvage_charge.json");
    let topology = temp_path("graph_qc_salvage.prmtop");
    let graph = temp_path("graph_qc_salvage_graph.json");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "graph-qc-salvage-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "polymer_graph",
            "graph_root": "n1",
            "graph_nodes": [
                {"id": "n1", "token": "A"},
                {"id": "n2", "token": "B"},
                {"id": "n3", "token": "A"}
            ],
            "graph_edges": [
                {"from": "n1", "to": "n2", "from_junction": "head", "to_junction": "head"},
                {"from": "n2", "to": "n3", "from_junction": "tail", "to_junction": "head"},
                {"from": "n3", "to": "n1", "from_junction": "tail", "to_junction": "tail"}
            ],
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "random_walk",
            "seed": 91,
            "qc_policy": "salvage"
        },
        "artifacts": {
            "coordinates": coords.to_string_lossy(),
            "build_manifest": manifest.to_string_lossy(),
            "charge_manifest": charge.to_string_lossy(),
            "topology": topology.to_string_lossy(),
            "topology_graph": graph.to_string_lossy()
        }
    });
    let (code, payload) =
        warp_build::run_request_json(&serde_json::to_string(&request).expect("serialize"), false);
    assert_eq!(
        code,
        3,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    assert_eq!(payload["status"], "salvaged");
    assert_eq!(payload["summary"]["acceptance_state"], "salvaged");
    assert_eq!(payload["summary"]["handoff_level"], "graph_bonded_only");
    assert!(payload["summary"]["limitations"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item == "salvaged_qc_failure"));
    assert!(payload["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["code"] == "W_SALVAGED_BUILD"));
    assert!(coords.exists());
    assert!(manifest.exists());
    assert!(charge.exists());
    assert!(graph.exists());
    assert!(topology.exists());
}

#[test]
fn validate_rejects_unknown_conformer_edge_override() {
    let (_dir, bundle) = make_bundle_dir("graph_override_validate");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "graph-override-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "polymer_graph",
            "graph_root": "n1",
            "graph_nodes": [
                {"id": "n1", "token": "A"},
                {"id": "n2", "token": "M2"}
            ],
            "graph_edges": [
                {"id": "e1", "from": "n1", "to": "n2", "from_junction": "tail", "to_junction": "head"}
            ],
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "random_walk",
            "seed": 19
        },
        "conformer_policy": {
            "layout_mode": "mixed",
            "default_torsion": "trans",
            "branch_spread": "staggered",
            "ring_mode": "planar",
            "edge_overrides": [
                {"edge_id": "missing-edge", "torsion_mode": "fixed_deg", "torsion_deg": 45.0}
            ]
        },
        "artifacts": {
            "coordinates": temp_path("coords").to_string_lossy(),
            "build_manifest": temp_path("manifest").to_string_lossy(),
            "charge_manifest": temp_path("charge").to_string_lossy()
        }
    });
    let (code, payload) =
        warp_build::validate_request_json(&serde_json::to_string(&request).expect("serialize"));
    assert_eq!(code, 2);
    assert!(payload["errors"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["path"] == "/conformer_policy/edge_overrides/0/edge_id"));
}

#[test]
fn run_fixture_motif_graph_build_emits_motif_instances() {
    let (_dir, bundle) = copy_fixture_dir("pmma_motif", "fixture_motif_graph");
    let coords = temp_path("fixture_graph_coords.pdb");
    let build_manifest = temp_path("fixture_graph_manifest.json");
    let charge_manifest = temp_path("fixture_graph_charge.json");
    let topology_graph = temp_path("fixture_graph_topology.json");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "fixture-graph-001",
        "source_ref": {
            "bundle_id": "pmma_fixture_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "polymer_graph",
            "graph_root": "g1",
            "graph_nodes": [
                {"id": "g1", "token": "M2"},
                {"id": "g2", "token": "A"}
            ],
            "graph_edges": [
                {"id": "e1", "from": "g1", "to": "g2", "from_junction": "tail", "to_junction": "head"}
            ],
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "random_walk",
            "seed": 41
        },
        "conformer_policy": {
            "layout_mode": "mixed",
            "default_torsion": "trans",
            "branch_spread": "staggered",
            "ring_mode": "planar",
            "edge_overrides": [
                {"edge_id": "e1", "torsion_mode": "fixed_deg", "torsion_deg": 60.0}
            ]
        },
        "artifacts": {
            "coordinates": coords.to_string_lossy(),
            "build_manifest": build_manifest.to_string_lossy(),
            "charge_manifest": charge_manifest.to_string_lossy(),
            "topology_graph": topology_graph.to_string_lossy()
        }
    });
    let (code, payload) =
        warp_build::run_request_json(&serde_json::to_string(&request).expect("serialize"), false);
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    let graph: Value =
        serde_json::from_str(&fs::read_to_string(&topology_graph).expect("read graph"))
            .expect("parse graph");
    assert_eq!(graph["schema_version"], "warp-build.topology-graph.v5");
    assert_eq!(
        graph["motif_instances"].as_array().map(|items| items.len()),
        Some(1)
    );
    assert_eq!(graph["motif_instances"][0]["motif_token"], "M2");
    assert_eq!(graph["motif_instances"][0]["request_node_id"], "g1");
    assert!(graph["conformer_edges"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["edge_id"] == "e1"));
}

#[test]
fn run_fixture_port_cap_graph_build_applies_default_caps() {
    let (_dir, bundle) = copy_fixture_dir("pmma_port_caps", "fixture_port_caps");
    let coords = temp_path("fixture_port_caps_coords.pdb");
    let build_manifest = temp_path("fixture_port_caps_manifest.json");
    let charge_manifest = temp_path("fixture_port_caps_charge.json");
    let topology_graph = temp_path("fixture_port_caps_topology.json");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "fixture-port-caps-001",
        "source_ref": {
            "bundle_id": "pmma_port_caps_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "polymer_graph",
            "graph_root": "g1",
            "graph_nodes": [
                {"id": "g1", "token": "M2C"}
            ],
            "graph_edges": [],
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "random_walk",
            "seed": 52
        },
        "artifacts": {
            "coordinates": coords.to_string_lossy(),
            "build_manifest": build_manifest.to_string_lossy(),
            "charge_manifest": charge_manifest.to_string_lossy(),
            "topology_graph": topology_graph.to_string_lossy()
        }
    });
    let (code, payload) =
        warp_build::run_request_json(&serde_json::to_string(&request).expect("serialize"), false);
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );

    let graph: Value =
        serde_json::from_str(&fs::read_to_string(&topology_graph).expect("read graph"))
            .expect("parse graph");
    assert_eq!(graph["schema_version"], "warp-build.topology-graph.v5");
    assert_eq!(
        graph["port_policies"].as_array().map(|items| items.len()),
        Some(2)
    );
    assert_eq!(
        graph["applied_caps"].as_array().map(|items| items.len()),
        Some(2)
    );
    assert_eq!(
        graph["open_ports"].as_array().map(|items| items.len()),
        Some(0)
    );
    assert!(graph["applied_caps"]
        .as_array()
        .unwrap()
        .iter()
        .all(|item| item["application_source"].as_str() == Some("bundle_default")));
}

#[test]
fn run_fixture_branched_mix_build_emits_branch_and_port_metadata() {
    let (_dir, bundle) = copy_fixture_dir("pmma_branched_mix", "fixture_branched_mix");
    let coords = temp_path("fixture_branched_mix_coords.pdb");
    let build_manifest = temp_path("fixture_branched_mix_manifest.json");
    let charge_manifest = temp_path("fixture_branched_mix_charge.json");
    let topology_graph = temp_path("fixture_branched_mix_topology.json");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "fixture-branched-mix-001",
        "source_ref": {
            "bundle_id": "pmma_branched_mix_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "branched_polymer",
            "branch_tree": {
                "token": "A",
                "children": [
                    {
                        "parent_junction": "head",
                        "child_junction": "head",
                        "sequence": ["ARM2"],
                        "repeat_count": 1,
                        "child": {
                            "token": "B",
                            "children": [
                                {
                                    "parent_junction": "head",
                                    "child_junction": "head",
                                    "sequence": ["A"],
                                    "repeat_count": 1
                                }
                            ]
                        }
                    }
                ]
            },
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "aligned",
            "alignment_axis": "z",
            "seed": 61
        },
        "artifacts": {
            "coordinates": coords.to_string_lossy(),
            "build_manifest": build_manifest.to_string_lossy(),
            "charge_manifest": charge_manifest.to_string_lossy(),
            "topology_graph": topology_graph.to_string_lossy()
        }
    });
    let (code, payload) =
        warp_build::run_request_json(&serde_json::to_string(&request).expect("serialize"), false);
    assert_eq!(
        code,
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );

    let graph: Value =
        serde_json::from_str(&fs::read_to_string(&topology_graph).expect("read graph"))
            .expect("parse graph");
    assert_eq!(graph["build_plan"]["target_mode"], "branched_polymer");
    assert!(graph["build_plan"]["max_branch_depth"]
        .as_u64()
        .map(|value| value > 0)
        .unwrap_or(false));
    assert!(graph["motif_instances"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["motif_token"] == "ARM2"));
    assert!(graph["port_policies"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["port_class"] == "reactive_end"));
    assert!(graph["alignment_paths"]
        .as_array()
        .map(|items| !items.is_empty())
        .unwrap_or(false));
}

#[test]
fn validate_fixture_bad_motif_bundle_reports_bundle_errors() {
    let (_dir, bundle) = copy_fixture_dir("pmma_bad_motif", "fixture_bad_motif");
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": "fixture-bad-motif-001",
        "source_ref": {
            "bundle_id": "pmma_bad_motif_bundle_v1",
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": {
            "mode": "polymer_graph",
            "graph_root": "g1",
            "graph_nodes": [
                {"id": "g1", "token": "BROKEN"}
            ],
            "graph_edges": [],
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"}
        },
        "realization": {
            "conformation_mode": "extended",
            "seed": 17
        },
        "artifacts": {
            "coordinates": temp_path("fixture_bad_bundle_coords.pdb").to_string_lossy(),
            "build_manifest": temp_path("fixture_bad_bundle_manifest.json").to_string_lossy(),
            "charge_manifest": temp_path("fixture_bad_bundle_charge.json").to_string_lossy()
        }
    });
    let (code, payload) =
        warp_build::validate_request_json(&serde_json::to_string(&request).expect("serialize"));
    assert_eq!(code, 2);
    assert!(payload["errors"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["path"] == "/motif_library/BROKEN/exposed_ports/head/node_id"));
}
