use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use warp_pack::io::{write_minimal_prmtop, AmberTopology};

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
    let bundle = dir.join("bundle.json");
    write_training_oligomer(&training);
    write_prmtop(&prmtop);
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
    (dir.clone(), dir.join("bundle.json"))
}

#[test]
fn schema_and_inspect_source_work() {
    let schema = warp_build::schema_json("request").expect("schema");
    let parsed: Value = serde_json::from_str(&schema).expect("parse schema");
    assert!(parsed["properties"].get("schema_version").is_some());
    let source_schema = warp_build::schema_json("source_bundle").expect("source schema");
    let source_schema_value: Value =
        serde_json::from_str(&source_schema).expect("parse source schema");
    assert!(source_schema_value["properties"].get("schema_version").is_some());
    let build_manifest_schema =
        warp_build::schema_json("build_manifest").expect("build manifest schema");
    let build_manifest_schema_value: Value =
        serde_json::from_str(&build_manifest_schema).expect("parse build manifest schema");
    assert!(build_manifest_schema_value["properties"].get("schema_version").is_some());
    let charge_manifest_schema =
        warp_build::schema_json("charge_manifest").expect("charge manifest schema");
    let charge_manifest_schema_value: Value =
        serde_json::from_str(&charge_manifest_schema).expect("parse charge manifest schema");
    assert!(charge_manifest_schema_value["properties"].get("schema_version").is_some());
    let graph_schema = warp_build::schema_json("topology_graph").expect("graph schema");
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
    assert_eq!(payload["bundle_id"], "pmma_param_bundle_v1");
    assert_eq!(payload["unit_tokens"], json!(["A", "B", "H", "T"]));
    assert_eq!(payload["motif_tokens"], json!(["M2"]));
    assert_eq!(payload["topology_transfer_supported"], json!(true));
    assert_eq!(
        payload["sequence_token_support"]["tokens"],
        json!(["H", "A", "B", "T", "M2"])
    );
    let caps = warp_build::capabilities();
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
    assert_eq!(manifest["summary"]["total_repeat_units"], 4);
    assert_eq!(manifest["realization"]["seed"], 12345);
    assert_eq!(manifest["realization"]["seed_policy"], "explicit");
    assert_eq!(manifest["summary"]["bond_count"], 3);
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
    assert_eq!(payload["resolved_inputs"]["seed_policy"], "deterministic_default");
    assert_eq!(
        payload["resolved_inputs"]["resolved_termini_policy"],
        json!({"head": "source_default", "tail": "source_default"})
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
    let warnings = payload["warnings"].as_array().expect("warnings array");
    assert!(!warnings.is_empty());
    assert_eq!(warnings[0]["severity"], "warning");
    assert_eq!(warnings[0]["path"], "/target/termini/head");
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
fn run_polymer_graph_build_writes_cycle_metadata() {
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
        0,
        "{}",
        serde_json::to_string_pretty(&payload).unwrap()
    );
    let graph_payload: Value =
        serde_json::from_str(&fs::read_to_string(&graph).expect("read graph")).expect("graph");
    assert_eq!(graph_payload["build_plan"]["target_mode"], "polymer_graph");
    assert!(topology.exists());
    assert_eq!(graph_payload["build_plan"]["request_root_node_id"], "n1");
    assert_eq!(graph_payload["build_plan"]["graph_has_cycle"], true);
    assert_eq!(graph_payload["build_plan"]["arm_count"], 2);
    assert_eq!(
        graph_payload["cycle_basis"]
            .as_array()
            .map(|items| items.len()),
        Some(1)
    );
    assert_eq!(graph_payload["residues"][0]["node_id"], "graph.n1");
    assert_eq!(graph_payload["residues"][1]["node_id"], "graph.n2");
    assert_eq!(graph_payload["residues"][2]["node_id"], "graph.n3");
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
