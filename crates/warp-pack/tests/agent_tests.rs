use std::fs;
use std::path::{Path, PathBuf};

mod common;
use common::{temp_path, write_text};

use serde_json::{json, Value};

fn fixture_bundle_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../warp-build/tests/fixtures")
        .join(name)
        .join("bundle.json")
}

fn run_fixture_polymer_build(
    fixture_name: &str,
    request_id: &str,
    target: Value,
    realization: Value,
) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    let bundle = fixture_bundle_path(fixture_name);
    let bundle_value: Value =
        serde_json::from_str(&fs::read_to_string(&bundle).expect("read fixture bundle"))
            .expect("parse fixture bundle");
    let bundle_id = bundle_value["bundle_id"]
        .as_str()
        .expect("fixture bundle_id");
    let coords = temp_path(&format!("{request_id}_coords.pdb"));
    let build_manifest = temp_path(&format!("{request_id}_manifest.json"));
    let charge_manifest = temp_path(&format!("{request_id}_charge.json"));
    let topology_graph = temp_path(&format!("{request_id}_topology.json"));
    let request = json!({
        "schema_version": "warp-build.agent.v1",
        "request_id": request_id,
        "source_ref": {
            "bundle_id": bundle_id,
            "bundle_path": bundle.to_string_lossy(),
        },
        "target": target,
        "realization": realization,
        "artifacts": {
            "coordinates": coords.to_string_lossy(),
            "build_manifest": build_manifest.to_string_lossy(),
            "charge_manifest": charge_manifest.to_string_lossy(),
            "topology_graph": topology_graph.to_string_lossy(),
        }
    });
    let (exit_code, envelope) = warp_build::run_request_json(
        &serde_json::to_string(&request).expect("serialize fixture build request"),
        false,
    );
    assert_eq!(
        exit_code,
        0,
        "{}",
        serde_json::to_string_pretty(&envelope).expect("fixture build envelope")
    );
    (coords, build_manifest, charge_manifest, topology_graph)
}

fn write_solute(path_label: &str) -> std::path::PathBuf {
    let path = temp_path(path_label);
    write_text(
        &path,
        "ATOM      1  C   MOL A   1       0.000   0.000   0.000  1.00  0.00           C\n\
ATOM      2  O   MOL A   1       1.500   0.000   0.000  1.00  0.00           O\n\
END\n",
    );
    path
}

fn write_training_oligomer(path_label: &str) -> std::path::PathBuf {
    let path = temp_path(path_label);
    write_text(
        &path,
        "ATOM      1  C1  HDA A   1       0.000   0.000   0.000  1.00  0.00           C\n\
ATOM      2  C2  RPT A   2       3.000   0.000   0.000  1.00  0.00           C\n\
ATOM      3  C3  TLA A   3       6.000   0.000   0.000  1.00  0.00           C\n\
END\n",
    );
    path
}

fn write_branched_training_oligomer(path_label: &str) -> std::path::PathBuf {
    let path = temp_path(path_label);
    write_text(
        &path,
        "ATOM      1  C1  HDA A   1       0.000   0.000   0.000  1.00  0.00           C\n\
ATOM      2  C2  BRC A   2       3.000   0.000   0.000  1.00  0.00           C\n\
ATOM      3  C3  TLA A   3       6.000   0.000   0.000  1.00  0.00           C\n\
ATOM      4  C4  TLB A   4       3.000   3.000   0.000  1.00  0.00           C\n\
END\n",
    );
    path
}

fn write_charge_manifest(path_label: &str, payload: Value) -> std::path::PathBuf {
    let path = temp_path(path_label);
    fs::write(
        &path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&payload).expect("serialize charge manifest")
        ),
    )
    .expect("write charge manifest");
    path
}

fn write_prmtop(path_label: &str, total_charge_e: f32) -> std::path::PathBuf {
    let path = temp_path(path_label);
    let amber_charge = total_charge_e * 18.2223f32;
    write_text(
        &path,
        &format!(
            "%FLAG ATOM_NAME\n%FORMAT(20a4)\nC O\n%FLAG RESIDUE_LABEL\n%FORMAT(20a4)\nMOL\n%FLAG RESIDUE_POINTER\n%FORMAT(10I8)\n1\n%FLAG ATOMIC_NUMBER\n%FORMAT(10I8)\n6 8\n%FLAG CHARGE\n%FORMAT(5E16.8)\n{:16.8E} {:16.8E}\n",
            amber_charge,
            0.0f32
        ),
    );
    path
}

fn write_polymer_build_manifest(
    path_label: &str,
    coords_path: &std::path::Path,
    charge_path: &std::path::Path,
) -> std::path::PathBuf {
    let path = temp_path(path_label);
    let coords_name = coords_path
        .file_name()
        .and_then(|value| value.to_str())
        .expect("coords file name");
    let charge_name = charge_path
        .file_name()
        .and_then(|value| value.to_str())
        .expect("charge file name");
    fs::write(
        &path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&json!({
                "version": "warp-build.manifest.v1",
                "request_id": "build-001",
                "normalized_request": {
                    "schema_version": "warp-build.agent.v1",
                    "request_id": "build-001",
                    "source_ref": {
                        "bundle_id": "pmma_param_bundle_v1",
                        "bundle_path": "bundle.json",
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
                        "seed": 12345,
                    },
                },
                "source_bundle": {
                    "bundle_id": "pmma_param_bundle_v1",
                    "bundle_digest": "sha256:test",
                    "training_context": {
                        "mode": "oligomer_training",
                        "training_oligomer_n": 3,
                    }
                },
                "realization": {
                    "conformation_mode": "random_walk",
                    "seed": 12345,
                },
                "artifacts": {
                    "coordinates": coords_name,
                    "charge_manifest": charge_name,
                },
                "summary": {
                    "build_mode": "linear_homopolymer",
                    "total_repeat_units": 4,
                    "net_charge_e": 3.0,
                }
            }))
            .expect("serialize polymer build manifest")
        ),
    )
    .expect("write polymer build manifest");
    path
}

fn write_polymer_build_manifest_with_topology(
    path_label: &str,
    coords_path: &std::path::Path,
    topology_path: &std::path::Path,
) -> std::path::PathBuf {
    let path = temp_path(path_label);
    let coords_name = coords_path
        .file_name()
        .and_then(|value| value.to_str())
        .expect("coords file name");
    let topology_name = topology_path
        .file_name()
        .and_then(|value| value.to_str())
        .expect("topology file name");
    fs::write(
        &path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&json!({
                "version": "warp-build.manifest.v1",
                "request_id": "build-002",
                "normalized_request": {
                    "schema_version": "warp-build.agent.v1",
                    "request_id": "build-002",
                    "source_ref": {
                        "bundle_id": "pmma_param_bundle_v1",
                        "bundle_path": "bundle.json",
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
                        "seed": 12345,
                    },
                },
                "source_bundle": {
                    "bundle_id": "pmma_param_bundle_v1",
                    "bundle_digest": "sha256:test",
                    "training_context": {
                        "mode": "oligomer_training",
                        "training_oligomer_n": 3,
                    }
                },
                "realization": {
                    "conformation_mode": "random_walk",
                    "seed": 12345,
                },
                "artifacts": {
                    "coordinates": coords_name,
                    "topology": topology_name,
                    "forcefield_ref": "training.ffxml",
                },
                "summary": {
                    "build_mode": "linear_homopolymer",
                    "total_repeat_units": 4,
                    "net_charge_e": 1.0,
                }
            }))
            .expect("serialize polymer build manifest")
        ),
    )
    .expect("write polymer build manifest");
    path
}

fn write_linear_topology_graph(path_label: &str) -> std::path::PathBuf {
    let path = temp_path(path_label);
    fs::write(
        &path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&json!({
                "version": "warp-build.topology-graph.v5",
                "request_id": "build-003",
                "bundle_id": "pmma_param_bundle_v1",
                "build_plan": {
                    "target_mode": "linear_homopolymer",
                    "realization_mode": "random_walk",
                    "resolved_sequence": ["A", "A", "A"],
                    "request_root_node_id": "chain-root",
                    "expanded_root_node_id": "chain-root",
                    "root_token": "A",
                    "graph_has_cycle": false,
                    "requested_termini": {"head": "default", "tail": "default"}
                },
                "atoms": [
                    {
                        "index": 0,
                        "name": "C1",
                        "element": "C",
                        "resid": 1,
                        "resname": "HDA",
                        "charge_e": 0.0,
                        "mass": 12.01,
                        "atom_type_index": 1,
                        "amber_atom_type": "CT",
                        "lj_class": "CT",
                        "position": [0.0, 0.0, 0.0],
                        "neighbors": [1]
                    },
                    {
                        "index": 1,
                        "name": "C2",
                        "element": "C",
                        "resid": 2,
                        "resname": "RPT",
                        "charge_e": 0.0,
                        "mass": 12.01,
                        "atom_type_index": 1,
                        "amber_atom_type": "CT",
                        "lj_class": "CT",
                        "position": [3.0, 0.0, 0.0],
                        "neighbors": [0, 2]
                    },
                    {
                        "index": 2,
                        "name": "C3",
                        "element": "C",
                        "resid": 3,
                        "resname": "TLA",
                        "charge_e": 0.0,
                        "mass": 12.01,
                        "atom_type_index": 1,
                        "amber_atom_type": "CT",
                        "lj_class": "CT",
                        "position": [6.0, 0.0, 0.0],
                        "neighbors": [1]
                    }
                ],
                "bonds": [{"a": 0, "b": 1}, {"a": 1, "b": 2}],
                "angles": [{"a": 0, "b": 1, "c": 2}],
                "dihedrals": [],
                "impropers": [],
                "exclusions": [],
                "branch_points": [],
                "residue_connections": [{"a": 1, "b": 2}, {"a": 2, "b": 3}],
                "inter_residue_bonds": [
                    {"a": 0, "b": 1, "resid_a": 1, "resid_b": 2},
                    {"a": 1, "b": 2, "resid_a": 2, "resid_b": 3}
                ],
                "connection_definitions": [],
                "nonbonded_typing": {
                    "atom_type_indices": [1, 1, 1],
                    "amber_atom_types": ["CT", "CT", "CT"],
                    "lj_classes": ["CT", "CT", "CT"]
                },
                "residues": [
                    {
                        "resid": 1,
                        "resname": "HDA",
                        "node_id": "n1",
                        "request_node_id": "n1",
                        "sequence_token": "A",
                        "token_kind": "unit",
                        "source_token": "A",
                        "atom_indices": [0],
                        "ports": [{"name": "head", "attach_atom": "C1", "leaving_atoms": []}]
                    },
                    {
                        "resid": 2,
                        "resname": "RPT",
                        "node_id": "n2",
                        "request_node_id": "n2",
                        "sequence_token": "A",
                        "token_kind": "unit",
                        "source_token": "A",
                        "atom_indices": [1],
                        "ports": []
                    },
                    {
                        "resid": 3,
                        "resname": "TLA",
                        "node_id": "n3",
                        "request_node_id": "n3",
                        "sequence_token": "A",
                        "token_kind": "unit",
                        "source_token": "A",
                        "atom_indices": [2],
                        "ports": [{"name": "tail", "attach_atom": "C3", "leaving_atoms": []}]
                    }
                ],
                "sequence": ["A", "A", "A"],
                "template_sequence_resnames": ["HDA", "RPT", "TLA"],
                "applied_residue_resnames": ["HDA", "RPT", "TLA"],
                "motif_instances": [],
                "cycle_basis": [],
                "open_ports": [
                    {"node_id": "n1", "request_node_id": "n1", "resid": 1, "port_name": "head", "junction": "head"},
                    {"node_id": "n3", "request_node_id": "n3", "resid": 3, "port_name": "tail", "junction": "tail"}
                ],
                "port_policies": [],
                "applied_caps": [],
                "conformer_edges": [],
                "alignment_paths": [
                    {"kind": "longest_path", "residue_ids": [1, 2, 3], "node_ids": ["n1", "n2", "n3"]}
                ],
                "relax_metadata": null,
                "metadata": {}
            }))
            .expect("serialize topology graph")
        ),
    )
    .expect("write topology graph");
    path
}

fn write_linear_topology_graph_with_port_class(
    path_label: &str,
    port_class: &str,
) -> std::path::PathBuf {
    let path = temp_path(path_label);
    fs::write(
        &path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&json!({
                "version": "warp-build.topology-graph.v5",
                "request_id": "build-005",
                "bundle_id": "pmma_param_bundle_v1",
                "build_plan": {
                    "target_mode": "linear_homopolymer",
                    "realization_mode": "random_walk",
                    "resolved_sequence": ["A", "A", "A"],
                    "request_root_node_id": "chain-root",
                    "expanded_root_node_id": "chain-root",
                    "root_token": "A",
                    "graph_has_cycle": false,
                    "requested_termini": {"head": "default", "tail": "default"}
                },
                "atoms": [
                    {"index": 0, "name": "C1", "element": "C", "resid": 1, "resname": "HDA", "charge_e": 0.0, "mass": 12.01, "atom_type_index": 1, "amber_atom_type": "CT", "lj_class": "CT", "position": [0.0, 0.0, 0.0], "neighbors": [1]},
                    {"index": 1, "name": "C2", "element": "C", "resid": 2, "resname": "RPT", "charge_e": 0.0, "mass": 12.01, "atom_type_index": 1, "amber_atom_type": "CT", "lj_class": "CT", "position": [3.0, 0.0, 0.0], "neighbors": [0, 2]},
                    {"index": 2, "name": "C3", "element": "C", "resid": 3, "resname": "TLA", "charge_e": 0.0, "mass": 12.01, "atom_type_index": 1, "amber_atom_type": "CT", "lj_class": "CT", "position": [6.0, 0.0, 0.0], "neighbors": [1]}
                ],
                "bonds": [{"a": 0, "b": 1}, {"a": 1, "b": 2}],
                "angles": [{"a": 0, "b": 1, "c": 2}],
                "dihedrals": [],
                "impropers": [],
                "exclusions": [],
                "branch_points": [],
                "residue_connections": [{"a": 1, "b": 2}, {"a": 2, "b": 3}],
                "inter_residue_bonds": [
                    {"a": 0, "b": 1, "resid_a": 1, "resid_b": 2},
                    {"a": 1, "b": 2, "resid_a": 2, "resid_b": 3}
                ],
                "connection_definitions": [],
                "nonbonded_typing": {
                    "atom_type_indices": [1, 1, 1],
                    "amber_atom_types": ["CT", "CT", "CT"],
                    "lj_classes": ["CT", "CT", "CT"]
                },
                "residues": [
                    {"resid": 1, "resname": "HDA", "node_id": "n1", "request_node_id": "n1", "sequence_token": "A", "token_kind": "unit", "source_token": "A", "atom_indices": [0], "ports": [{"name": "head", "attach_atom": "C1", "leaving_atoms": []}]},
                    {"resid": 2, "resname": "RPT", "node_id": "n2", "request_node_id": "n2", "sequence_token": "A", "token_kind": "unit", "source_token": "A", "atom_indices": [1], "ports": []},
                    {"resid": 3, "resname": "TLA", "node_id": "n3", "request_node_id": "n3", "sequence_token": "A", "token_kind": "unit", "source_token": "A", "atom_indices": [2], "ports": [{"name": "tail", "attach_atom": "C3", "leaving_atoms": []}]}
                ],
                "sequence": ["A", "A", "A"],
                "template_sequence_resnames": ["HDA", "RPT", "TLA"],
                "applied_residue_resnames": ["HDA", "RPT", "TLA"],
                "motif_instances": [],
                "cycle_basis": [],
                "open_ports": [
                    {"node_id": "n1", "request_node_id": "n1", "resid": 1, "port_name": "head", "junction": "head", "port_class": port_class},
                    {"node_id": "n3", "request_node_id": "n3", "resid": 3, "port_name": "tail", "junction": "tail", "port_class": port_class}
                ],
                "port_policies": [
                    {"node_id": "n1", "request_node_id": "n1", "resid": 1, "port_name": "head", "junction": "head", "port_class": port_class, "default_cap": null, "allowed_caps": []},
                    {"node_id": "n3", "request_node_id": "n3", "resid": 3, "port_name": "tail", "junction": "tail", "port_class": port_class, "default_cap": null, "allowed_caps": []}
                ],
                "applied_caps": [],
                "conformer_edges": [],
                "alignment_paths": [
                    {"kind": "longest_path", "residue_ids": [1, 2, 3], "node_ids": ["n1", "n2", "n3"]}
                ],
                "relax_metadata": null,
                "metadata": {}
            }))
            .expect("serialize topology graph")
        ),
    )
    .expect("write topology graph");
    path
}

fn write_branched_topology_graph(path_label: &str) -> std::path::PathBuf {
    let path = temp_path(path_label);
    fs::write(
        &path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&json!({
                "version": "warp-build.topology-graph.v5",
                "request_id": "build-004",
                "bundle_id": "pmma_param_bundle_v1",
                "build_plan": {
                    "target_mode": "branched_polymer",
                    "realization_mode": "random_walk",
                    "resolved_sequence": ["A", "B", "A", "A"],
                    "request_root_node_id": "branch-root",
                    "expanded_root_node_id": "branch-root",
                    "root_token": "B",
                    "graph_has_cycle": false,
                    "max_branch_depth": 1,
                    "requested_termini": {"head": "default", "tail": "default"}
                },
                "atoms": [
                    {"index": 0, "name": "C1", "element": "C", "resid": 1, "resname": "HDA", "charge_e": 0.0, "mass": 12.01, "atom_type_index": 1, "amber_atom_type": "CT", "lj_class": "CT", "position": [0.0, 0.0, 0.0], "neighbors": [1]},
                    {"index": 1, "name": "C2", "element": "C", "resid": 2, "resname": "BRC", "charge_e": 0.0, "mass": 12.01, "atom_type_index": 1, "amber_atom_type": "CT", "lj_class": "CT", "position": [3.0, 0.0, 0.0], "neighbors": [0, 2, 3]},
                    {"index": 2, "name": "C3", "element": "C", "resid": 3, "resname": "TLA", "charge_e": 0.0, "mass": 12.01, "atom_type_index": 1, "amber_atom_type": "CT", "lj_class": "CT", "position": [6.0, 0.0, 0.0], "neighbors": [1]},
                    {"index": 3, "name": "C4", "element": "C", "resid": 4, "resname": "TLB", "charge_e": 0.0, "mass": 12.01, "atom_type_index": 1, "amber_atom_type": "CT", "lj_class": "CT", "position": [3.0, 3.0, 0.0], "neighbors": [1]}
                ],
                "bonds": [{"a": 0, "b": 1}, {"a": 1, "b": 2}, {"a": 1, "b": 3}],
                "angles": [{"a": 0, "b": 1, "c": 2}, {"a": 0, "b": 1, "c": 3}, {"a": 2, "b": 1, "c": 3}],
                "dihedrals": [],
                "impropers": [],
                "exclusions": [],
                "branch_points": [{"atom_index": 1, "degree": 3, "resid": 2, "atom_name": "C2"}],
                "residue_connections": [{"a": 1, "b": 2}, {"a": 2, "b": 3}, {"a": 2, "b": 4}],
                "inter_residue_bonds": [
                    {"a": 0, "b": 1, "resid_a": 1, "resid_b": 2},
                    {"a": 1, "b": 2, "resid_a": 2, "resid_b": 3},
                    {"a": 1, "b": 3, "resid_a": 2, "resid_b": 4}
                ],
                "connection_definitions": [],
                "nonbonded_typing": {
                    "atom_type_indices": [1, 1, 1, 1],
                    "amber_atom_types": ["CT", "CT", "CT", "CT"],
                    "lj_classes": ["CT", "CT", "CT", "CT"]
                },
                "residues": [
                    {"resid": 1, "resname": "HDA", "node_id": "n1", "request_node_id": "n1", "sequence_token": "A", "token_kind": "unit", "source_token": "A", "atom_indices": [0], "ports": [{"name": "head", "attach_atom": "C1", "leaving_atoms": []}]},
                    {"resid": 2, "resname": "BRC", "node_id": "n2", "request_node_id": "n2", "sequence_token": "B", "token_kind": "unit", "source_token": "B", "atom_indices": [1], "ports": []},
                    {"resid": 3, "resname": "TLA", "node_id": "n3", "request_node_id": "n3", "sequence_token": "A", "token_kind": "unit", "source_token": "A", "atom_indices": [2], "ports": [{"name": "tail_a", "attach_atom": "C3", "leaving_atoms": []}]},
                    {"resid": 4, "resname": "TLB", "node_id": "n4", "request_node_id": "n4", "sequence_token": "A", "token_kind": "unit", "source_token": "A", "atom_indices": [3], "ports": [{"name": "tail_b", "attach_atom": "C4", "leaving_atoms": []}]}
                ],
                "sequence": ["A", "B", "A", "A"],
                "template_sequence_resnames": ["HDA", "BRC", "TLA", "TLB"],
                "applied_residue_resnames": ["HDA", "BRC", "TLA", "TLB"],
                "motif_instances": [],
                "cycle_basis": [],
                "open_ports": [
                    {"node_id": "n1", "request_node_id": "n1", "resid": 1, "port_name": "head", "junction": "head", "port_class": "reactive_end"},
                    {"node_id": "n3", "request_node_id": "n3", "resid": 3, "port_name": "tail_a", "junction": "tail_a", "port_class": "reactive_end"},
                    {"node_id": "n4", "request_node_id": "n4", "resid": 4, "port_name": "tail_b", "junction": "tail_b", "port_class": "reactive_end"}
                ],
                "port_policies": [
                    {"node_id": "n1", "request_node_id": "n1", "resid": 1, "port_name": "head", "junction": "head", "port_class": "reactive_end", "default_cap": null, "allowed_caps": []},
                    {"node_id": "n3", "request_node_id": "n3", "resid": 3, "port_name": "tail_a", "junction": "tail_a", "port_class": "reactive_end", "default_cap": null, "allowed_caps": []},
                    {"node_id": "n4", "request_node_id": "n4", "resid": 4, "port_name": "tail_b", "junction": "tail_b", "port_class": "reactive_end", "default_cap": null, "allowed_caps": []}
                ],
                "applied_caps": [],
                "conformer_edges": [],
                "alignment_paths": [
                    {"kind": "longest_path", "residue_ids": [1, 2, 3], "node_ids": ["n1", "n2", "n3"]}
                ],
                "relax_metadata": null,
                "metadata": {}
            }))
            .expect("serialize topology graph")
        ),
    )
    .expect("write topology graph");
    path
}

fn write_polymer_build_manifest_with_topology_graph(
    path_label: &str,
    coords_path: &std::path::Path,
    topology_path: &std::path::Path,
    topology_graph_path: &std::path::Path,
) -> std::path::PathBuf {
    write_polymer_build_manifest_with_topology_graph_request(
        path_label,
        coords_path,
        topology_path,
        topology_graph_path,
        "build-003",
        "linear_homopolymer",
        3,
    )
}

fn write_polymer_build_manifest_with_topology_graph_request(
    path_label: &str,
    coords_path: &std::path::Path,
    topology_path: &std::path::Path,
    topology_graph_path: &std::path::Path,
    request_id: &str,
    build_mode: &str,
    repeat_units: usize,
) -> std::path::PathBuf {
    let path = temp_path(path_label);
    let coords_name = coords_path
        .file_name()
        .and_then(|value| value.to_str())
        .expect("coords file name");
    let topology_name = topology_path
        .file_name()
        .and_then(|value| value.to_str())
        .expect("topology file name");
    let topology_graph_name = topology_graph_path
        .file_name()
        .and_then(|value| value.to_str())
        .expect("topology graph file name");
    fs::write(
        &path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&json!({
                "version": "warp-build.manifest.v1",
                "request_id": request_id,
                "normalized_request": {
                    "schema_version": "warp-build.agent.v1",
                    "request_id": request_id,
                    "source_ref": {
                        "bundle_id": "pmma_param_bundle_v1",
                        "bundle_path": "bundle.json",
                    },
                    "target": {
                        "mode": build_mode,
                        "repeat_unit": "A",
                        "n_repeat": repeat_units,
                        "termini": {"head": "default", "tail": "default"},
                    },
                    "realization": {
                        "conformation_mode": "random_walk",
                        "seed": 12345,
                    },
                },
                "source_bundle": {
                    "bundle_id": "pmma_param_bundle_v1",
                    "bundle_digest": "sha256:test",
                },
                "artifacts": {
                    "coordinates": coords_name,
                    "topology": topology_name,
                    "topology_graph": topology_graph_name,
                    "forcefield_ref": "training.ffxml",
                },
                "summary": {
                    "build_mode": build_mode,
                    "total_repeat_units": repeat_units,
                    "net_charge_e": 1.0,
                }
            }))
            .expect("serialize polymer build manifest")
        ),
    )
    .expect("write polymer build manifest");
    path
}

fn write_polymer_build_manifest_with_md_ready_handoff_only(
    path_label: &str,
    coords_path: &std::path::Path,
    topology_path: &std::path::Path,
    topology_graph_path: &std::path::Path,
) -> std::path::PathBuf {
    let path = write_polymer_build_manifest_with_topology_graph_request(
        path_label,
        coords_path,
        topology_path,
        topology_graph_path,
        "build-003",
        "linear_homopolymer",
        3,
    );
    let mut manifest: Value =
        serde_json::from_str(&fs::read_to_string(&path).expect("read build manifest"))
            .expect("parse build manifest");
    let topology = manifest["artifacts"]["topology"].clone();
    let topology_graph = manifest["artifacts"]["topology_graph"].clone();
    let forcefield_ref = manifest["artifacts"]["forcefield_ref"].clone();
    manifest["artifacts"]["topology"] = Value::Null;
    manifest["artifacts"]["topology_graph"] = Value::Null;
    manifest["artifacts"]["forcefield_ref"] = Value::Null;
    manifest["md_ready_handoff"] = json!({
        "topology": topology,
        "topology_graph": topology_graph,
        "forcefield_ref": forcefield_ref,
    });
    fs::write(
        &path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&manifest).expect("serialize build manifest")
        ),
    )
    .expect("rewrite build manifest");
    path
}

#[test]
fn agent_schema_includes_default_version() {
    let schema_text = warp_pack::agent::schema_json("request").expect("request schema");
    let schema: Value = serde_json::from_str(&schema_text).expect("parse schema");
    assert_eq!(
        schema["properties"]["schema_version"]["default"],
        "warp-pack.agent.v1"
    );
}

#[test]
fn capabilities_advertise_components_and_prmtop() {
    let caps = warp_pack::agent::capabilities();
    assert_eq!(caps["preferred_solute_input"], "components");
    assert_eq!(
        caps["supported_ion_species"],
        json!(["Br-", "Ca2+", "Cl-", "I-", "K+", "Li+", "Mg2+", "Na+"])
    );
    assert_eq!(
        caps["supported_salt_names"],
        json!(["cacl2", "kcl", "libr", "licl", "mgbr2", "mgcl2", "nabr", "nacl", "nai"])
    );
    assert_eq!(
        caps["supported_custom_chemistry_inputs"],
        json!(["catalog.ions", "catalog.salts"])
    );
    assert_eq!(
        caps["supported_salt_inputs"],
        json!(["salt.name", "salt.formula", "salt.species", "legacy_pair"])
    );
    assert_eq!(
        caps["supported_morphology_modes"],
        json!([
            "single_chain_solution",
            "amorphous_bulk",
            "backbone_aligned_bulk"
        ])
    );
    assert_eq!(
        caps["supported_charge_sources"],
        json!(["charge_manifest", "prmtop"])
    );
    assert_eq!(
        caps["supported_solute_inputs"],
        json!(["components", "solute", "polymer_build"])
    );
}

#[test]
fn agent_validate_rejects_removed_inline_polymer_request() {
    let training = write_training_oligomer("polymer_validate_training.pdb");
    let payload = json!({
        "version": "warp-pack.agent.v1",
        "polymer": {
            "param_source": {
                "mode": "oligomer_training",
                "artifact": training.to_string_lossy(),
                "nmer": 3,
            },
            "target": {
                "mode": "linear_homopolymer",
                "n_repeat": 4,
                "sequence": "AAAA",
                "tacticity": "isotactic",
                "termini": {"head": "HEND", "tail": "TEND"},
                "conformation": {"mode": "extended"},
            },
        },
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": 30.0, "shape": "cubic"},
            "solvent": {"mode": "none"},
            "ions": {"neutralize": true, "cation": "Na+", "anion": "Cl-"},
            "morphology": {"mode": "single_chain_solution"},
        },
        "outputs": {
            "coordinates": temp_path("polymer_validate_out.pdb").to_string_lossy(),
            "manifest": temp_path("polymer_validate_manifest.json").to_string_lossy(),
        },
    });

    let (exit_code, result) = warp_pack::agent::validate_request_json(
        &serde_json::to_string(&payload).expect("serialize payload"),
    );
    assert_eq!(exit_code, 2);
    assert_eq!(result["errors"][0]["path"], "/polymer");
}

#[test]
fn agent_validate_rejects_removed_inline_polymer_sequence_request() {
    let training = write_training_oligomer("polymer_invalid_sequence_training.pdb");
    let payload = json!({
        "version": "warp-pack.agent.v1",
        "polymer": {
            "param_source": {
                "mode": "oligomer_training",
                "artifact": training.to_string_lossy(),
                "nmer": 3,
            },
            "target": {
                "mode": "linear_homopolymer",
                "n_repeat": 4,
                "sequence": "ABBA",
                "conformation": {"mode": "extended"},
            },
        },
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": 30.0, "shape": "cubic"},
            "solvent": {"mode": "none"},
            "ions": {"neutralize": false, "cation": "Na+", "anion": "Cl-"},
            "morphology": {"mode": "single_chain_solution"},
        },
        "outputs": {
            "coordinates": temp_path("polymer_invalid_sequence_out.pdb").to_string_lossy(),
            "manifest": temp_path("polymer_invalid_sequence_manifest.json").to_string_lossy(),
        },
    });

    let (exit_code, result) = warp_pack::agent::validate_request_json(
        &serde_json::to_string(&payload).expect("serialize payload"),
    );
    assert_eq!(exit_code, 2);
    assert_eq!(result["errors"][0]["path"], "/polymer");
}

#[test]
fn agent_validate_rejects_inline_polymer_without_sequence_token() {
    let training = write_training_oligomer("polymer_missing_sequence_training.pdb");
    let payload = json!({
        "version": "warp-pack.agent.v1",
        "polymer": {
            "param_source": {
                "mode": "oligomer_training",
                "artifact": training.to_string_lossy(),
                "nmer": 3,
            },
            "target": {
                "mode": "linear_homopolymer",
                "n_repeat": 4,
                "conformation": {"mode": "extended"},
            },
        },
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": 30.0, "shape": "cubic"},
            "solvent": {"mode": "none"},
            "ions": {"neutralize": false, "cation": "Na+", "anion": "Cl-"},
            "morphology": {"mode": "single_chain_solution"},
        },
        "outputs": {
            "coordinates": temp_path("polymer_missing_sequence_out.pdb").to_string_lossy(),
            "manifest": temp_path("polymer_missing_sequence_manifest.json").to_string_lossy(),
        },
    });

    let (exit_code, result) = warp_pack::agent::validate_request_json(
        &serde_json::to_string(&payload).expect("serialize payload"),
    );
    assert_eq!(exit_code, 2);
    assert_eq!(result["errors"][0]["path"], "/polymer");
}

#[test]
fn agent_validate_rejects_mixed_components_and_legacy_input() {
    let solute = write_solute("components_mixed_solute.pdb");
    let payload = json!({
        "version": "warp-pack.agent.v1",
        "components": [
            {
                "name": "solute_copy",
                "count": 1,
                "source": {
                    "kind": "artifact",
                    "artifact": {
                        "path": solute.to_string_lossy(),
                        "kind": "small_molecule",
                    }
                }
            }
        ],
        "solute": {
            "path": solute.to_string_lossy(),
            "kind": "small_molecule",
        },
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": 30.0, "shape": "cubic"},
            "solvent": {"mode": "none"},
            "ions": {"neutralize": false, "cation": "Na+", "anion": "Cl-"},
            "morphology": {"mode": "amorphous_bulk"},
        },
        "outputs": {
            "coordinates": temp_path("components_mixed_out.pdb").to_string_lossy(),
            "manifest": temp_path("components_mixed_manifest.json").to_string_lossy(),
        },
    });

    let (exit_code, result) = warp_pack::agent::validate_request_json(
        &serde_json::to_string(&payload).expect("serialize payload"),
    );
    assert_eq!(exit_code, 2);
    assert_eq!(result["errors"][0]["path"], "/components");
}

#[test]
fn agent_run_solute_writes_manifest_and_outputs() {
    let solute = write_solute("agent_solute.pdb");
    let charges = write_charge_manifest(
        "agent_solute_charge.json",
        json!({
            "version": "warp-pack.charge-manifest.v1",
            "net_charge_e": -2.0,
        }),
    );
    let coords = temp_path("agent_solute_out.pdb");
    let manifest = temp_path("agent_solute_manifest.json");

    let payload = json!({
        "version": "warp-pack.agent.v1",
        "run_id": "solute-run",
        "solute": {
            "path": solute.to_string_lossy(),
            "kind": "small_molecule",
            "charge_manifest": charges.to_string_lossy(),
        },
        "environment": {
            "box": {"mode": "padding", "padding_angstrom": 8.0, "shape": "cubic"},
            "solvent": {"mode": "explicit", "model": "tip3p"},
            "ions": {"neutralize": true, "salt_molar": 0.05, "cation": "Na+", "anion": "Cl-"},
            "morphology": {"mode": "single_chain_solution"},
        },
        "outputs": {
            "coordinates": coords.to_string_lossy(),
            "manifest": manifest.to_string_lossy(),
        },
    });

    let (exit_code, envelope) = warp_pack::agent::run_request_json(
        &serde_json::to_string(&payload).expect("serialize payload"),
        false,
    );
    assert_eq!(
        exit_code,
        0,
        "{}",
        serde_json::to_string_pretty(&envelope).expect("envelope text")
    );
    assert_eq!(envelope["status"], "ok");
    assert!(coords.exists());
    assert!(manifest.exists());

    let manifest_value: Value =
        serde_json::from_str(&fs::read_to_string(&manifest).expect("read manifest"))
            .expect("parse manifest");
    assert_eq!(
        manifest_value["neutralization_policy_applied"],
        "charge_manifest.net_charge_e"
    );
    assert_eq!(manifest_value["net_charge_before_neutralization"], -2.0);
    assert_eq!(
        manifest_value["charge_source_kinds"],
        json!(["net_charge_e"])
    );
    assert!(manifest_value["water_count"].as_u64().unwrap_or(0) > 0);
    assert!(
        manifest_value["achieved_salt_counts_by_species"]["Na+"]
            .as_u64()
            .unwrap_or(0)
            >= manifest_value["achieved_salt_counts_by_species"]["Cl-"]
                .as_u64()
                .unwrap_or(0)
    );
}

#[test]
fn agent_run_polymer_build_handoff_uses_manifest_artifacts() {
    let coords_in = write_training_oligomer("agent_polymer_handoff_coords.pdb");
    let charge = write_charge_manifest(
        "agent_polymer_handoff_charge.json",
        json!({
            "version": "warp-pack.charge-manifest.v1",
            "net_charge_e": 3.0,
        }),
    );
    let build_manifest =
        write_polymer_build_manifest("agent_polymer_handoff_build.json", &coords_in, &charge);
    let coords_out = temp_path("agent_polymer_handoff_out.pdb");
    let manifest_out = temp_path("agent_polymer_handoff_manifest.json");
    let md_package_out = temp_path("agent_polymer_handoff_manifest.md-ready.json");

    let payload = json!({
        "version": "warp-pack.agent.v1",
        "run_id": "warp-build-handoff-run",
        "polymer_build": {
            "build_manifest": build_manifest.to_string_lossy(),
        },
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": 30.0, "shape": "cubic"},
            "solvent": {"mode": "none"},
            "ions": {"neutralize": true, "cation": "Na+", "anion": "Cl-"},
            "morphology": {"mode": "single_chain_solution"},
        },
        "outputs": {
            "coordinates": coords_out.to_string_lossy(),
            "manifest": manifest_out.to_string_lossy(),
            "md_package": md_package_out.to_string_lossy(),
            "format": "pdb-strict",
            "write_conect": true,
            "preserve_topology_graph": true,
        },
    });

    let (validate_code, validate_result) = warp_pack::agent::validate_request_json(
        &serde_json::to_string(&payload).expect("serialize validate payload"),
    );
    assert_eq!(validate_code, 0);
    assert_eq!(validate_result["valid"], true);
    assert_eq!(validate_result["schema_version"], "warp-pack.agent.v1");
    assert_eq!(
        validate_result["resolved_inputs"]["polymer_build_manifest_version"],
        "warp-build.manifest.v1"
    );
    assert_eq!(
        validate_result["resolved_inputs"]["topology_graph_present"],
        false
    );
    assert_eq!(
        validate_result["resolved_inputs"]["morphology_mode"],
        "single_chain_solution"
    );
    assert_eq!(
        validate_result["resolved_inputs"]["neutralization_preconditions"]["satisfied"],
        true
    );
    assert_eq!(
        validate_result["warnings"][0]["code"],
        "W_TOPOLOGY_GRAPH_MISSING"
    );
    assert_eq!(
        validate_result["warnings"][0]["path"],
        "/polymer_build/topology_graph"
    );

    let (exit_code, envelope) = warp_pack::agent::run_request_json(
        &serde_json::to_string(&payload).expect("serialize payload"),
        false,
    );
    assert_eq!(
        exit_code,
        0,
        "{}",
        serde_json::to_string_pretty(&envelope).expect("envelope text")
    );
    assert!(coords_out.exists());
    assert!(manifest_out.exists());
    assert!(md_package_out.exists());

    let manifest_value: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_out).expect("read manifest"))
            .expect("parse manifest");
    assert_eq!(
        manifest_value["neutralization_policy_applied"],
        "charge_manifest.net_charge_e"
    );
    assert_eq!(manifest_value["net_charge_before_neutralization"], 3.0);
    assert_eq!(
        manifest_value["polymer_build_handoff"]["manifest_version"],
        "warp-build.manifest.v1"
    );
    assert_eq!(
        manifest_value["built_solute_artifact"]["target_n_repeat"],
        4
    );
    assert_eq!(
        manifest_value["source_solute_artifact"]["connectivity_hint"],
        "polymer_build_handoff"
    );
    assert_eq!(manifest_value["achieved_salt_counts_by_species"]["Cl-"], 3);
    assert!(manifest_value["artifact_digests"]["coordinates"]
        .as_str()
        .expect("coordinates digest")
        .starts_with("sha256:"));
    assert_eq!(
        manifest_value["output_metadata"]["coordinates"]["format"],
        "pdb-strict"
    );
    assert_eq!(
        manifest_value["output_metadata"]["coordinates"]["write_conect"],
        true
    );
    assert_eq!(
        manifest_value["output_metadata"]["md_package"]["path"],
        md_package_out.to_string_lossy().to_string()
    );

    let md_package_value: Value =
        serde_json::from_str(&fs::read_to_string(&md_package_out).expect("read md package"))
            .expect("parse md package");
    assert_eq!(md_package_value["coordinates"]["format"], "pdb-strict");
    assert_eq!(md_package_value["coordinates"]["write_conect"], true);
    assert_eq!(
        md_package_value["pack_manifest"]["path"],
        manifest_out.to_string_lossy().to_string()
    );
    let coords_text = fs::read_to_string(&coords_out).expect("read output coordinates");
    let atom_line = coords_text
        .lines()
        .find(|line| line.starts_with("ATOM") || line.starts_with("HETATM"))
        .expect("polymer atom line");
    assert_eq!(atom_line[17..20].trim(), "HDA");
    assert_eq!(atom_line[21..22].trim(), "A");
    assert_eq!(atom_line[22..26].trim(), "1");
    assert!(atom_line[30..38].trim().parse::<f32>().is_ok());
}

#[test]
fn agent_run_components_bulk_uses_prmtop_charge_fallback() {
    let solute = write_solute("components_bulk_solute.pdb");
    let prmtop = write_prmtop("components_bulk_solute.prmtop", 1.0);
    let coords_out = temp_path("components_bulk_out.pdb");
    let manifest_out = temp_path("components_bulk_manifest.json");

    let payload = json!({
        "version": "warp-pack.agent.v1",
        "run_id": "components-bulk-run",
        "components": [
            {
                "name": "chain_a",
                "count": 2,
                "source": {
                    "kind": "artifact",
                    "artifact": {
                        "path": solute.to_string_lossy(),
                        "kind": "polymer_chain",
                        "topology": prmtop.to_string_lossy(),
                    }
                }
            }
        ],
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": [40.0, 40.0, 40.0], "shape": "orthorhombic"},
            "solvent": {"mode": "none"},
            "ions": {"neutralize": true, "cation": "Na+", "anion": "Cl-"},
            "morphology": {"mode": "amorphous_bulk"},
        },
        "outputs": {
            "coordinates": coords_out.to_string_lossy(),
            "manifest": manifest_out.to_string_lossy(),
        },
    });

    let (exit_code, envelope) = warp_pack::agent::run_request_json(
        &serde_json::to_string(&payload).expect("serialize payload"),
        false,
    );
    assert_eq!(
        exit_code,
        0,
        "{}",
        serde_json::to_string_pretty(&envelope).expect("envelope text")
    );
    assert_eq!(
        envelope["warnings"][0]["code"],
        "W_CHARGE_SOURCE_PRMTOP_FALLBACK"
    );

    let manifest_value: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_out).expect("read manifest"))
            .expect("parse manifest");
    assert_eq!(manifest_value["morphology_mode"], "amorphous_bulk");
    assert_eq!(manifest_value["net_charge_before_neutralization"], 2.0);
    assert_eq!(
        manifest_value["neutralization_policy_applied"],
        "component_sum"
    );
    assert_eq!(
        manifest_value["charge_source_kinds"],
        json!(["prmtop.total_charge"])
    );
    assert_eq!(
        manifest_value["warnings"][0]["code"],
        "W_CHARGE_SOURCE_PRMTOP_FALLBACK"
    );
    assert_eq!(manifest_value["component_inventory"][0]["count"], 2);
    assert_eq!(
        manifest_value["component_charge_resolution"][0]["total_component_charge_e"],
        2.0
    );
    assert_eq!(manifest_value["achieved_salt_counts_by_species"]["Cl-"], 2);
}

#[test]
fn agent_run_components_bulk_resolves_box_from_density_target() {
    let solute = write_solute("components_density_solute.pdb");
    let prmtop = write_prmtop("components_density_solute.prmtop", 0.0);
    let coords_out = temp_path("components_density_out.pdb");
    let manifest_out = temp_path("components_density_manifest.json");

    let payload = json!({
        "version": "warp-pack.agent.v1",
        "run_id": "components-density-run",
        "components": [
            {
                "name": "chain_a",
                "count": 2,
                "source": {
                    "kind": "artifact",
                    "artifact": {
                        "path": solute.to_string_lossy(),
                        "kind": "polymer_chain",
                        "topology": prmtop.to_string_lossy(),
                    }
                }
            }
        ],
        "environment": {
            "box": {"mode": "fixed_size", "shape": "cubic"},
            "solvent": {"mode": "none"},
            "ions": {"neutralize": false, "cation": "Na+", "anion": "Cl-"},
            "morphology": {"mode": "amorphous_bulk", "target_density_g_cm3": 1.0},
        },
        "outputs": {
            "coordinates": coords_out.to_string_lossy(),
            "manifest": manifest_out.to_string_lossy(),
        },
    });

    let (exit_code, envelope) = warp_pack::agent::run_request_json(
        &serde_json::to_string(&payload).expect("serialize payload"),
        false,
    );
    assert_eq!(
        exit_code,
        0,
        "{}",
        serde_json::to_string_pretty(&envelope).expect("envelope text")
    );

    let manifest_value: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_out).expect("read manifest"))
            .expect("parse manifest");
    assert_eq!(manifest_value["morphology"]["target_density_g_cm3"], 1.0);
    assert_eq!(
        manifest_value["engine_decisions"]["details"]["box_resolution"]["box_policy"],
        "target_density_g_cm3"
    );
    assert!(
        manifest_value["final_box_size_angstrom"][0]
            .as_f64()
            .unwrap()
            > 0.0
    );
}

#[test]
fn agent_run_backbone_aligned_bulk_records_alignment() {
    let solute = write_solute("components_aligned_solute.pdb");
    let prmtop = write_prmtop("components_aligned_solute.prmtop", 0.0);
    let coords_out = temp_path("components_aligned_out.pdb");
    let manifest_out = temp_path("components_aligned_manifest.json");

    let payload = json!({
        "version": "warp-pack.agent.v1",
        "run_id": "components-aligned-run",
        "components": [
            {
                "name": "chain_a",
                "count": 2,
                "source": {
                    "kind": "artifact",
                    "artifact": {
                        "path": solute.to_string_lossy(),
                        "kind": "polymer_chain",
                        "topology": prmtop.to_string_lossy(),
                    }
                }
            }
        ],
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": [40.0, 40.0, 60.0], "shape": "orthorhombic"},
            "solvent": {"mode": "none"},
            "ions": {"neutralize": false, "cation": "Na+", "anion": "Cl-"},
            "morphology": {"mode": "backbone_aligned_bulk", "alignment_axis": "z"},
        },
        "outputs": {
            "coordinates": coords_out.to_string_lossy(),
            "manifest": manifest_out.to_string_lossy(),
        },
    });

    let (exit_code, envelope) = warp_pack::agent::run_request_json(
        &serde_json::to_string(&payload).expect("serialize payload"),
        false,
    );
    assert_eq!(
        exit_code,
        0,
        "{}",
        serde_json::to_string_pretty(&envelope).expect("envelope text")
    );

    let manifest_value: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_out).expect("read manifest"))
            .expect("parse manifest");
    assert_eq!(
        manifest_value["morphology"]["mode"],
        "backbone_aligned_bulk"
    );
    assert_eq!(manifest_value["morphology"]["alignment_axis"], "z");
    assert!(manifest_value["component_inventory"][0]["aligned_euler"].is_array());
    assert_eq!(
        manifest_value["engine_decisions"]["details"]["morphology_policy"]["orientation_policy"],
        "principal_axis_aligned.azimuth_staggered"
    );
}

#[test]
fn agent_run_polymer_build_handoff_uses_manifest_topology_fallback() {
    let coords_in = write_training_oligomer("agent_polymer_handoff_topology_coords.pdb");
    let topology = write_prmtop("agent_polymer_handoff_topology.prmtop", 1.0);
    let build_manifest = write_polymer_build_manifest_with_topology(
        "agent_polymer_handoff_topology_build.json",
        &coords_in,
        &topology,
    );
    let coords_out = temp_path("agent_polymer_handoff_topology_out.pdb");
    let manifest_out = temp_path("agent_polymer_handoff_topology_manifest.json");

    let payload = json!({
        "version": "warp-pack.agent.v1",
        "run_id": "warp-build-topology-run",
        "polymer_build": {
            "build_manifest": build_manifest.to_string_lossy(),
        },
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": 30.0, "shape": "cubic"},
            "solvent": {"mode": "none"},
            "ions": {"neutralize": true, "cation": "Na+", "anion": "Cl-"},
            "morphology": {"mode": "single_chain_solution"},
        },
        "outputs": {
            "coordinates": coords_out.to_string_lossy(),
            "manifest": manifest_out.to_string_lossy(),
        },
    });

    let (exit_code, envelope) = warp_pack::agent::run_request_json(
        &serde_json::to_string(&payload).expect("serialize payload"),
        false,
    );
    assert_eq!(
        exit_code,
        0,
        "{}",
        serde_json::to_string_pretty(&envelope).expect("envelope text")
    );

    let manifest_value: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_out).expect("read manifest"))
            .expect("parse manifest");
    assert_eq!(
        manifest_value["charge_source_kinds"],
        json!(["prmtop.total_charge"])
    );
    assert_eq!(manifest_value["net_charge_before_neutralization"], 1.0);
    assert_eq!(
        manifest_value["component_inventory"][0]["topology"],
        topology.to_string_lossy().to_string()
    );
    assert_eq!(
        manifest_value["component_inventory"][0]["forcefield_ref"],
        build_manifest
            .parent()
            .expect("manifest parent")
            .join("training.ffxml")
            .to_string_lossy()
            .to_string()
    );
}

#[test]
fn agent_run_amorphous_bulk_records_typed_graph_packing_hints() {
    let coords_in = write_training_oligomer("agent_polymer_graph_handoff_coords.pdb");
    let topology = write_prmtop("agent_polymer_graph_handoff.prmtop", 1.0);
    let topology_graph = write_linear_topology_graph("agent_polymer_graph_handoff.topology.json");
    let build_manifest = write_polymer_build_manifest_with_topology_graph(
        "agent_polymer_graph_handoff_build.json",
        &coords_in,
        &topology,
        &topology_graph,
    );
    let coords_out = temp_path("agent_polymer_graph_handoff_out.pdb");
    let manifest_out = temp_path("agent_polymer_graph_handoff_manifest.json");

    let payload = json!({
        "version": "warp-pack.agent.v1",
        "run_id": "polymer-graph-handoff-run",
        "components": [
            {
                "name": "pmma_chain",
                "count": 2,
                "source": {
                    "kind": "polymer_build",
                    "polymer_build": {
                        "build_manifest": build_manifest.to_string_lossy(),
                    }
                }
            }
        ],
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": 48.0, "shape": "cubic"},
            "solvent": {"mode": "none"},
            "ions": {"neutralize": false, "cation": "Na+", "anion": "Cl-"},
            "morphology": {"mode": "amorphous_bulk"},
        },
        "outputs": {
            "coordinates": coords_out.to_string_lossy(),
            "manifest": manifest_out.to_string_lossy(),
        },
    });

    let (exit_code, envelope) = warp_pack::agent::run_request_json(
        &serde_json::to_string(&payload).expect("serialize payload"),
        false,
    );
    assert_eq!(
        exit_code,
        0,
        "{}",
        serde_json::to_string_pretty(&envelope).expect("envelope text")
    );

    let manifest_value: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_out).expect("read manifest"))
            .expect("parse manifest");
    assert_eq!(
        manifest_value["engine_decisions"]["details"]["morphology_policy"]
            ["typed_graph_helper_applied"],
        true
    );
    let hints = manifest_value["engine_decisions"]["details"]["morphology_policy"]
        ["component_packing_hints"]
        .as_array()
        .expect("packing hints");
    assert_eq!(hints.len(), 2);
    assert_eq!(hints[0]["architecture"], "linear");
    assert_eq!(hints[0]["region_policy"], "open_port_shell_faces");
    assert_eq!(hints[0]["open_port_policy"], "free_end_distribution");
    assert_eq!(
        hints[0]["alignment_source"],
        "topology_graph.alignment_path"
    );
    assert_eq!(hints[0]["constraint_shape"], "inside_box");
    assert_eq!(hints[0]["cohort_policy"], "single_architecture");
    assert_eq!(
        manifest_value["component_inventory"][0]["topology_graph_summary"]["open_port_count"],
        2
    );
    assert_eq!(
        manifest_value["component_inventory"][0]["topology_graph_summary"]["architecture"],
        "linear"
    );
}

#[test]
fn agent_run_polymer_build_handoff_reads_md_ready_handoff_fallbacks() {
    let coords_in = write_training_oligomer("agent_polymer_md_ready_coords.pdb");
    let topology = write_prmtop("agent_polymer_md_ready.prmtop", 1.0);
    let topology_graph = write_linear_topology_graph("agent_polymer_md_ready.topology.json");
    let build_manifest = write_polymer_build_manifest_with_md_ready_handoff_only(
        "agent_polymer_md_ready_build.json",
        &coords_in,
        &topology,
        &topology_graph,
    );
    let coords_out = temp_path("agent_polymer_md_ready_out.pdb");
    let manifest_out = temp_path("agent_polymer_md_ready_manifest.json");

    let payload = json!({
        "version": "warp-pack.agent.v1",
        "run_id": "warp-build-md-ready-run",
        "polymer_build": {
            "build_manifest": build_manifest.to_string_lossy(),
        },
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": 30.0, "shape": "cubic"},
            "solvent": {"mode": "none"},
            "ions": {"neutralize": false, "cation": "Na+", "anion": "Cl-"},
            "morphology": {"mode": "single_chain_solution"},
        },
        "outputs": {
            "coordinates": coords_out.to_string_lossy(),
            "manifest": manifest_out.to_string_lossy(),
            "format": "pdb-strict",
            "write_conect": true,
            "preserve_topology_graph": true,
        },
    });

    let (exit_code, envelope) = warp_pack::agent::run_request_json(
        &serde_json::to_string(&payload).expect("serialize payload"),
        false,
    );
    assert_eq!(
        exit_code,
        0,
        "{}",
        serde_json::to_string_pretty(&envelope).expect("envelope text")
    );

    let manifest_value: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_out).expect("read manifest"))
            .expect("parse manifest");
    assert_eq!(
        manifest_value["component_inventory"][0]["topology"],
        topology.to_string_lossy().to_string()
    );
    assert_eq!(
        manifest_value["component_inventory"][0]["topology_graph"],
        topology_graph.to_string_lossy().to_string()
    );
    assert_eq!(
        manifest_value["component_inventory"][0]["forcefield_ref"],
        build_manifest
            .parent()
            .expect("manifest parent")
            .join("training.ffxml")
            .to_string_lossy()
            .to_string()
    );
}

#[test]
fn agent_run_amorphous_bulk_records_port_class_bias() {
    let coords_in = write_training_oligomer("agent_polymer_port_class_coords.pdb");
    let topology = write_prmtop("agent_polymer_port_class.prmtop", 1.0);
    let topology_graph = write_linear_topology_graph_with_port_class(
        "agent_polymer_port_class.topology.json",
        "reactive_end",
    );
    let build_manifest = write_polymer_build_manifest_with_topology_graph_request(
        "agent_polymer_port_class_build.json",
        &coords_in,
        &topology,
        &topology_graph,
        "build-005",
        "linear_homopolymer",
        3,
    );
    let coords_out = temp_path("agent_polymer_port_class_out.pdb");
    let manifest_out = temp_path("agent_polymer_port_class_manifest.json");

    let payload = json!({
        "version": "warp-pack.agent.v1",
        "run_id": "polymer-port-class-run",
        "polymer_build": {
            "build_manifest": build_manifest.to_string_lossy(),
        },
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": 42.0, "shape": "cubic"},
            "solvent": {"mode": "none"},
            "ions": {"neutralize": false, "cation": "Na+", "anion": "Cl-"},
            "morphology": {"mode": "amorphous_bulk"},
        },
        "outputs": {
            "coordinates": coords_out.to_string_lossy(),
            "manifest": manifest_out.to_string_lossy(),
        },
    });

    let (exit_code, envelope) = warp_pack::agent::run_request_json(
        &serde_json::to_string(&payload).expect("serialize payload"),
        false,
    );
    assert_eq!(
        exit_code,
        0,
        "{}",
        serde_json::to_string_pretty(&envelope).expect("envelope")
    );

    let manifest_value: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_out).expect("read manifest"))
            .expect("parse manifest");
    let hints = manifest_value["engine_decisions"]["details"]["morphology_policy"]
        ["component_packing_hints"]
        .as_array()
        .expect("packing hints");
    assert_eq!(hints.len(), 1);
    assert_eq!(hints[0]["region_policy"], "port_class_shell_faces");
    assert_eq!(hints[0]["port_class_policy"], "class_axis_partitioned");
    assert_eq!(hints[0]["port_classes"], json!(["reactive_end"]));
    assert_eq!(hints[0]["cap_state"], "open_ports_present");
    assert_eq!(
        manifest_value["component_inventory"][0]["topology_graph_summary"]["port_class_count"],
        1
    );
}

#[test]
fn agent_run_amorphous_bulk_records_mixed_architecture_cohort() {
    let linear_coords = write_training_oligomer("agent_mixed_linear_coords.pdb");
    let linear_topology = write_prmtop("agent_mixed_linear.prmtop", 1.0);
    let linear_graph = write_linear_topology_graph("agent_mixed_linear.topology.json");
    let linear_manifest = write_polymer_build_manifest_with_topology_graph_request(
        "agent_mixed_linear_build.json",
        &linear_coords,
        &linear_topology,
        &linear_graph,
        "build-003",
        "linear_homopolymer",
        3,
    );

    let branched_coords = write_branched_training_oligomer("agent_mixed_branched_coords.pdb");
    let branched_topology = write_prmtop("agent_mixed_branched.prmtop", 1.0);
    let branched_graph = write_branched_topology_graph("agent_mixed_branched.topology.json");
    let branched_manifest = write_polymer_build_manifest_with_topology_graph_request(
        "agent_mixed_branched_build.json",
        &branched_coords,
        &branched_topology,
        &branched_graph,
        "build-004",
        "branched_polymer",
        4,
    );

    let coords_out = temp_path("agent_mixed_arch_out.pdb");
    let manifest_out = temp_path("agent_mixed_arch_manifest.json");

    let payload = json!({
        "version": "warp-pack.agent.v1",
        "run_id": "mixed-architecture-run",
        "components": [
            {
                "name": "linear_chain",
                "count": 1,
                "source": {
                    "kind": "polymer_build",
                    "polymer_build": {"build_manifest": linear_manifest.to_string_lossy()}
                }
            },
            {
                "name": "branched_chain",
                "count": 1,
                "source": {
                    "kind": "polymer_build",
                    "polymer_build": {"build_manifest": branched_manifest.to_string_lossy()}
                }
            }
        ],
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": 56.0, "shape": "cubic"},
            "solvent": {"mode": "none"},
            "ions": {"neutralize": false, "cation": "Na+", "anion": "Cl-"},
            "morphology": {"mode": "amorphous_bulk"},
        },
        "outputs": {
            "coordinates": coords_out.to_string_lossy(),
            "manifest": manifest_out.to_string_lossy(),
        },
    });

    let (exit_code, envelope) = warp_pack::agent::run_request_json(
        &serde_json::to_string(&payload).expect("serialize payload"),
        false,
    );
    assert_eq!(
        exit_code,
        0,
        "{}",
        serde_json::to_string_pretty(&envelope).expect("envelope")
    );

    let manifest_value: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_out).expect("read manifest"))
            .expect("parse manifest");
    let morphology = &manifest_value["engine_decisions"]["details"]["morphology_policy"];
    assert_eq!(
        morphology["graph_architecture_cohort"]["distinct_architecture_count"],
        2
    );
    let hints = morphology["component_packing_hints"]
        .as_array()
        .expect("packing hints");
    assert_eq!(hints.len(), 2);
    assert!(hints
        .iter()
        .all(|hint| hint["cohort_policy"] == "mixed_architecture_partitioned"));
    assert!(hints
        .iter()
        .any(|hint| hint["architecture"] == "linear"
            && hint["region_policy"] == "open_port_shell_faces"));
    assert!(hints.iter().any(|hint| hint["architecture"] == "branched"
        && hint["region_policy"] == "port_class_access_bands"));
    assert!(hints
        .iter()
        .any(|hint| hint["port_class_policy"] == "class_axis_partitioned"));
    assert!(hints
        .iter()
        .any(|hint| hint["port_classes"] == json!(["reactive_end"])));
}

#[test]
fn agent_run_fixture_port_caps_handoff_records_applied_caps() {
    let (_coords_in, build_manifest, _charge_manifest, _topology_graph) = run_fixture_polymer_build(
        "pmma_port_caps",
        "fixture-pack-port-caps-001",
        json!({
            "mode": "polymer_graph",
            "graph_root": "g1",
            "graph_nodes": [{"id": "g1", "token": "M2C"}],
            "graph_edges": [],
            "termini": {"head": "default", "tail": "default"},
            "stereochemistry": {"mode": "inherit"}
        }),
        json!({
            "conformation_mode": "random_walk",
            "seed": 72
        }),
    );
    let coords_out = temp_path("fixture_pack_port_caps_out.pdb");
    let manifest_out = temp_path("fixture_pack_port_caps_manifest.json");

    let payload = json!({
        "version": "warp-pack.agent.v1",
        "run_id": "fixture-pack-port-caps-run",
        "polymer_build": {
            "build_manifest": build_manifest.to_string_lossy(),
        },
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": 42.0, "shape": "cubic"},
            "solvent": {"mode": "none"},
            "ions": {"neutralize": false, "cation": "Na+", "anion": "Cl-"},
            "morphology": {"mode": "amorphous_bulk"},
        },
        "outputs": {
            "coordinates": coords_out.to_string_lossy(),
            "manifest": manifest_out.to_string_lossy(),
        },
    });

    let (exit_code, envelope) = warp_pack::agent::run_request_json(
        &serde_json::to_string(&payload).expect("serialize payload"),
        false,
    );
    assert_eq!(
        exit_code,
        0,
        "{}",
        serde_json::to_string_pretty(&envelope).expect("envelope")
    );

    let manifest_value: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_out).expect("read manifest"))
            .expect("parse manifest");
    assert_eq!(
        manifest_value["component_inventory"][0]["topology_graph_summary"]["applied_cap_count"],
        2
    );
    assert_eq!(
        manifest_value["component_inventory"][0]["topology_graph_summary"]["port_class_count"],
        1
    );
    let hints = manifest_value["engine_decisions"]["details"]["morphology_policy"]
        ["component_packing_hints"]
        .as_array()
        .expect("packing hints");
    assert_eq!(hints.len(), 1);
    assert_eq!(hints[0]["cap_state"], "capped_ports_only");
    assert_eq!(hints[0]["port_class_policy"], "class_axis_partitioned");
    assert!(manifest_value["artifact_digests"]["coordinates"]
        .as_str()
        .expect("coordinates digest")
        .starts_with("sha256:"));
}

#[test]
fn agent_run_fixture_branched_mix_handoff_records_mixed_fixture_architecture() {
    let (_linear_coords, linear_manifest, _linear_charge, _linear_graph) =
        run_fixture_polymer_build(
            "pmma_motif",
            "fixture-pack-linear-001",
            json!({
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
            }),
            json!({
                "conformation_mode": "random_walk",
                "seed": 41
            }),
        );
    let (_branched_coords, branched_manifest, _branched_charge, _branched_graph) =
        run_fixture_polymer_build(
            "pmma_branched_mix",
            "fixture-pack-branched-001",
            json!({
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
            }),
            json!({
                "conformation_mode": "aligned",
                "alignment_axis": "z",
                "seed": 61
            }),
        );
    let coords_out = temp_path("fixture_pack_mixed_out.pdb");
    let manifest_out = temp_path("fixture_pack_mixed_manifest.json");

    let payload = json!({
        "version": "warp-pack.agent.v1",
        "run_id": "fixture-pack-mixed-run",
        "components": [
            {
                "name": "fixture_linear",
                "count": 1,
                "source": {
                    "kind": "polymer_build",
                    "polymer_build": {"build_manifest": linear_manifest.to_string_lossy()}
                }
            },
            {
                "name": "fixture_branched",
                "count": 1,
                "source": {
                    "kind": "polymer_build",
                    "polymer_build": {"build_manifest": branched_manifest.to_string_lossy()}
                }
            }
        ],
        "environment": {
            "box": {"mode": "fixed_size", "size_angstrom": 56.0, "shape": "cubic"},
            "solvent": {"mode": "none"},
            "ions": {"neutralize": false, "cation": "Na+", "anion": "Cl-"},
            "morphology": {"mode": "amorphous_bulk"},
        },
        "outputs": {
            "coordinates": coords_out.to_string_lossy(),
            "manifest": manifest_out.to_string_lossy(),
        },
    });

    let (exit_code, envelope) = warp_pack::agent::run_request_json(
        &serde_json::to_string(&payload).expect("serialize payload"),
        false,
    );
    assert_eq!(
        exit_code,
        0,
        "{}",
        serde_json::to_string_pretty(&envelope).expect("envelope")
    );

    let manifest_value: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_out).expect("read manifest"))
            .expect("parse manifest");
    let morphology = &manifest_value["engine_decisions"]["details"]["morphology_policy"];
    let inventory = manifest_value["component_inventory"]
        .as_array()
        .expect("component inventory");
    assert!(inventory
        .iter()
        .any(|item| item["topology_graph_summary"]["build_mode"] == "polymer_graph"));
    assert!(inventory
        .iter()
        .any(|item| item["topology_graph_summary"]["build_mode"] == "branched_polymer"));
    let hints = morphology["component_packing_hints"]
        .as_array()
        .expect("packing hints");
    assert_eq!(hints.len(), 2);
    assert!(hints
        .iter()
        .any(|hint| hint["graph_summary"]["build_mode"] == "branched_polymer"));
    assert!(manifest_value["artifact_digests"]["coordinates"]
        .as_str()
        .expect("coordinates digest")
        .starts_with("sha256:"));
}
