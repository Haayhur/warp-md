use super::*;

#[test]
fn coordinate_less_solutes_reuse_standard_solvent_registry_templates() {
    let temp = tempfile::tempdir().unwrap();
    let gro = temp.path().join("standard_solvent_solutes.gro");
    let topology = temp.path().join("standard_solvent_solutes.top");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {
            "box_size_angstrom": [80.0, 80.0, 80.0]
        },
        "membranes": [],
        "solutes": [
            {"name": "DMSO", "count": 1},
            {"name": "HEX", "count": 1}
        ],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {
            "gro": gro,
            "topology": topology,
            "manifest": temp.path().join("manifest.json")
        }
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["inserted_counts"]["DMSO"], 1);
    assert_eq!(value["summary"]["inserted_counts"]["HEX"], 1);
    assert_eq!(value["summary"]["bead_count"], 4);
    assert_eq!(value["charge"]["net_charge_before_neutralization_e"], 0.0);

    let dmso_atoms = read_gro_residue_atoms(&gro, "DMSO");
    assert_eq!(dmso_atoms.len(), 2);
    assert_eq!(dmso_atoms[0].0, "S1");
    assert_eq!(dmso_atoms[1].0, "O2");
    assert!((dmso_atoms[1].1[0] - dmso_atoms[0].1[0] - 3.0).abs() < 0.02);

    let hex_atoms = read_gro_residue_atoms(&gro, "HEX");
    assert_eq!(hex_atoms.len(), 2);
    assert_eq!(hex_atoms[0].0, "C1");
    assert_eq!(hex_atoms[1].0, "C2");
    assert!((hex_atoms[1].1[0] - hex_atoms[0].1[0] - 4.05).abs() < 0.02);

    let topology_text = std::fs::read_to_string(&topology).unwrap();
    assert_eq!(topology_molecule_charge_sum(&topology_text, "DMSO"), 0.0);
    assert_eq!(topology_molecule_charge_sum(&topology_text, "HEX"), 0.0);
    assert!(topology_text.contains("1     2     1    0.30000   8000.000"));
    assert!(topology_text.contains("1     2     1    0.40500   5000.000"));
}

#[test]
fn coordinate_less_solvent_registry_charge_mismatch_is_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [80.0, 80.0, 80.0]},
        "membranes": [],
        "solutes": [
            {"name": "DMSO", "count": 1, "net_charge_e": 1.0}
        ],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {
            "manifest": temp.path().join("manifest.json")
        }
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 2);
    assert!(value["error"]["message"]
        .as_str()
        .unwrap()
        .contains("does not match solvent-library-derived charge"));
}
