use super::*;

fn source_polymer_pdb() -> String {
    [
        "ATOM      1 C1   STA A   1       0.000   0.000   0.000  1.00  0.00           C",
        "ATOM      2 C2   STA A   1       1.400   0.000   0.000  1.00  0.00           C",
        "ATOM      3 C1   MID A   2       2.800   0.000   0.000  1.00  0.00           C",
        "ATOM      4 C2   MID A   2       4.200   0.000   0.000  1.00  0.00           C",
        "ATOM      5 C1   END A   3       5.600   0.000   0.000  1.00  0.00           C",
        "ATOM      6 C2   END A   3       7.000   0.000   0.000  1.00  0.00           C",
        "CONECT    1    2",
        "CONECT    2    3",
        "CONECT    3    4",
        "CONECT    4    5",
        "CONECT    5    6",
        "END",
        "",
    ]
    .join("\n")
}

fn paa_like_polymer_pdb() -> String {
    [
        "ATOM      1 C1   STA A   1       0.000   0.000   0.000  1.00  0.00           C",
        "ATOM      2 C2   STA A   1       1.400   0.000   0.000  1.00  0.00           C",
        "ATOM      3 C3   STA A   1       2.800   0.000   0.000  1.00  0.00           C",
        "ATOM      4 O1   STA A   1       3.500   0.900   0.000  1.00  0.00           O",
        "ATOM      5 O2   STA A   1       3.500  -0.900   0.000  1.00  0.00           O",
        "ATOM      6 C1   MID A   2       4.200   0.000   0.000  1.00  0.00           C",
        "ATOM      7 C2   MID A   2       5.600   0.000   0.000  1.00  0.00           C",
        "ATOM      8 C3   MID A   2       7.000   0.000   0.000  1.00  0.00           C",
        "ATOM      9 O1   MID A   2       7.700   0.900   0.000  1.00  0.00           O",
        "ATOM     10 O2   MID A   2       7.700  -0.900   0.000  1.00  0.00           O",
        "ATOM     11 C1   END A   3       8.400   0.000   0.000  1.00  0.00           C",
        "ATOM     12 C2   END A   3       9.800   0.000   0.000  1.00  0.00           C",
        "ATOM     13 C3   END A   3      11.200   0.000   0.000  1.00  0.00           C",
        "ATOM     14 O1   END A   3      11.900   0.900   0.000  1.00  0.00           O",
        "ATOM     15 O2   END A   3      11.900  -0.900   0.000  1.00  0.00           O",
        "CONECT    1    2",
        "CONECT    2    3    6",
        "CONECT    3    4    5",
        "CONECT    6    7",
        "CONECT    7    8   11",
        "CONECT    8    9   10",
        "CONECT   11   12",
        "CONECT   12   13",
        "CONECT   13   14   15",
        "END",
        "",
    ]
    .join("\n")
}

fn pes_like_source_pdb() -> String {
    [
        "ATOM      1 C1   PES A   1       1.400   0.000   0.000  1.00  0.00           C",
        "ATOM      2 C2   PES A   1       0.700   1.212   0.000  1.00  0.00           C",
        "ATOM      3 C3   PES A   1      -0.700   1.212   0.000  1.00  0.00           C",
        "ATOM      4 C4   PES A   1      -1.400   0.000   0.000  1.00  0.00           C",
        "ATOM      5 C5   PES A   1      -0.700  -1.212   0.000  1.00  0.00           C",
        "ATOM      6 C6   PES A   1       0.700  -1.212   0.000  1.00  0.00           C",
        "ATOM      7 C7   PES A   1       6.400   0.000   0.000  1.00  0.00           C",
        "ATOM      8 C8   PES A   1       5.700   1.212   0.000  1.00  0.00           C",
        "ATOM      9 C9   PES A   1       4.300   1.212   0.000  1.00  0.00           C",
        "ATOM     10 C10  PES A   1       3.600   0.000   0.000  1.00  0.00           C",
        "ATOM     11 C11  PES A   1       4.300  -1.212   0.000  1.00  0.00           C",
        "ATOM     12 C12  PES A   1       5.700  -1.212   0.000  1.00  0.00           C",
        "ATOM     13 S1   PES A   1       2.500   0.900   0.000  1.00  0.00           S",
        "ATOM     14 O1   PES A   1       2.500   2.100   0.000  1.00  0.00           O",
        "ATOM     15 O2   PES A   1       2.500   0.900   1.200  1.00  0.00           O",
        "ATOM     16 O3   PES A   1       2.500  -1.000   0.000  1.00  0.00           O",
        "CONECT    1    2    6   13",
        "CONECT    2    3",
        "CONECT    3    4",
        "CONECT    4    5   16",
        "CONECT    5    6",
        "CONECT    7    8   12   13",
        "CONECT    8    9",
        "CONECT    9   10",
        "CONECT   10   11   16",
        "CONECT   11   12",
        "CONECT   13   14   15",
        "END",
        "",
    ]
    .join("\n")
}

fn benzene_structure_pdb_without_conect() -> String {
    [
        "ATOM      1 C1   BEN A   1       1.400   0.000   0.000  1.00  0.00           C",
        "ATOM      2 C2   BEN A   1       0.700   1.212   0.000  1.00  0.00           C",
        "ATOM      3 C3   BEN A   1      -0.700   1.212   0.000  1.00  0.00           C",
        "ATOM      4 C4   BEN A   1      -1.400   0.000   0.000  1.00  0.00           C",
        "ATOM      5 C5   BEN A   1      -0.700  -1.212   0.000  1.00  0.00           C",
        "ATOM      6 C6   BEN A   1       0.700  -1.212   0.000  1.00  0.00           C",
        "END",
        "",
    ]
    .join("\n")
}

fn distorted_benzene_structure_pdb_without_conect() -> String {
    [
        "ATOM      1 C1   BEN A   1       1.700   0.000   0.000  1.00  0.00           C",
        "ATOM      2 C2   BEN A   1       0.850   1.472   0.000  1.00  0.00           C",
        "ATOM      3 C3   BEN A   1      -0.850   1.472   0.000  1.00  0.00           C",
        "ATOM      4 C4   BEN A   1      -1.700   0.000   0.000  1.00  0.00           C",
        "ATOM      5 C5   BEN A   1      -0.850  -1.472   0.000  1.00  0.00           C",
        "ATOM      6 C6   BEN A   1       0.850  -1.472   0.000  1.00  0.00           C",
        "END",
        "",
    ]
    .join("\n")
}

fn source_request(
    name: &str,
    source_path: &Path,
    out_dir: &Path,
    mode: &str,
    template: Option<String>,
) -> CgRequest {
    CgRequest {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        name: name.to_string(),
        smiles: None,
        repeat_smiles: None,
        source: Some(CgSource {
            kind: "coordinates_topology".to_string(),
            path: None,
            coordinates: Some(source_path.to_string_lossy().to_string()),
            topology: Some(source_path.to_string_lossy().to_string()),
            charge_manifest: None,
            trajectory: None,
            target_selection: None,
            selection: None,
            format: Some("pdb".to_string()),
            topology_format: Some("pdb".to_string()),
        }),
        bonding: None,
        chemistry_hints: Vec::new(),
        chemistry_policy: None,
        polymer: None,
        mapping: Some(CgMappingRequest {
            mode: mode.to_string(),
            strategy: Some("polymer_residue_graph".to_string()),
            target_bead_size: Some(4),
            preserve_functional_groups: Some(true),
            template,
            template_policy: None,
            expected_beads_per_role: std::collections::BTreeMap::new(),
            on_bead_count_mismatch: None,
            ndx: None,
            repeat_unit_hint: Some("PAA".to_string()),
            terminal_aware: Some(true),
        }),
        topology: None,
        trajectory_source: None,
        reference_source: None,
        forcefield: None,
        optimization: None,
        output: CgOutputRequest {
            out_dir: out_dir.to_string_lossy().to_string(),
            mapped_trajectory: None,
            write_mapping_json: true,
            write_topology_itp: true,
            write_topology_top: true,
            write_cg_pdb: true,
            cg_pdb: None,
            write_bonded_parameter_map: true,
        },
    }
}

#[path = "agent_test_parts/candidate_extraction_validation.rs"]
mod candidate_extraction_validation;
#[path = "agent_test_parts/ndx_mapping.rs"]
mod ndx_mapping;
#[path = "agent_test_parts/precomputed_reference.rs"]
mod precomputed_reference;
#[path = "agent_test_parts/reference_targets.rs"]
mod reference_targets;
#[path = "agent_test_parts/request_examples_and_sources.rs"]
mod request_examples_and_sources;
#[path = "agent_test_parts/validation_and_outputs.rs"]
mod validation_and_outputs;
