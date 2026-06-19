// Generated during contract migration. Rust-native catalog literals.
use super::*;

fn field(
    field_type: &str,
    semantic_type: &str,
    description: &str,
    default: Option<serde_json::Value>,
    minimum: Option<f64>,
    unit: Option<&str>,
) -> WarpMdFieldSpec {
    WarpMdFieldSpec {
        field_type: field_type.into(),
        semantic_type: semantic_type.into(),
        description: Some(description.into()),
        default,
        minimum,
        maximum: None,
        unit: unit.map(str::to_string),
        choices: None,
    }
}

fn lipid_fields(
    fields: &[(
        &str,
        &str,
        &str,
        &str,
        Option<serde_json::Value>,
        Option<f64>,
        Option<&str>,
    )],
) -> std::collections::BTreeMap<String, WarpMdFieldSpec> {
    fields
        .iter()
        .map(
            |(name, field_type, semantic_type, description, default, minimum, unit)| {
                (
                    (*name).into(),
                    field(
                        field_type,
                        semantic_type,
                        description,
                        default.clone(),
                        *minimum,
                        *unit,
                    ),
                )
            },
        )
        .collect()
}

fn lipid_matrix_contract(
    name: &str,
    aliases: &[&str],
    description: &str,
    required_fields: &[&str],
    optional_fields: &[&str],
    fields: &[(
        &str,
        &str,
        &str,
        &str,
        Option<serde_json::Value>,
        Option<f64>,
        Option<&str>,
    )],
    output_kind: &str,
    output_fields: &[&str],
    example: serde_json::Value,
) -> WarpMdAnalysisContract {
    WarpMdAnalysisContract {
        name: name.into(),
        aliases: aliases.iter().map(|value| (*value).into()).collect(),
        description: description.into(),
        required_fields: required_fields
            .iter()
            .map(|value| (*value).into())
            .collect(),
        optional_fields: optional_fields
            .iter()
            .map(|value| (*value).into())
            .collect(),
        fields: lipid_fields(fields),
        outputs: vec![WarpMdArtifactSpec {
            kind: output_kind.into(),
            format: "npz".into(),
            fields: output_fields.iter().map(|value| (*value).into()).collect(),
            description: Some(description.into()),
        }],
        tags: vec!["membrane".into(), "lipid".into()],
        examples: vec![example],
    }
}

fn native_matrix_contract(
    name: &str,
    aliases: &[&str],
    description: &str,
    required_fields: &[&str],
    optional_fields: &[&str],
    fields: &[(
        &str,
        &str,
        &str,
        &str,
        Option<serde_json::Value>,
        Option<f64>,
        Option<&str>,
    )],
    output_kind: &str,
    output_fields: &[&str],
    tags: &[&str],
    example: serde_json::Value,
) -> WarpMdAnalysisContract {
    WarpMdAnalysisContract {
        name: name.into(),
        aliases: aliases.iter().map(|value| (*value).into()).collect(),
        description: description.into(),
        required_fields: required_fields
            .iter()
            .map(|value| (*value).into())
            .collect(),
        optional_fields: optional_fields
            .iter()
            .map(|value| (*value).into())
            .collect(),
        fields: lipid_fields(fields),
        outputs: vec![WarpMdArtifactSpec {
            kind: output_kind.into(),
            format: "npz".into(),
            fields: output_fields.iter().map(|value| (*value).into()).collect(),
            description: Some(description.into()),
        }],
        tags: tags.iter().map(|value| (*value).into()).collect(),
        examples: vec![example],
    }
}

pub(crate) fn warp_md_agent_contract_catalog_native() -> WarpMdContractCatalog {
    WarpMdContractCatalog {
        schema_version: "warp-md.agent.v1".into(),
        analysis_shared_fields: vec!["out".into(), "device".into(), "chunk_frames".into()],
        cli_to_analysis: std::collections::BTreeMap::from([
            ("bond-angle-distribution".into(), "bond_angle_distribution".into()),
            ("bond-length-distribution".into(), "bond_length_distribution".into()),
            ("bond_angle_distribution".into(), "bond_angle_distribution".into()),
            ("bond_length_distribution".into(), "bond_length_distribution".into()),
            ("bondi-ffv".into(), "bondi_ffv".into()),
            ("bondi_ffv".into(), "bondi_ffv".into()),
            ("chain-rg".into(), "chain_rg".into()),
            ("chain_rg".into(), "chain_rg".into()),
            ("conductivity".into(), "conductivity".into()),
            ("contour-length".into(), "contour_length".into()),
            ("contour_length".into(), "contour_length".into()),
            ("density".into(), "density".into()),
            ("dielectric".into(), "dielectric".into()),
            ("diffusion".into(), "diffusion".into()),
            ("dipole-alignment".into(), "dipole_alignment".into()),
            ("dipole_alignment".into(), "dipole_alignment".into()),
            ("docking".into(), "docking".into()),
            ("drid".into(), "drid".into()),
            ("dssp".into(), "dssp".into()),
            ("end-to-end".into(), "end_to_end".into()),
            ("end_to_end".into(), "end_to_end".into()),
            ("equipartition".into(), "equipartition".into()),
            ("ffv".into(), "bondi_ffv".into()),
            ("fractional-free-volume".into(), "bondi_ffv".into()),
            ("free-volume".into(), "free_volume".into()),
            ("free-volume-grid".into(), "free_volume".into()),
            ("free_volume".into(), "free_volume".into()),
            ("gist".into(), "gist".into()),
            ("hbond".into(), "hbond".into()),
            ("hydrophobic-defects".into(), "hydrophobic_defects".into()),
            ("hydrophobic_defects".into(), "hydrophobic_defects".into()),
            ("membrane-packing-defects".into(), "hydrophobic_defects".into()),
            ("ion-pair-correlation".into(), "ion_pair_correlation".into()),
            ("ion_pair_correlation".into(), "ion_pair_correlation".into()),
            ("jcoupling".into(), "jcoupling".into()),
            ("lipid-area".into(), "lipid_area".into()),
            ("lipid_area".into(), "lipid_area".into()),
            ("lipid-curved-leaflets".into(), "lipid_curved_leaflets".into()),
            ("lipid_curved_leaflets".into(), "lipid_curved_leaflets".into()),
            ("lipid-flip-flop".into(), "lipid_flip_flop".into()),
            ("lipid_flip_flop".into(), "lipid_flip_flop".into()),
            ("lipid-largest-cluster".into(), "lipid_largest_cluster".into()),
            ("lipid_largest_cluster".into(), "lipid_largest_cluster".into()),
            ("lipid-leaflets".into(), "lipid_leaflets".into()),
            ("lipid_leaflets".into(), "lipid_leaflets".into()),
            ("lipid-membrane-thickness".into(), "lipid_membrane_thickness".into()),
            ("lipid_membrane_thickness".into(), "lipid_membrane_thickness".into()),
            ("lipid-msd".into(), "lipid_msd".into()),
            ("lipid_msd".into(), "lipid_msd".into()),
            ("lipid-neighbour-matrix".into(), "lipid_neighbour_matrix".into()),
            ("lipid_neighbour_matrix".into(), "lipid_neighbour_matrix".into()),
            ("lipid-neighbours".into(), "lipid_neighbours".into()),
            ("lipid_neighbours".into(), "lipid_neighbours".into()),
            ("lipid-registration".into(), "lipid_registration".into()),
            ("lipid_registration".into(), "lipid_registration".into()),
            ("lipid-scc".into(), "lipid_scc".into()),
            ("lipid_scc".into(), "lipid_scc".into()),
            ("lipid-z-angles".into(), "lipid_z_angles".into()),
            ("lipid_z_angles".into(), "lipid_z_angles".into()),
            ("lipid-z-positions".into(), "lipid_z_positions".into()),
            ("lipid_z_positions".into(), "lipid_z_positions".into()),
            ("lipid-z-thickness".into(), "lipid_z_thickness".into()),
            ("lipid_z_thickness".into(), "lipid_z_thickness".into()),
            ("linear-density".into(), "lineardensity".into()),
            ("lineardensity".into(), "lineardensity".into()),
            ("molsurf".into(), "molsurf".into()),
            ("msd".into(), "msd".into()),
            ("native-contacts".into(), "native_contacts".into()),
            ("native_contacts".into(), "native_contacts".into()),
            ("nematic-order".into(), "nematic_order".into()),
            ("nematic_order".into(), "nematic_order".into()),
            ("nmr".into(), "nmr".into()),
            ("pca".into(), "pca".into()),
            ("pairdist".into(), "pairdist".into()),
            ("persistence-length".into(), "persistence_length".into()),
            ("persistence_length".into(), "persistence_length".into()),
            ("projection".into(), "projection".into()),
            ("rdf".into(), "rdf".into()),
            ("rg".into(), "rg".into()),
            ("rmsd".into(), "rmsd".into()),
            ("rmsf".into(), "rmsf".into()),
            ("rotacf".into(), "rotacf".into()),
            ("runningavg".into(), "runningavg".into()),
            ("shape-descriptors".into(), "shape_descriptors".into()),
            ("shape_descriptors".into(), "shape_descriptors".into()),
            ("structure-factor".into(), "structure_factor".into()),
            ("structure_factor".into(), "structure_factor".into()),
            ("surf".into(), "surf".into()),
            ("tordiff".into(), "tordiff".into()),
            ("volmap".into(), "volmap".into()),
            ("voxel-free-volume".into(), "free_volume".into()),
            ("water-count".into(), "water_count".into()),
            ("water_count".into(), "water_count".into()),
            ("watershell".into(), "watershell".into()),
        ]),
        analyses: vec![
        WarpMdAnalysisContract {
            name: "bond_angle_distribution".into(),
            aliases: vec!["bond-angle".into()],
            description: "Bond angle distribution".into(),
            required_fields: vec!["selection".into(), "bins".into()],
            optional_fields: vec!["degrees".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Atom selection for angle analysis".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "bins".into(),
                    WarpMdFieldSpec {
                        field_type: "integer".into(),
                        semantic_type: "integer".into(),
                        description: Some("Number of angle bins".into()),
                        default: None,
                        minimum: Some(1.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "degrees".into(),
                    WarpMdFieldSpec {
                        field_type: "boolean".into(),
                        semantic_type: "boolean".into(),
                        description: Some("Use degrees (vs radians)".into()),
                        default: Some(serde_json::json!(true)),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "histogram".into(),
                    format: "npz".into(),
                    fields: vec!["angle".into(), "probability".into()],
                    description: Some("Bond angle distribution".into()),
                }
            ],
            tags: vec!["structural".into(), "bond".into()],
            examples: vec![
                serde_json::json!({"name": "bond_angle_distribution", "selection": "polymer", "bins": 180})
            ],
        },
        WarpMdAnalysisContract {
            name: "bond_length_distribution".into(),
            aliases: vec!["bond-length".into()],
            description: "Bond length distribution".into(),
            required_fields: vec!["selection".into(), "bins".into(), "r_max".into()],
            optional_fields: vec![],
            fields: std::collections::BTreeMap::from([
                (
                    "selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Atom selection for bond analysis".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "bins".into(),
                    WarpMdFieldSpec {
                        field_type: "integer".into(),
                        semantic_type: "integer".into(),
                        description: Some("Number of distance bins".into()),
                        default: None,
                        minimum: Some(1.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "r_max".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Maximum bond distance".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: Some("nm".into()),
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "histogram".into(),
                    format: "npz".into(),
                    fields: vec!["distance_nm".into(), "probability".into()],
                    description: Some("Bond length distribution".into()),
                }
            ],
            tags: vec!["structural".into(), "bond".into()],
            examples: vec![
                serde_json::json!({"name": "bond_length_distribution", "selection": "polymer", "bins": 100, "r_max": 0.2})
            ],
        },
        WarpMdAnalysisContract {
            name: "bondi_ffv".into(),
            aliases: vec!["bondi-ffv".into(), "ffv".into(), "fractional-free-volume".into()],
            description: "GROMACS-style free-volume Monte Carlo with Bondi radii. Reports raw free-volume fraction and the Lourenco/GROMACS FFV convention FFV = 1 - scale * (1 - free_volume_fraction).".into(),
            required_fields: vec!["selection".into()],
            optional_fields: vec!["bondi_scale".into(), "probe_radius".into(), "seed".into(), "ninsert_per_nm3".into(), "length_scale".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Atoms treated as excluded volume during probe insertion".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "bondi_scale".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Homogeneity scale factor in FFV = 1 - scale * (1 - free_volume_fraction)".into()),
                        default: Some(serde_json::json!(1.3)),
                        minimum: Some(0.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "probe_radius".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Probe radius in Angstroms. 0.0 yields true free volume.".into()),
                        default: Some(serde_json::json!(0.0)),
                        minimum: Some(0.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "seed".into(),
                    WarpMdFieldSpec {
                        field_type: "integer".into(),
                        semantic_type: "integer".into(),
                        description: Some("Deterministic Monte Carlo seed".into()),
                        default: Some(serde_json::json!(0)),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "ninsert_per_nm3".into(),
                    WarpMdFieldSpec {
                        field_type: "integer".into(),
                        semantic_type: "integer".into(),
                        description: Some("Probe insertions per cubic nanometer".into()),
                        default: Some(serde_json::json!(1000)),
                        minimum: Some(1.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "length_scale".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Coordinate/box length scale".into()),
                        default: Some(serde_json::json!(1.0)),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "table".into(),
                    format: "json".into(),
                    fields: vec!["time".into(), "total_volume_a3".into(), "vdw_volume_a3".into(), "raw_free_volume_a3".into(), "raw_free_volume_fraction".into(), "fractional_free_volume".into(), "density_g_cm3".into(), "molar_mass_dalton".into(), "bondi_scale".into()],
                    description: Some("Per-frame Bondi/Lourenco free-volume metrics".into()),
                }
            ],
            tags: vec!["polymer".into(), "free-volume".into(), "bondi".into(), "gromacs-parity".into()],
            examples: vec![
                serde_json::json!({"name": "bondi_ffv", "selection": "not name QQQQ", "bondi_scale": 1.3, "probe_radius": 0.0, "seed": -1107428613, "ninsert_per_nm3": 1000})
            ],
        },
        WarpMdAnalysisContract {
            name: "chain_rg".into(),
            aliases: vec!["chain-rg".into()],
            description: "Radius of gyration per chain/molecule".into(),
            required_fields: vec!["selection".into()],
            optional_fields: vec![],
            fields: std::collections::BTreeMap::from([
                (
                    "selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Polymer atom selection".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "table".into(),
                    format: "npz".into(),
                    fields: vec!["chain_id".into(), "rg_nm".into()],
                    description: Some("Rg per chain".into()),
                }
            ],
            tags: vec!["structural".into(), "polymer".into()],
            examples: vec![
                serde_json::json!({"name": "chain_rg", "selection": "polymer"})
            ],
        },
        WarpMdAnalysisContract {
            name: "conductivity".into(),
            aliases: vec!["electrical-conductivity".into()],
            description: "Electrical conductivity via Einstein relation".into(),
            required_fields: vec!["selection".into(), "charges".into(), "temperature".into()],
            optional_fields: vec!["group_by".into(), "transference".into(), "length_scale".into(), "frame_decimation".into(), "dt_decimation".into(), "time_binning".into(), "lag_mode".into(), "max_lag".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Ion selection for conductivity".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "charges".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "charges".into(),
                        description: Some("Charge specification method".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: Some(vec!["by_atom".into(), "by_resname".into(), "by_name".into()]),
                    },
                ),
                (
                    "temperature".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Temperature in Kelvin".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: Some("K".into()),
                        choices: None,
                    },
                ),
                (
                    "transference".into(),
                    WarpMdFieldSpec {
                        field_type: "boolean".into(),
                        semantic_type: "boolean".into(),
                        description: Some("Compute transference numbers".into()),
                        default: Some(serde_json::json!(false)),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "timeseries".into(),
                    format: "npz".into(),
                    fields: vec!["lag_time_ps".into(), "conductivity_S_per_cm".into()],
                    description: Some("Conductivity vs lag time".into()),
                }
            ],
            tags: vec!["dynamic".into(), "electrostatics".into(), "transport".into()],
            examples: vec![
                serde_json::json!({"name": "conductivity", "selection": "resname Na or resname CL", "charges": "by_resname", "temperature": 300.0})
            ],
        },
        WarpMdAnalysisContract {
            name: "contour_length".into(),
            aliases: vec!["contour-length".into()],
            description: "Contour length (bond path length) for polymers".into(),
            required_fields: vec!["selection".into()],
            optional_fields: vec![],
            fields: std::collections::BTreeMap::from([
                (
                    "selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Polymer atom selection".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "timeseries".into(),
                    format: "npz".into(),
                    fields: vec!["time_ps".into(), "contour_nm".into()],
                    description: Some("Contour length vs time".into()),
                }
            ],
            tags: vec!["structural".into(), "polymer".into()],
            examples: vec![
                serde_json::json!({"name": "contour_length", "selection": "polymer"})
            ],
        },
        WarpMdAnalysisContract {
            name: "density".into(),
            aliases: vec!["number-density".into(), "mass-density".into()],
            description: "Density profile along a direction".into(),
            required_fields: vec![],
            optional_fields: vec!["mask".into(), "density_type".into(), "delta".into(), "direction".into(), "cutoff".into(), "center".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "mask".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "mask".into(),
                        description: Some("Atom mask".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "density_type".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "string".into(),
                        description: Some("Type of density (number, mass, charge)".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: Some(vec!["number".into(), "mass".into(), "charge".into()]),
                    },
                ),
                (
                    "delta".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Bin width".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "direction".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "string".into(),
                        description: Some("Direction for profile (x, y, z, or normal)".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "profile".into(),
                    format: "npz".into(),
                    fields: vec!["position".into(), "density".into()],
                    description: Some("Density vs position".into()),
                }
            ],
            tags: vec!["structural".into(), "spatial".into()],
            examples: vec![
                serde_json::json!({"name": "density", "mask": "resname SOL", "direction": "z"})
            ],
        },
        WarpMdAnalysisContract {
            name: "dielectric".into(),
            aliases: vec!["dielectric-constant".into()],
            description: "Dielectric constant from dipole fluctuations".into(),
            required_fields: vec!["selection".into(), "charges".into()],
            optional_fields: vec!["group_by".into(), "length_scale".into(), "temperature".into(), "make_whole".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Molecule selection for dielectric".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "charges".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "charges".into(),
                        description: Some("Charge specification method".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "temperature".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Simulation temperature in Kelvin".into()),
                        default: Some(serde_json::json!(300.0)),
                        minimum: Some(0.0),
                        maximum: None,
                        unit: Some("K".into()),
                        choices: None,
                    },
                ),
                (
                    "make_whole".into(),
                    WarpMdFieldSpec {
                        field_type: "boolean".into(),
                        semantic_type: "boolean".into(),
                        description: Some("Reconstruct grouped molecules across periodic boundaries".into()),
                        default: Some(serde_json::json!(true)),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "artifact".into(),
                    format: "npz".into(),
                    fields: vec!["dielectric_constant".into(), "dipole_moment_debye".into()],
                    description: Some("Dielectric constant and dipole moment".into()),
                }
            ],
            tags: vec!["electrostatics".into()],
            examples: vec![
                serde_json::json!({"name": "dielectric", "selection": "resname SOL", "charges": "by_resname"})
            ],
        },
        WarpMdAnalysisContract {
            name: "diffusion".into(),
            aliases: vec!["diffusion-coefficient".into()],
            description: "Diffusion coefficient from MSD slope".into(),
            required_fields: vec![],
            optional_fields: vec!["mask".into(), "tstep".into(), "individual".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "mask".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "mask".into(),
                        description: Some("Atom mask".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "tstep".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Time step in picoseconds".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "individual".into(),
                    WarpMdFieldSpec {
                        field_type: "boolean".into(),
                        semantic_type: "boolean".into(),
                        description: Some("Compute per-species diffusion".into()),
                        default: Some(serde_json::json!(false)),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "artifact".into(),
                    format: "npz".into(),
                    fields: vec!["diffusion_coefficient".into(), "msd_data".into()],
                    description: Some("Diffusion coefficient and MSD data".into()),
                }
            ],
            tags: vec!["dynamic".into(), "transport".into(), "diffusion".into()],
            examples: vec![
                serde_json::json!({"name": "diffusion", "mask": "resname SOL"})
            ],
        },
        WarpMdAnalysisContract {
            name: "dipole_alignment".into(),
            aliases: vec!["dipole-alignment".into()],
            description: "Dipole-dipole alignment correlation".into(),
            required_fields: vec!["selection".into(), "charges".into()],
            optional_fields: vec!["group_by".into(), "length_scale".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Molecule selection for dipole alignment".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "charges".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "charges".into(),
                        description: Some("Charge specification method".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "histogram".into(),
                    format: "npz".into(),
                    fields: vec!["cos_theta".into(), "probability".into()],
                    description: Some("Dipole alignment distribution".into()),
                }
            ],
            tags: vec!["electrostatics".into()],
            examples: vec![
                serde_json::json!({"name": "dipole_alignment", "selection": "resname SOL", "charges": "by_resname"})
            ],
        },
        WarpMdAnalysisContract {
            name: "docking".into(),
            aliases: vec!["docking-analysis".into()],
            description: "Docking pose analysis (protein-ligand interactions)".into(),
            required_fields: vec!["receptor_mask".into(), "ligand_mask".into()],
            optional_fields: vec!["close_contact_cutoff".into(), "hydrophobic_cutoff".into(), "hydrogen_bond_cutoff".into(), "clash_cutoff".into(), "salt_bridge_cutoff".into(), "halogen_bond_cutoff".into(), "metal_coordination_cutoff".into(), "cation_pi_cutoff".into(), "pi_pi_cutoff".into(), "hbond_min_angle_deg".into(), "donor_hydrogen_cutoff".into(), "allow_missing_hydrogen".into(), "length_scale".into(), "frame_indices".into(), "max_events_per_frame".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "receptor_mask".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "mask".into(),
                        description: Some("Receptor atom mask".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "ligand_mask".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "mask".into(),
                        description: Some("Ligand atom mask".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "close_contact_cutoff".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Close contact distance cutoff".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: Some("nm".into()),
                        choices: None,
                    },
                ),
                (
                    "hydrophobic_cutoff".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Hydrophobic interaction cutoff".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: Some("nm".into()),
                        choices: None,
                    },
                ),
                (
                    "hydrogen_bond_cutoff".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Hydrogen bond distance cutoff".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: Some("nm".into()),
                        choices: None,
                    },
                ),
                (
                    "clash_cutoff".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Clash detection cutoff".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: Some("nm".into()),
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "table".into(),
                    format: "json".into(),
                    fields: vec!["interaction_type".into(), "count".into(), "details".into()],
                    description: Some("Docking interaction summary".into()),
                }
            ],
            tags: vec!["docking".into(), "protein".into(), "structural".into()],
            examples: vec![
                serde_json::json!({"name": "docking", "receptor_mask": "protein", "ligand_mask": "resname LIG"})
            ],
        },
        WarpMdAnalysisContract {
            name: "dssp".into(),
            aliases: vec!["secondary-structure".into()],
            description: "Protein secondary structure assignment (DSSP)".into(),
            required_fields: vec![],
            optional_fields: vec!["mask".into(), "simplified".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "mask".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "mask".into(),
                        description: Some("Protein atom mask".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "simplified".into(),
                    WarpMdFieldSpec {
                        field_type: "boolean".into(),
                        semantic_type: "boolean".into(),
                        description: Some("Use simplified 3-state classification".into()),
                        default: Some(serde_json::json!(false)),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "table".into(),
                    format: "npz".into(),
                    fields: vec!["residue".into(), "structure".into()],
                    description: Some("Secondary structure per residue".into()),
                }
            ],
            tags: vec!["protein".into(), "structural".into()],
            examples: vec![
                serde_json::json!({"name": "dssp", "mask": "protein"})
            ],
        },
        WarpMdAnalysisContract {
            name: "end_to_end".into(),
            aliases: vec!["end-to-end-distance".into()],
            description: "End-to-end distance for polymers".into(),
            required_fields: vec!["selection".into()],
            optional_fields: vec![],
            fields: std::collections::BTreeMap::from([
                (
                    "selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Polymer atom selection".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "timeseries".into(),
                    format: "npz".into(),
                    fields: vec!["time_ps".into(), "end_to_end_nm".into()],
                    description: Some("End-to-end distance vs time".into()),
                }
            ],
            tags: vec!["structural".into(), "polymer".into()],
            examples: vec![
                serde_json::json!({"name": "end_to_end", "selection": "polymer"})
            ],
        },
        WarpMdAnalysisContract {
            name: "equipartition".into(),
            aliases: vec!["kinetic-energy".into(), "ke-distribution".into()],
            description: "Kinetic energy distribution by group".into(),
            required_fields: vec!["selection".into()],
            optional_fields: vec!["group_by".into(), "velocity_scale".into(), "length_scale".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Atom selection for kinetic energy".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "group_by".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "string".into(),
                        description: Some("How to group atoms".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "velocity_scale".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Velocity scaling factor".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "table".into(),
                    format: "npz".into(),
                    fields: vec!["group".into(), "kinetic_energy_kJ_per_mol".into()],
                    description: Some("Kinetic energy per group".into()),
                }
            ],
            tags: vec!["thermodynamic".into()],
            examples: vec![
                serde_json::json!({"name": "equipartition", "selection": "protein"})
            ],
        },
        WarpMdAnalysisContract {
            name: "free_volume".into(),
            aliases: vec!["free-volume".into(), "free-volume-grid".into(), "voxel-free-volume".into()],
            description: "Voxel-grid free-volume fraction. Useful for spatial void maps, not Bondi-style polymer FFV.".into(),
            required_fields: vec!["selection".into(), "center_selection".into()],
            optional_fields: vec!["box_unit".into(), "region_size".into(), "probe_radius".into(), "shift".into(), "length_scale".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Atoms treated as occupied volume".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "center_selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Selection used to define grid origin and auto-detect region_size".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "box_unit".into(),
                    WarpMdFieldSpec {
                        field_type: "array".into(),
                        semantic_type: "vector".into(),
                        description: Some("Voxel size (x, y, z) in Angstroms. Defaults to [1.0, 1.0, 1.0] if not specified.".into()),
                        default: Some(serde_json::json!([1.0, 1.0, 1.0])),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "region_size".into(),
                    WarpMdFieldSpec {
                        field_type: "array".into(),
                        semantic_type: "vector".into(),
                        description: Some("Region extents (x, y, z) in Angstroms. Auto-detected from center_selection bounding box with 10%% padding if not specified.".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "probe_radius".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Probe radius that expands occupied volume".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "shift".into(),
                    WarpMdFieldSpec {
                        field_type: "array".into(),
                        semantic_type: "vector".into(),
                        description: Some("Shift for centered coordinates".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "length_scale".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Coordinate length scale".into()),
                        default: Some(serde_json::json!(1.0)),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "grid".into(),
                    format: "npz".into(),
                    fields: vec!["dims".into(), "mean".into(), "std".into(), "first".into(), "last".into(), "min".into(), "max".into()],
                    description: Some("Per-voxel free-volume fraction statistics".into()),
                }
            ],
            tags: vec!["spatial".into(), "grid".into(), "void".into(), "solvation".into()],
            examples: vec![
                serde_json::json!({"name": "free_volume", "selection": "protein", "center_selection": "protein", "box_unit": [1.0, 1.0, 1.0], "region_size": [30.0, 30.0, 30.0], "probe_radius": 0.5, "note": "Explicit parameters"}),
                serde_json::json!({"name": "free_volume_auto", "selection": "protein", "center_selection": "protein", "note": "Auto-detects region_size from bounding box, defaults box_unit to 1.0 \u{C5}"})
            ],
        },
        WarpMdAnalysisContract {
            name: "hydrophobic_defects".into(),
            aliases: vec!["hydrophobic-defects".into(), "membrane-packing-defects".into()],
            description: "Membrane hydrophobic packing-defect voxel analysis with XY periodic boundaries.".into(),
            required_fields: vec!["lipid_selection".into(), "reference_selection".into()],
            optional_fields: vec!["voxel_size".into(), "z_bounds".into(), "probe_radius".into(), "defect_radius".into(), "grid_mode".into(), "leaflet".into(), "midplane_selection".into(), "leaflet_bins".into(), "length_scale".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "lipid_selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Membrane or lipid atoms treated as occupied volume".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "reference_selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Reference atoms defining candidate defect columns".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "midplane_selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Optional selection used to define the leaflet midplane; defaults to lipid_selection when leaflet filtering is enabled.".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "voxel_size".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Uniform voxel edge length".into()),
                        default: Some(serde_json::json!(1.0)),
                        minimum: Some(0.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "z_bounds".into(),
                    WarpMdFieldSpec {
                        field_type: "array".into(),
                        semantic_type: "vector".into(),
                        description: Some("Lower and upper z bounds for the membrane defect slab".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "probe_radius".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Probe radius added to van der Waals radii when marking occupied voxels".into()),
                        default: Some(serde_json::json!(1.4)),
                        minimum: Some(0.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "defect_radius".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("XY search radius around each reference atom".into()),
                        default: Some(serde_json::json!(7.3)),
                        minimum: Some(0.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "grid_mode".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "string".into(),
                        description: Some("Grid sampling mode: voxel_centers for centered voxels, lattice_nodes for compatibility with 3DLipidPackingDefects.".into()),
                        default: Some(serde_json::json!("voxel_centers")),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: Some(vec!["voxel_centers".into(), "lattice_nodes".into()]),
                    },
                ),
                (
                    "leaflet".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "string".into(),
                        description: Some("Optional automatic leaflet filter applied per frame before defect detection.".into()),
                        default: Some(serde_json::json!("both")),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: Some(vec!["both".into(), "upper".into(), "lower".into()]),
                    },
                ),
                (
                    "leaflet_bins".into(),
                    WarpMdFieldSpec {
                        field_type: "integer".into(),
                        semantic_type: "integer".into(),
                        description: Some("XY bins per dimension for local leaflet midplane inference".into()),
                        default: Some(serde_json::json!(1)),
                        minimum: Some(1.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "length_scale".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Coordinate length scale applied before analysis".into()),
                        default: Some(serde_json::json!(1.0)),
                        minimum: Some(0.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "grid".into(),
                    format: "npz".into(),
                    fields: vec!["dims".into(), "voxel_size".into(), "z_bounds".into(), "grid_mode".into(), "leaflet".into(), "mean".into(), "first".into(), "last".into(), "min".into(), "max".into(), "frame_counts".into(), "frame_area".into(), "frame_volume".into(), "frame_cluster_count".into(), "frame_largest_cluster".into(), "max_lifetime".into()],
                    description: Some("Hydrophobic defect voxel masks and per-frame count/area/volume summaries".into()),
                }
            ],
            tags: vec!["membrane".into(), "lipid".into(), "grid".into(), "void".into()],
            examples: vec![
                serde_json::json!({"name": "hydrophobic_defects", "lipid_selection": "resname POPC", "reference_selection": "name C2 C3 C21 C31", "voxel_size": 1.0, "z_bounds": [-20.0, 20.0], "probe_radius": 1.4, "defect_radius": 7.3, "grid_mode": "voxel_centers", "leaflet": "upper"})
            ],
        },
        WarpMdAnalysisContract {
            name: "gist".into(),
            aliases: vec!["grid-inhomogeneous-solvation-theory".into()],
            description: "Grid inhomogeneous solvation theory (water thermodynamics)".into(),
            required_fields: vec!["solute".into(), "solvent".into()],
            optional_fields: vec![],
            fields: std::collections::BTreeMap::from([
                (
                    "solute".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Solute selection".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "solvent".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Solvent selection".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "grid".into(),
                    format: "npz".into(),
                    fields: vec!["energy".into(), "entropy".into()],
                    description: Some("3D thermodynamic maps".into()),
                }
            ],
            tags: vec!["thermodynamic".into(), "solvent".into(), "grid".into()],
            examples: vec![
                serde_json::json!({"name": "gist", "solute": "protein", "solvent": "resname SOL"})
            ],
        },
        WarpMdAnalysisContract {
            name: "hbond".into(),
            aliases: vec!["hydrogen-bond".into()],
            description: "Hydrogen bond analysis".into(),
            required_fields: vec!["donors".into(), "acceptors".into(), "dist_cutoff".into()],
            optional_fields: vec!["hydrogens".into(), "angle_cutoff".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "donors".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Hydrogen bond donor selection".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "acceptors".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Hydrogen bond acceptor selection".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "dist_cutoff".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Distance cutoff for H-bond".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: Some("nm".into()),
                        choices: None,
                    },
                ),
                (
                    "hydrogens".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Hydrogen atom selection (for angle criteria)".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "angle_cutoff".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Angle cutoff in degrees".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: Some(180.0),
                        unit: Some("degrees".into()),
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "timeseries".into(),
                    format: "npz".into(),
                    fields: vec!["time_ps".into(), "hbond_count".into()],
                    description: Some("Hydrogen bond count vs time".into()),
                }
            ],
            tags: vec!["structural".into(), "solvent".into()],
            examples: vec![
                serde_json::json!({"name": "hbond", "donors": "resname SOL and name OH", "acceptors": "resname SOL and name O", "dist_cutoff": 0.35})
            ],
        },
        WarpMdAnalysisContract {
            name: "ion_pair_correlation".into(),
            aliases: vec!["ion-pair-correlation".into(), "ion-pair".into()],
            description: "Ion pair lifetime and correlation analysis".into(),
            required_fields: vec!["selection".into(), "rclust_cat".into(), "rclust_ani".into()],
            optional_fields: vec!["group_by".into(), "cation_type".into(), "anion_type".into(), "max_cluster".into(), "length_scale".into(), "lag_mode".into(), "max_lag".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Ion selection".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "rclust_cat".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Cation clustering cutoff distance".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: Some("nm".into()),
                        choices: None,
                    },
                ),
                (
                    "rclust_ani".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Anion clustering cutoff distance".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: Some("nm".into()),
                        choices: None,
                    },
                ),
                (
                    "cation_type".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "string".into(),
                        description: Some("Cation residue name".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "anion_type".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "string".into(),
                        description: Some("Anion residue name".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "max_cluster".into(),
                    WarpMdFieldSpec {
                        field_type: "integer".into(),
                        semantic_type: "integer".into(),
                        description: Some("Maximum cluster size".into()),
                        default: None,
                        minimum: Some(1.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "timeseries".into(),
                    format: "npz".into(),
                    fields: vec!["lag_time_ps".into(), "correlation".into()],
                    description: Some("Ion pair correlation function".into()),
                }
            ],
            tags: vec!["electrostatics".into(), "dynamic".into()],
            examples: vec![
                serde_json::json!({"name": "ion_pair_correlation", "selection": "resname Na or resname CL", "rclust_cat": 0.35, "rclust_ani": 0.35})
            ],
        },
        WarpMdAnalysisContract {
            name: "jcoupling".into(),
            aliases: vec!["j-coupling".into(), "scalar-coupling".into()],
            description: "J-coupling constants from dihedral angles".into(),
            required_fields: vec!["dihedrals".into()],
            optional_fields: vec!["karplus".into(), "kfile".into(), "phase_deg".into(), "length_scale".into(), "pbc".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "dihedrals".into(),
                    WarpMdFieldSpec {
                        field_type: "array".into(),
                        semantic_type: "vector".into(),
                        description: Some("Dihedral atom indices (quadruples)".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "karplus".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "string".into(),
                        description: Some("Karplus parameterization".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "kfile".into(),
                    WarpMdFieldSpec {
                        field_type: "path".into(),
                        semantic_type: "path".into(),
                        description: Some("Path to Karplus parameter file".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "table".into(),
                    format: "npz".into(),
                    fields: vec!["dihedral".into(), "j_coupling_hz".into()],
                    description: Some("J-coupling per dihedral".into()),
                }
            ],
            tags: vec!["nmr".into()],
            examples: vec![
                serde_json::json!({"name": "jcoupling", "dihedrals": [[1, 2, 3, 4], [5, 6, 7, 8]]})
            ],
        },
        lipid_matrix_contract(
            "lipid_leaflets",
            &["lipid-leaflets", "leaflets"],
            "Assign lipid residues to upper/lower leaflets from headgroup positions",
            &["selection"],
            &["midplane_selection", "midplane_cutoff", "bins", "length_scale"],
            &[
                ("selection", "string", "selection", "Lipid headgroup selection", None, None, None),
                ("midplane_selection", "string", "selection", "Selection used to define the membrane midplane", None, None, None),
                ("midplane_cutoff", "float", "float", "Midplane exclusion cutoff", Some(serde_json::json!(0.0)), Some(0.0), None),
                ("bins", "integer", "integer", "XY bins per dimension for local midplanes", Some(serde_json::json!(1)), Some(1.0), None),
                ("length_scale", "float", "float", "Coordinate length scale", Some(serde_json::json!(1.0)), None, None),
            ],
            "table",
            &["values", "residue_ids", "frames"],
            serde_json::json!({"name": "lipid_leaflets", "selection": "resname POPC and name PO4"}),
        ),
        lipid_matrix_contract(
            "lipid_curved_leaflets",
            &["lipid-curved-leaflets", "curved-leaflets"],
            "Assign curved membrane leaflets by graph connectivity",
            &["selection"],
            &["cutoff", "midplane_selection", "midplane_cutoff", "length_scale"],
            &[
                ("selection", "string", "selection", "Lipid headgroup selection", None, None, None),
                ("cutoff", "float", "float", "Graph connectivity cutoff", Some(serde_json::json!(15.0)), Some(0.0), None),
                ("midplane_selection", "string", "selection", "Selection used to define the membrane midplane", None, None, None),
                ("midplane_cutoff", "float", "float", "Midplane exclusion cutoff", Some(serde_json::json!(0.0)), Some(0.0), None),
                ("length_scale", "float", "float", "Coordinate length scale", Some(serde_json::json!(1.0)), None, None),
            ],
            "table",
            &["values", "residue_ids", "frames"],
            serde_json::json!({"name": "lipid_curved_leaflets", "selection": "resname POPC and name PO4", "cutoff": 15.0}),
        ),
        lipid_matrix_contract(
            "lipid_z_positions",
            &["lipid-z-positions"],
            "Per-residue lipid height relative to a membrane reference",
            &["membrane_selection", "height_selection"],
            &["bins", "length_scale"],
            &[
                ("membrane_selection", "string", "selection", "Membrane reference selection", None, None, None),
                ("height_selection", "string", "selection", "Selection whose z-position is measured", None, None, None),
                ("bins", "integer", "integer", "XY bins per dimension", Some(serde_json::json!(1)), Some(1.0), None),
                ("length_scale", "float", "float", "Coordinate length scale", Some(serde_json::json!(1.0)), None, None),
            ],
            "table",
            &["values", "residue_ids", "frames"],
            serde_json::json!({"name": "lipid_z_positions", "membrane_selection": "resname POPC and name PO4", "height_selection": "resname POPC and name PO4"}),
        ),
        lipid_matrix_contract(
            "lipid_z_thickness",
            &["lipid-z-thickness"],
            "Per-residue lipid z-thickness",
            &["selection"],
            &["length_scale"],
            &[
                ("selection", "string", "selection", "Lipid selection", None, None, None),
                ("length_scale", "float", "float", "Coordinate length scale", Some(serde_json::json!(1.0)), None, None),
            ],
            "table",
            &["values", "residue_ids", "frames"],
            serde_json::json!({"name": "lipid_z_thickness", "selection": "resname POPC"}),
        ),
        lipid_matrix_contract(
            "lipid_z_angles",
            &["lipid-z-angles"],
            "Angle between a lipid atom vector and the membrane z-axis",
            &["atom_a", "atom_b"],
            &["degrees", "length_scale"],
            &[
                ("atom_a", "string", "selection", "First atom selection", None, None, None),
                ("atom_b", "string", "selection", "Second atom selection", None, None, None),
                ("degrees", "boolean", "boolean", "Return angles in degrees", Some(serde_json::json!(true)), None, None),
                ("length_scale", "float", "float", "Coordinate length scale", Some(serde_json::json!(1.0)), None, None),
            ],
            "table",
            &["values", "residue_ids", "frames"],
            serde_json::json!({"name": "lipid_z_angles", "atom_a": "name C1", "atom_b": "name C2"}),
        ),
        lipid_matrix_contract(
            "lipid_area",
            &["lipid-area", "area-per-lipid"],
            "Area per lipid from 2D Voronoi tessellation",
            &["selection", "leaflets"],
            &["length_scale"],
            &[
                ("selection", "string", "selection", "Lipid headgroup selection", None, None, None),
                ("leaflets", "path", "path", "Leaflet assignment array path or inline array", None, None, None),
                ("length_scale", "float", "float", "Coordinate length scale", Some(serde_json::json!(1.0)), None, None),
            ],
            "table",
            &["values", "residue_ids", "frames"],
            serde_json::json!({"name": "lipid_area", "selection": "resname POPC and name PO4", "leaflets": "leaflets.npz"}),
        ),
        lipid_matrix_contract(
            "lipid_flip_flop",
            &["lipid-flip-flop", "flip-flop"],
            "Detect lipid leaflet transitions from leaflet assignments",
            &["leaflets"],
            &["residue_ids", "frame_cutoff"],
            &[
                ("leaflets", "path", "path", "Leaflet assignment array path or inline array", None, None, None),
                ("residue_ids", "path", "path", "Residue id array path or inline array", None, None, None),
                ("frame_cutoff", "integer", "integer", "Minimum transition dwell frames", Some(serde_json::json!(1)), Some(1.0), None),
            ],
            "table",
            &["events", "success", "residue_ids"],
            serde_json::json!({"name": "lipid_flip_flop", "leaflets": "leaflets.npz", "frame_cutoff": 2}),
        ),
        lipid_matrix_contract(
            "lipid_neighbours",
            &["lipid-neighbours"],
            "Per-lipid neighbour counts within a cutoff",
            &["selection"],
            &["cutoff", "length_scale"],
            &[
                ("selection", "string", "selection", "Lipid selection", None, None, None),
                ("cutoff", "float", "float", "Neighbour cutoff", Some(serde_json::json!(10.0)), Some(0.0), None),
                ("length_scale", "float", "float", "Coordinate length scale", Some(serde_json::json!(1.0)), None, None),
            ],
            "table",
            &["values", "residue_ids", "frames"],
            serde_json::json!({"name": "lipid_neighbours", "selection": "resname POPC and name PO4", "cutoff": 10.0}),
        ),
        lipid_matrix_contract(
            "lipid_neighbour_matrix",
            &["lipid-neighbour-matrix"],
            "Per-frame lipid adjacency matrices within a cutoff",
            &["selection"],
            &["cutoff", "length_scale"],
            &[
                ("selection", "string", "selection", "Lipid selection", None, None, None),
                ("cutoff", "float", "float", "Neighbour cutoff", Some(serde_json::json!(10.0)), Some(0.0), None),
                ("length_scale", "float", "float", "Coordinate length scale", Some(serde_json::json!(1.0)), None, None),
            ],
            "table",
            &["values", "residue_ids", "frames"],
            serde_json::json!({"name": "lipid_neighbour_matrix", "selection": "resname POPC and name PO4", "cutoff": 10.0}),
        ),
        lipid_matrix_contract(
            "lipid_largest_cluster",
            &["lipid-largest-cluster"],
            "Largest lipid neighbour-connected cluster size per frame",
            &["selection"],
            &["cutoff", "length_scale"],
            &[
                ("selection", "string", "selection", "Lipid selection", None, None, None),
                ("cutoff", "float", "float", "Cluster neighbour cutoff", Some(serde_json::json!(10.0)), Some(0.0), None),
                ("length_scale", "float", "float", "Coordinate length scale", Some(serde_json::json!(1.0)), None, None),
            ],
            "timeseries",
            &["values", "residue_ids", "frames"],
            serde_json::json!({"name": "lipid_largest_cluster", "selection": "resname POPC and name PO4", "cutoff": 10.0}),
        ),
        lipid_matrix_contract(
            "lipid_membrane_thickness",
            &["lipid-membrane-thickness", "membrane-thickness"],
            "Local membrane thickness from opposite leaflet positions",
            &["selection", "leaflets"],
            &["bins", "length_scale"],
            &[
                ("selection", "string", "selection", "Lipid headgroup selection", None, None, None),
                ("leaflets", "path", "path", "Leaflet assignment array path or inline array", None, None, None),
                ("bins", "integer", "integer", "XY bins per dimension", Some(serde_json::json!(1)), Some(1.0), None),
                ("length_scale", "float", "float", "Coordinate length scale", Some(serde_json::json!(1.0)), None, None),
            ],
            "table",
            &["values", "residue_ids", "frames"],
            serde_json::json!({"name": "lipid_membrane_thickness", "selection": "resname POPC and name PO4", "leaflets": "leaflets.npz"}),
        ),
        lipid_matrix_contract(
            "lipid_registration",
            &["lipid-registration"],
            "Leaflet registration/correlation on an XY grid",
            &["upper_selection", "lower_selection", "leaflets"],
            &["bins", "gaussian_sd", "length_scale"],
            &[
                ("upper_selection", "string", "selection", "Upper leaflet selection", None, None, None),
                ("lower_selection", "string", "selection", "Lower leaflet selection", None, None, None),
                ("leaflets", "path", "path", "Leaflet assignment array path or inline array", None, None, None),
                ("bins", "integer", "integer", "XY bins per dimension", Some(serde_json::json!(1)), Some(1.0), None),
                ("gaussian_sd", "float", "float", "Optional Gaussian smoothing width", Some(serde_json::json!(0.0)), Some(0.0), None),
                ("length_scale", "float", "float", "Coordinate length scale", Some(serde_json::json!(1.0)), None, None),
            ],
            "table",
            &["values", "residue_ids", "frames"],
            serde_json::json!({"name": "lipid_registration", "upper_selection": "resname POPC", "lower_selection": "resname POPC", "leaflets": "leaflets.npz"}),
        ),
        lipid_matrix_contract(
            "lipid_msd",
            &["lipid-msd"],
            "Lipid lateral mean-square displacement",
            &["selection"],
            &["com_removal_selection", "length_scale"],
            &[
                ("selection", "string", "selection", "Lipid selection", None, None, None),
                ("com_removal_selection", "string", "selection", "Selection whose COM is removed before MSD", None, None, None),
                ("length_scale", "float", "float", "Coordinate length scale", Some(serde_json::json!(1.0)), None, None),
            ],
            "timeseries",
            &["values", "residue_ids", "frames"],
            serde_json::json!({"name": "lipid_msd", "selection": "resname POPC and name PO4"}),
        ),
        lipid_matrix_contract(
            "lipid_scc",
            &["lipid-scc", "scc"],
            "Lipid tail deuterium order parameter SCC",
            &["tail_selection"],
            &["normals", "length_scale"],
            &[
                ("tail_selection", "string", "selection", "Tail atom selection ordered by residue", None, None, None),
                ("normals", "path", "path", "Optional normal vectors array path or inline array", None, None, None),
                ("length_scale", "float", "float", "Coordinate length scale", Some(serde_json::json!(1.0)), None, None),
            ],
            "table",
            &["values", "residue_ids", "frames"],
            serde_json::json!({"name": "lipid_scc", "tail_selection": "resname POPC and name C*"}),
        ),
        WarpMdAnalysisContract {
            name: "molsurf".into(),
            aliases: vec!["molecular-surface".into()],
            description: "Molecular surface area (Connolly)".into(),
            required_fields: vec![],
            optional_fields: vec!["mask".into(), "algorithm".into(), "probe_radius".into(), "radii".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "mask".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "mask".into(),
                        description: Some("Atom mask".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "probe_radius".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Probe radius".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: Some("nm".into()),
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "timeseries".into(),
                    format: "npz".into(),
                    fields: vec!["time_ps".into(), "surface_area_nm2".into()],
                    description: Some("Molecular surface area vs time".into()),
                }
            ],
            tags: vec!["structural".into(), "surface".into()],
            examples: vec![
                serde_json::json!({"name": "molsurf", "mask": "protein"})
            ],
        },
        WarpMdAnalysisContract {
            name: "msd".into(),
            aliases: vec!["mean-square-displacement".into()],
            description: "Mean square displacement - diffusion analysis".into(),
            required_fields: vec!["selection".into()],
            optional_fields: vec!["group_by".into(), "axis".into(), "length_scale".into(), "frame_decimation".into(), "dt_decimation".into(), "time_binning".into(), "lag_mode".into(), "max_lag".into(), "memory_budget_bytes".into(), "multi_tau_m".into(), "multi_tau_levels".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Atom selection for MSD calculation".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "group_by".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "string".into(),
                        description: Some("How to group atoms (resid, molecule, etc.)".into()),
                        default: Some(serde_json::json!("resid")),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "axis".into(),
                    WarpMdFieldSpec {
                        field_type: "array".into(),
                        semantic_type: "vector".into(),
                        description: Some("3D vector for directional MSD (x, y, z)".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "length_scale".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Length unit conversion factor (nm per unit)".into()),
                        default: Some(serde_json::json!(1.0)),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "max_lag".into(),
                    WarpMdFieldSpec {
                        field_type: "integer".into(),
                        semantic_type: "integer".into(),
                        description: Some("Maximum lag time in frames".into()),
                        default: None,
                        minimum: Some(1.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "lag_mode".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "string".into(),
                        description: Some("Lag computation mode".into()),
                        default: Some(serde_json::json!("linear")),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: Some(vec!["linear".into(), "log".into(), "ring".into()]),
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "timeseries".into(),
                    format: "npz".into(),
                    fields: vec!["lag_time_ps".into(), "msd_nm2".into()],
                    description: Some("MSD vs lag time".into()),
                }
            ],
            tags: vec!["dynamic".into(), "diffusion".into(), "transport".into()],
            examples: vec![
                serde_json::json!({"name": "msd", "selection": "resname CL and name NA"})
            ],
        },
        WarpMdAnalysisContract {
            name: "native_contacts".into(),
            aliases: vec!["native-contacts".into(), "q-value".into()],
            description: "Native contact analysis (folding)".into(),
            required_fields: vec![],
            optional_fields: vec!["mask".into(), "mask2".into(), "ref".into(), "distance".into(), "mindist".into(), "maxdist".into(), "image".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "mask".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "mask".into(),
                        description: Some("First atom mask".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "mask2".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "mask".into(),
                        description: Some("Second atom mask".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "distance".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Contact distance cutoff".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: Some("nm".into()),
                        choices: None,
                    },
                ),
                (
                    "mindist".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Minimum contact distance".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: Some("nm".into()),
                        choices: None,
                    },
                ),
                (
                    "maxdist".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Maximum contact distance".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: Some("nm".into()),
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "timeseries".into(),
                    format: "npz".into(),
                    fields: vec!["time_ps".into(), "native_contacts".into(), "q_value".into()],
                    description: Some("Native contacts vs time".into()),
                }
            ],
            tags: vec!["structural".into(), "protein".into(), "folding".into()],
            examples: vec![
                serde_json::json!({"name": "native_contacts", "mask": "protein"})
            ],
        },
        WarpMdAnalysisContract {
            name: "nmr".into(),
            aliases: vec!["nmr-order-parameters".into()],
            description: "NMR NH order parameters".into(),
            required_fields: vec!["selection".into()],
            optional_fields: vec!["vector_pairs".into(), "method".into(), "order".into(), "tstep".into(), "tcorr".into(), "length_scale".into(), "pbc".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("NH vector selection".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "vector_pairs".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "string".into(),
                        description: Some("Vector pair specification".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "method".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "string".into(),
                        description: Some("Computation method".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "order".into(),
                    WarpMdFieldSpec {
                        field_type: "integer".into(),
                        semantic_type: "integer".into(),
                        description: Some("Order parameter (2 for S^2)".into()),
                        default: Some(serde_json::json!(2)),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "table".into(),
                    format: "npz".into(),
                    fields: vec!["residue".into(), "order_parameter".into()],
                    description: Some("Order parameters per residue".into()),
                }
            ],
            tags: vec!["nmr".into(), "protein".into()],
            examples: vec![
                serde_json::json!({"name": "nmr", "selection": "protein"})
            ],
        },
        WarpMdAnalysisContract {
            name: "pca".into(),
            aliases: vec!["principal-component-analysis".into(), "pca-analysis".into()],
            description: "Principal component analysis of atomic fluctuations".into(),
            required_fields: vec!["mask".into()],
            optional_fields: vec!["n_vecs".into(), "fit".into(), "ref".into(), "ref_mask".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "mask".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "mask".into(),
                        description: Some("Atom mask for PCA".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "n_vecs".into(),
                    WarpMdFieldSpec {
                        field_type: "integer".into(),
                        semantic_type: "integer".into(),
                        description: Some("Number of eigenvectors to compute".into()),
                        default: None,
                        minimum: Some(1.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "fit".into(),
                    WarpMdFieldSpec {
                        field_type: "boolean".into(),
                        semantic_type: "boolean".into(),
                        description: Some("Fit to average structure".into()),
                        default: Some(serde_json::json!(true)),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "artifact".into(),
                    format: "npz".into(),
                    fields: vec!["eigenvalues".into(), "eigenvectors".into(), "projections".into()],
                    description: Some("PCA eigendecomposition results".into()),
                }
            ],
            tags: vec!["structural".into(), "protein".into()],
            examples: vec![
                serde_json::json!({"name": "pca", "mask": "protein and name CA", "n_vecs": 10})
            ],
        },
        WarpMdAnalysisContract {
            name: "persistence_length".into(),
            aliases: vec!["persistence-length".into()],
            description: "Polymer persistence length from bond vectors".into(),
            required_fields: vec!["selection".into()],
            optional_fields: vec![],
            fields: std::collections::BTreeMap::from([
                (
                    "selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Polymer backbone selection".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "artifact".into(),
                    format: "npz".into(),
                    fields: vec!["persistence_length_nm".into()],
                    description: Some("Persistence length".into()),
                }
            ],
            tags: vec!["structural".into(), "polymer".into()],
            examples: vec![
                serde_json::json!({"name": "persistence_length", "selection": "polymer"})
            ],
        },
        WarpMdAnalysisContract {
            name: "projection".into(),
            aliases: vec!["pca-projection".into()],
            description: "Project trajectory onto PCA eigenvectors".into(),
            required_fields: vec!["mask".into()],
            optional_fields: vec!["eigenvec".into(), "n_vecs".into(), "fit".into(), "ref".into(), "ref_mask".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "mask".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "mask".into(),
                        description: Some("Atom mask".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "eigenvec".into(),
                    WarpMdFieldSpec {
                        field_type: "path".into(),
                        semantic_type: "path".into(),
                        description: Some("Path to eigenvector file".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "n_vecs".into(),
                    WarpMdFieldSpec {
                        field_type: "integer".into(),
                        semantic_type: "integer".into(),
                        description: Some("Number of eigenvectors".into()),
                        default: None,
                        minimum: Some(1.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "timeseries".into(),
                    format: "npz".into(),
                    fields: vec!["time_ps".into(), "pc1".into(), "pc2".into(), "...".into()],
                    description: Some("PCA projections vs time".into()),
                }
            ],
            tags: vec!["structural".into(), "protein".into()],
            examples: vec![
                serde_json::json!({"name": "projection", "mask": "protein and name CA", "n_vecs": 3})
            ],
        },
        WarpMdAnalysisContract {
            name: "pairdist".into(),
            aliases: vec!["pair-distance".into(), "distance-distribution".into()],
            description: "Pair-distance histogram or per-frame distance extrema".into(),
            required_fields: vec!["mask".into()],
            optional_fields: vec!["mask2".into(), "delta".into(), "maxdist".into(), "mode".into(), "image".into(), "frame_indices".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "mask".into(),
                    field("string", "mask", "First atom selection", None, None, None),
                ),
                (
                    "mask2".into(),
                    field("string", "mask", "Second atom selection", None, None, None),
                ),
                (
                    "delta".into(),
                    field(
                        "float",
                        "float",
                        "Distance histogram bin width",
                        Some(serde_json::json!(0.1)),
                        Some(0.0),
                        None,
                    ),
                ),
                (
                    "maxdist".into(),
                    field(
                        "float",
                        "float",
                        "Maximum distance cutoff",
                        None,
                        Some(0.0),
                        None,
                    ),
                ),
                (
                    "mode".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "string".into(),
                        description: Some("Histogram or per-frame extrema mode".into()),
                        default: Some(serde_json::json!("hist")),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: Some(vec!["hist".into(), "min".into(), "max".into()]),
                    },
                ),
                (
                    "image".into(),
                    field(
                        "boolean",
                        "boolean",
                        "Apply orthorhombic periodic imaging",
                        Some(serde_json::json!(false)),
                        None,
                        None,
                    ),
                ),
                (
                    "frame_indices".into(),
                    field(
                        "array",
                        "integer",
                        "Optional frame indices to evaluate",
                        None,
                        None,
                        None,
                    ),
                ),
            ]),
            outputs: vec![WarpMdArtifactSpec {
                kind: "histogram".into(),
                format: "npz".into(),
                fields: vec!["bin_centers".into(), "hist".into(), "std".into(), "counts".into()],
                description: Some("Pair-distance distribution or extrema output".into()),
            }],
            tags: vec!["structural".into(), "distribution".into(), "distance".into()],
            examples: vec![
                serde_json::json!({"name": "pairdist", "mask": "@CA", "mask2": "@CB", "delta": 0.1, "maxdist": 10.0, "mode": "hist"})
            ],
        },
        WarpMdAnalysisContract {
            name: "rdf".into(),
            aliases: vec!["radial-distribution-function".into(), "pair-distribution".into()],
            description: "Radial distribution function g(r)".into(),
            required_fields: vec!["sel_a".into(), "sel_b".into(), "bins".into(), "r_max".into()],
            optional_fields: vec!["pbc".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "sel_a".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Selection for group A".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "sel_b".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Selection for group B".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "bins".into(),
                    WarpMdFieldSpec {
                        field_type: "integer".into(),
                        semantic_type: "integer".into(),
                        description: Some("Number of r bins".into()),
                        default: Some(serde_json::json!(200)),
                        minimum: Some(1.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "r_max".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Maximum r distance".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: Some("nm".into()),
                        choices: None,
                    },
                ),
                (
                    "pbc".into(),
                    WarpMdFieldSpec {
                        field_type: "boolean".into(),
                        semantic_type: "boolean".into(),
                        description: Some("Apply periodic boundary conditions".into()),
                        default: Some(serde_json::json!(true)),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "histogram".into(),
                    format: "npz".into(),
                    fields: vec!["r_nm".into(), "gr".into()],
                    description: Some("RDF g(r) vs distance".into()),
                }
            ],
            tags: vec!["structural".into(), "solvent".into(), "distribution".into()],
            examples: vec![
                serde_json::json!({"name": "rdf", "sel_a": "resname SOL and name OW", "sel_b": "resname SOL and name OW", "bins": 200, "r_max": 1.0})
            ],
        },
        WarpMdAnalysisContract {
            name: "rg".into(),
            aliases: vec!["radius-of-gyration".into()],
            description: "Radius of gyration - measure of polymer compactness".into(),
            required_fields: vec!["selection".into()],
            optional_fields: vec!["mass_weighted".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Atom selection for Rg calculation".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "mass_weighted".into(),
                    WarpMdFieldSpec {
                        field_type: "boolean".into(),
                        semantic_type: "boolean".into(),
                        description: Some("Use mass-weighted coordinates".into()),
                        default: Some(serde_json::json!(false)),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "timeseries".into(),
                    format: "npz".into(),
                    fields: vec!["time_ps".into(), "rg_nm".into()],
                    description: Some("Time series of radius of gyration values".into()),
                }
            ],
            tags: vec!["structural".into(), "polymer".into(), "protein".into()],
            examples: vec![
                serde_json::json!({"name": "rg", "selection": "protein", "mass_weighted": true})
            ],
        },
        WarpMdAnalysisContract {
            name: "rmsd".into(),
            aliases: vec!["root-mean-square-deviation".into()],
            description: "Root mean square deviation from reference structure".into(),
            required_fields: vec!["selection".into()],
            optional_fields: vec!["reference".into(), "align".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Atom selection for RMSD calculation".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "reference".into(),
                    WarpMdFieldSpec {
                        field_type: "integer".into(),
                        semantic_type: "integer".into(),
                        description: Some("Reference frame index (0-based)".into()),
                        default: Some(serde_json::json!(0)),
                        minimum: Some(0.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "align".into(),
                    WarpMdFieldSpec {
                        field_type: "boolean".into(),
                        semantic_type: "boolean".into(),
                        description: Some("Align to reference before RMSD calculation".into()),
                        default: Some(serde_json::json!(true)),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "timeseries".into(),
                    format: "npz".into(),
                    fields: vec!["time_ps".into(), "rmsd_nm".into()],
                    description: Some("Time series of RMSD values".into()),
                }
            ],
            tags: vec!["structural".into(), "protein".into()],
            examples: vec![
                serde_json::json!({"name": "rmsd", "selection": "protein and name CA", "reference": 0, "align": true})
            ],
        },
        WarpMdAnalysisContract {
            name: "rmsf".into(),
            aliases: vec!["root-mean-square-fluctuation".into()],
            description: "Root mean square fluctuation (per-atom mobility)".into(),
            required_fields: vec![],
            optional_fields: vec!["mask".into(), "byres".into(), "bymask".into(), "calcadp".into(), "length_scale".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "mask".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "mask".into(),
                        description: Some("Atom mask".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "byres".into(),
                    WarpMdFieldSpec {
                        field_type: "boolean".into(),
                        semantic_type: "boolean".into(),
                        description: Some("Aggregate by residue".into()),
                        default: Some(serde_json::json!(false)),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "bymask".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "mask".into(),
                        description: Some("Mask for aggregation".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "calcadp".into(),
                    WarpMdFieldSpec {
                        field_type: "boolean".into(),
                        semantic_type: "boolean".into(),
                        description: Some("Calculate ADP (B-factors)".into()),
                        default: Some(serde_json::json!(false)),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "table".into(),
                    format: "npz".into(),
                    fields: vec!["atom".into(), "rmsf_nm".into()],
                    description: Some("RMSF per atom/residue".into()),
                }
            ],
            tags: vec!["structural".into(), "protein".into()],
            examples: vec![
                serde_json::json!({"name": "rmsf", "mask": "protein and name CA", "byres": true})
            ],
        },
        WarpMdAnalysisContract {
            name: "rotacf".into(),
            aliases: vec!["rotational-autocorrelation".into(), "rotational-acf".into()],
            description: "Rotational autocorrelation function".into(),
            required_fields: vec!["selection".into(), "orientation".into()],
            optional_fields: vec!["group_by".into(), "p2_legendre".into(), "length_scale".into(), "frame_decimation".into(), "dt_decimation".into(), "time_binning".into(), "lag_mode".into(), "max_lag".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Atom selection for rotational ACF".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "orientation".into(),
                    WarpMdFieldSpec {
                        field_type: "array".into(),
                        semantic_type: "vector".into(),
                        description: Some("Vector indices for orientation (2 or 3 atoms)".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "p2_legendre".into(),
                    WarpMdFieldSpec {
                        field_type: "boolean".into(),
                        semantic_type: "boolean".into(),
                        description: Some("Use second Legendre polynomial".into()),
                        default: Some(serde_json::json!(true)),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "max_lag".into(),
                    WarpMdFieldSpec {
                        field_type: "integer".into(),
                        semantic_type: "integer".into(),
                        description: Some("Maximum lag time in frames".into()),
                        default: None,
                        minimum: Some(1.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "timeseries".into(),
                    format: "npz".into(),
                    fields: vec!["lag_time_ps".into(), "acf".into()],
                    description: Some("Rotational ACF vs lag time".into()),
                }
            ],
            tags: vec!["dynamic".into(), "rotational".into()],
            examples: vec![
                serde_json::json!({"name": "rotacf", "selection": "resname MEOH and name OH", "orientation": [0, 1, 2]})
            ],
        },
        WarpMdAnalysisContract {
            name: "structure_factor".into(),
            aliases: vec!["structure-factor".into(), "sk".into()],
            description: "Static structure factor S(q)".into(),
            required_fields: vec!["selection".into(), "bins".into(), "r_max".into(), "q_bins".into(), "q_max".into()],
            optional_fields: vec!["pbc".into(), "length_scale".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Atom selection for structure factor".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "bins".into(),
                    WarpMdFieldSpec {
                        field_type: "integer".into(),
                        semantic_type: "integer".into(),
                        description: Some("Number of real-space bins".into()),
                        default: None,
                        minimum: Some(1.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "r_max".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Maximum real-space distance".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: Some("nm".into()),
                        choices: None,
                    },
                ),
                (
                    "q_bins".into(),
                    WarpMdFieldSpec {
                        field_type: "integer".into(),
                        semantic_type: "integer".into(),
                        description: Some("Number of q-space bins".into()),
                        default: None,
                        minimum: Some(1.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "q_max".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Maximum q value".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "pbc".into(),
                    WarpMdFieldSpec {
                        field_type: "boolean".into(),
                        semantic_type: "boolean".into(),
                        description: Some("Apply periodic boundary conditions".into()),
                        default: Some(serde_json::json!(true)),
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "histogram".into(),
                    format: "npz".into(),
                    fields: vec!["q_inv_nm".into(), "structure_factor".into()],
                    description: Some("Structure factor vs q".into()),
                }
            ],
            tags: vec!["structural".into()],
            examples: vec![
                serde_json::json!({"name": "structure_factor", "selection": "resname SOL", "bins": 200, "r_max": 1.0, "q_bins": 200, "q_max": 30.0})
            ],
        },
        WarpMdAnalysisContract {
            name: "surf".into(),
            aliases: vec!["surface-area".into(), "sas".into()],
            description: "Solvent accessible surface area".into(),
            required_fields: vec![],
            optional_fields: vec!["mask".into(), "algorithm".into(), "probe_radius".into(), "n_sphere_points".into(), "radii".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "mask".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "mask".into(),
                        description: Some("Atom mask".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "algorithm".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "string".into(),
                        description: Some("Surface area algorithm".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "probe_radius".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Probe radius".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: Some("nm".into()),
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "timeseries".into(),
                    format: "npz".into(),
                    fields: vec!["time_ps".into(), "surface_area_nm2".into()],
                    description: Some("Surface area vs time".into()),
                }
            ],
            tags: vec!["structural".into(), "surface".into()],
            examples: vec![
                serde_json::json!({"name": "surf", "mask": "protein"})
            ],
        },
        native_matrix_contract(
            "drid",
            &["distribution-reciprocal-distance"],
            "Distribution of reciprocal interatomic distance descriptors",
            &["selection"],
            &["exclude_bonds", "frame_indices"],
            &[
                (
                    "selection",
                    "string",
                    "selection",
                    "Atom selection used to compute per-atom DRID moments",
                    None,
                    None,
                    None,
                ),
                (
                    "exclude_bonds",
                    "boolean",
                    "boolean",
                    "Exclude directly bonded selected atom pairs",
                    Some(serde_json::json!(true)),
                    None,
                    None,
                ),
                (
                    "frame_indices",
                    "array",
                    "integer",
                    "Optional zero-based frame subset",
                    None,
                    None,
                    None,
                ),
            ],
            "table",
            &["values"],
            &["geometry", "descriptor", "native"],
            serde_json::json!({"name": "drid", "selection": "protein", "exclude_bonds": true}),
        ),
        native_matrix_contract(
            "shape_descriptors",
            &["shape-descriptors", "principal-moments"],
            "Gyration-tensor shape descriptors",
            &["selection"],
            &["mass", "frame_indices"],
            &[
                (
                    "selection",
                    "string",
                    "selection",
                    "Atom selection for gyration tensor",
                    None,
                    None,
                    None,
                ),
                (
                    "mass",
                    "boolean",
                    "boolean",
                    "Mass-weight the gyration tensor",
                    Some(serde_json::json!(false)),
                    None,
                    None,
                ),
                (
                    "frame_indices",
                    "array",
                    "integer",
                    "Optional zero-based frame subset",
                    None,
                    None,
                    None,
                ),
            ],
            "table",
            &[
                "rg",
                "principal_moments",
                "asphericity",
                "acylindricity",
                "relative_shape_anisotropy",
            ],
            &["geometry", "shape", "native"],
            serde_json::json!({"name": "shape_descriptors", "selection": "protein", "mass": false}),
        ),
        native_matrix_contract(
            "runningavg",
            &["running-average"],
            "Coordinate running average",
            &["selection"],
            &["window", "frame_indices"],
            &[
                (
                    "selection",
                    "string",
                    "selection",
                    "Atom selection to average",
                    None,
                    None,
                    None,
                ),
                (
                    "window",
                    "integer",
                    "integer",
                    "Fixed running-average window; omit for cumulative average",
                    None,
                    Some(0.0),
                    None,
                ),
                (
                    "frame_indices",
                    "array",
                    "integer",
                    "Optional zero-based frame subset",
                    None,
                    None,
                    None,
                ),
            ],
            "artifact",
            &["coords"],
            &["trajectory", "transform", "native"],
            serde_json::json!({"name": "runningavg", "selection": "protein", "window": 10}),
        ),
        native_matrix_contract(
            "lineardensity",
            &["linear-density"],
            "One-dimensional density profile along a coordinate axis",
            &["selection"],
            &[
                "axis",
                "bin",
                "range",
                "weight",
                "norm",
                "charges",
                "cross_section_area",
                "length_scale",
                "frame_indices",
            ],
            &[
                (
                    "selection",
                    "string",
                    "selection",
                    "Atom selection to bin",
                    None,
                    None,
                    None,
                ),
                (
                    "axis",
                    "string",
                    "string",
                    "Axis name: x, y, or z",
                    Some(serde_json::json!("z")),
                    None,
                    None,
                ),
                (
                    "bin",
                    "float",
                    "float",
                    "Bin width",
                    Some(serde_json::json!(1.0)),
                    Some(0.0),
                    None,
                ),
                (
                    "range",
                    "array",
                    "vector",
                    "Optional lower and upper coordinate bounds",
                    None,
                    None,
                    None,
                ),
                (
                    "weight",
                    "string",
                    "string",
                    "Profile weight: number, mass, or charge",
                    Some(serde_json::json!("number")),
                    None,
                    None,
                ),
                (
                    "norm",
                    "string",
                    "string",
                    "Profile normalization: count or density",
                    Some(serde_json::json!("count")),
                    None,
                    None,
                ),
                (
                    "charges",
                    "array",
                    "charges",
                    "Per-atom charges for charge-weighted profiles",
                    None,
                    None,
                    None,
                ),
                (
                    "cross_section_area",
                    "float",
                    "float",
                    "Cross-section area for density normalization",
                    None,
                    Some(0.0),
                    None,
                ),
                (
                    "length_scale",
                    "float",
                    "float",
                    "Coordinate length scale",
                    Some(serde_json::json!(1.0)),
                    Some(0.0),
                    None,
                ),
                (
                    "frame_indices",
                    "array",
                    "integer",
                    "Optional zero-based frame subset",
                    None,
                    None,
                    None,
                ),
            ],
            "table",
            &["axis", "profile", "mean_weight"],
            &["density", "spatial", "native"],
            serde_json::json!({"name": "lineardensity", "selection": "resname SOL", "axis": "z", "bin": 0.1}),
        ),
        native_matrix_contract(
            "nematic_order",
            &["nematic-order", "directors"],
            "Nematic order and director from atom-pair vectors",
            &["selection"],
            &["reference_axis", "pbc", "length_scale", "frame_indices"],
            &[
                (
                    "selection",
                    "string",
                    "selection",
                    "Even atom selection interpreted as adjacent tail/head pairs",
                    None,
                    None,
                    None,
                ),
                (
                    "reference_axis",
                    "array",
                    "vector",
                    "Optional reference axis for signed director and axis order",
                    None,
                    None,
                    None,
                ),
                (
                    "pbc",
                    "boolean",
                    "boolean",
                    "Use orthorhombic minimum-image vectors",
                    Some(serde_json::json!(false)),
                    None,
                    None,
                ),
                (
                    "length_scale",
                    "float",
                    "float",
                    "Coordinate length scale",
                    Some(serde_json::json!(1.0)),
                    Some(0.0),
                    None,
                ),
                (
                    "frame_indices",
                    "array",
                    "integer",
                    "Optional zero-based frame subset",
                    None,
                    None,
                    None,
                ),
            ],
            "table",
            &[
                "time",
                "order",
                "director",
                "q_tensor",
                "axis_order",
                "valid_vectors",
            ],
            &["order", "orientation", "native"],
            serde_json::json!({"name": "nematic_order", "selection": "@1,@2", "reference_axis": [0.0, 0.0, 1.0]}),
        ),
        WarpMdAnalysisContract {
            name: "tordiff".into(),
            aliases: vec!["torsional-diffusion".into()],
            description: "Torsional diffusion coefficient".into(),
            required_fields: vec!["mask".into()],
            optional_fields: vec!["tstep".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "mask".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "mask".into(),
                        description: Some("Atom mask defining torsion".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "tstep".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Time step in picoseconds".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "artifact".into(),
                    format: "npz".into(),
                    fields: vec!["torsional_diffusion".into()],
                    description: Some("Torsional diffusion coefficient".into()),
                }
            ],
            tags: vec!["dynamic".into(), "diffusion".into()],
            examples: vec![
                serde_json::json!({"name": "tordiff", "mask": "dihedral selection"})
            ],
        },
        WarpMdAnalysisContract {
            name: "volmap".into(),
            aliases: vec!["volumetric-map".into(), "density-map".into()],
            description: "Volumetric density map".into(),
            required_fields: vec![],
            optional_fields: vec!["mask".into(), "grid_spacing".into(), "size".into(), "center".into(), "buffer".into(), "centermask".into(), "radscale".into(), "peakcut".into(), "dtype".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "mask".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "mask".into(),
                        description: Some("Atom mask".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "grid_spacing".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Grid spacing".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: Some("nm".into()),
                        choices: None,
                    },
                ),
                (
                    "size".into(),
                    WarpMdFieldSpec {
                        field_type: "array".into(),
                        semantic_type: "vector".into(),
                        description: Some("Grid dimensions (nx, ny, nz)".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "center".into(),
                    WarpMdFieldSpec {
                        field_type: "array".into(),
                        semantic_type: "vector".into(),
                        description: Some("Grid center (x, y, z)".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "grid".into(),
                    format: "dx".into(),
                    fields: vec!["density".into()],
                    description: Some("3D volumetric density".into()),
                }
            ],
            tags: vec!["spatial".into(), "grid".into(), "solvent".into()],
            examples: vec![
                serde_json::json!({"name": "volmap", "mask": "resname SOL and name OW", "grid_spacing": 0.1})
            ],
        },
        WarpMdAnalysisContract {
            name: "water_count".into(),
            aliases: vec!["water-count".into()],
            description: "Water molecule count in spatial regions".into(),
            required_fields: vec!["water_selection".into(), "center_selection".into(), "box_unit".into(), "region_size".into()],
            optional_fields: vec!["shift".into(), "length_scale".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "water_selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Water molecule selection".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "center_selection".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "selection".into(),
                        description: Some("Center point selection".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "box_unit".into(),
                    WarpMdFieldSpec {
                        field_type: "array".into(),
                        semantic_type: "vector".into(),
                        description: Some("Box dimensions (x, y, z)".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "region_size".into(),
                    WarpMdFieldSpec {
                        field_type: "array".into(),
                        semantic_type: "vector".into(),
                        description: Some("Region dimensions (x, y, z)".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "shift".into(),
                    WarpMdFieldSpec {
                        field_type: "array".into(),
                        semantic_type: "vector".into(),
                        description: Some("Origin shift (x, y, z)".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "table".into(),
                    format: "npz".into(),
                    fields: vec!["counts".into()],
                    description: Some("Water counts per region".into()),
                }
            ],
            tags: vec!["solvent".into(), "spatial".into()],
            examples: vec![
                serde_json::json!({"name": "water_count", "water_selection": "resname SOL", "center_selection": "resname LIG", "box_unit": [0.5, 0.5, 0.5], "region_size": [0.5, 0.5, 0.5]})
            ],
        },
        WarpMdAnalysisContract {
            name: "watershell".into(),
            aliases: vec!["water-shell".into(), "solvation-shell".into()],
            description: "Water shell analysis around solute".into(),
            required_fields: vec!["solute_mask".into()],
            optional_fields: vec!["solvent_mask".into(), "lower".into(), "upper".into(), "image".into()],
            fields: std::collections::BTreeMap::from([
                (
                    "solute_mask".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "mask".into(),
                        description: Some("Solute atom mask".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "solvent_mask".into(),
                    WarpMdFieldSpec {
                        field_type: "string".into(),
                        semantic_type: "mask".into(),
                        description: Some("Solvent atom mask".into()),
                        default: None,
                        minimum: None,
                        maximum: None,
                        unit: None,
                        choices: None,
                    },
                ),
                (
                    "lower".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Lower distance cutoff".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: Some("nm".into()),
                        choices: None,
                    },
                ),
                (
                    "upper".into(),
                    WarpMdFieldSpec {
                        field_type: "float".into(),
                        semantic_type: "float".into(),
                        description: Some("Upper distance cutoff".into()),
                        default: None,
                        minimum: Some(0.0),
                        maximum: None,
                        unit: Some("nm".into()),
                        choices: None,
                    },
                )
            ]),
            outputs: vec![
                WarpMdArtifactSpec {
                    kind: "timeseries".into(),
                    format: "npz".into(),
                    fields: vec!["time_ps".into(), "water_count".into()],
                    description: Some("Water count in shell vs time".into()),
                }
            ],
            tags: vec!["solvent".into(), "solvation".into()],
            examples: vec![
                serde_json::json!({"name": "watershell", "solute_mask": "protein", "lower": 0.3, "upper": 0.5})
            ],
        }
        ],
    }
}
