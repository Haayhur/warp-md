use super::{CgOutputRequest, AGENT_SCHEMA_VERSION};

pub(super) fn default_schema_version() -> String {
    AGENT_SCHEMA_VERSION.to_string()
}

pub(super) fn default_out_dir() -> String {
    ".".to_string()
}

pub(super) fn default_write_mapping() -> bool {
    true
}

pub(super) fn default_write_topology_itp() -> bool {
    true
}

pub(super) fn default_write_topology_top() -> bool {
    false
}

pub(super) fn default_write_cg_pdb() -> bool {
    true
}

pub(super) fn default_write_bonded_parameter_map() -> bool {
    true
}

pub(super) fn default_external_trajectory_kind() -> String {
    "external".to_string()
}

pub(super) fn default_reference_kind() -> String {
    "external".to_string()
}

pub(super) fn default_forcefield_kind() -> String {
    "martini3".to_string()
}

pub(super) fn default_forcefield_source() -> String {
    "bundled".to_string()
}

pub(super) fn default_forcefield_materialize() -> String {
    "copy".to_string()
}

pub(super) fn default_bonded_term_source_kind() -> String {
    "gromacs_topology".to_string()
}

pub(super) fn default_reference_metric_kind() -> String {
    "json".to_string()
}

pub(super) fn default_mapping_mode() -> String {
    "auto".to_string()
}

pub(super) fn default_tuning_enabled() -> bool {
    false
}

pub(super) fn default_tuning_source() -> String {
    "external_trajectory".to_string()
}

pub(super) fn default_tuning_method() -> String {
    "bayesian_optimization".to_string()
}

pub(super) fn default_tuning_objective() -> String {
    "bonded_parameter_parity".to_string()
}

pub(super) fn default_objective_evaluator_kind() -> String {
    "json_file".to_string()
}

pub(super) fn default_simulation_runner_kind() -> String {
    "martini_openmm".to_string()
}

pub(super) fn default_output() -> CgOutputRequest {
    CgOutputRequest {
        out_dir: default_out_dir(),
        mapped_trajectory: None,
        write_mapping_json: true,
        write_topology_itp: true,
        write_topology_top: true,
        write_cg_pdb: true,
        cg_pdb: None,
        write_bonded_parameter_map: true,
    }
}
