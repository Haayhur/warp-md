use anyhow::{anyhow, Result};

use super::super::{BondedTermSource, CandidateTrajectoryExtractionRequest};

pub(super) fn validate_candidate_trajectory_extraction(
    extraction: &CandidateTrajectoryExtractionRequest,
    field: &str,
) -> Result<()> {
    let bead_count = extraction.mapping.bead_names.len();
    if bead_count == 0 {
        return Err(anyhow!("{field}.mapping.bead_names must not be empty"));
    }
    if extraction.mapping.atom_indices.len() != bead_count {
        return Err(anyhow!(
            "{field}.mapping.atom_indices length must match mapping.bead_names length"
        ));
    }
    for (idx, name) in extraction.mapping.bead_names.iter().enumerate() {
        if name.trim().is_empty() {
            return Err(anyhow!(
                "{field}.mapping.bead_names[{idx}] must not be empty"
            ));
        }
    }
    for (idx, group) in extraction.mapping.atom_indices.iter().enumerate() {
        if group.is_empty() {
            return Err(anyhow!(
                "{field}.mapping.atom_indices[{idx}] must not be empty"
            ));
        }
    }
    for [i, j] in &extraction.connections {
        if *i >= bead_count || *j >= bead_count {
            return Err(anyhow!(
                "{field}.connections entries must reference valid bead indices"
            ));
        }
    }
    if extraction.connections.is_empty() && extraction.bonded_terms.is_none() {
        return Err(anyhow!(
            "{field} requires connections or bonded_terms for target extraction"
        ));
    }
    if let Some(terms) = &extraction.bonded_terms {
        validate_candidate_bonded_term_source(terms, &format!("{field}.bonded_terms"))?;
    }
    if extraction
        .mapped_trajectory_name
        .as_ref()
        .is_some_and(|name| name.trim().is_empty())
    {
        return Err(anyhow!("{field}.mapped_trajectory_name must not be empty"));
    }
    for (value, name) in [
        (extraction.format.as_ref(), "format"),
        (extraction.topology.as_ref(), "topology"),
        (extraction.topology_format.as_ref(), "topology_format"),
        (extraction.target_selection.as_ref(), "target_selection"),
    ] {
        if value.is_some_and(|value| value.trim().is_empty()) {
            return Err(anyhow!("{field}.{name} must not be empty"));
        }
    }
    validate_positive_usize(extraction.stride, &format!("{field}.stride"))?;
    validate_positive_usize(extraction.chunk_frames, &format!("{field}.chunk_frames"))?;
    if extraction
        .length_scale
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err(anyhow!("{field}.length_scale must be finite and positive"));
    }
    if extraction
        .start
        .zip(extraction.stop)
        .is_some_and(|(start, stop)| start >= stop)
    {
        return Err(anyhow!("{field}.start must be less than stop"));
    }
    Ok(())
}

fn validate_candidate_bonded_term_source(source: &BondedTermSource, field: &str) -> Result<()> {
    if source.kind != "gromacs_topology" && source.kind != "gromacs_itp" {
        return Err(anyhow!(
            "{field}.kind must be gromacs_topology or gromacs_itp"
        ));
    }
    if source.path.trim().is_empty() {
        return Err(anyhow!("{field}.path is required"));
    }
    if source.molecule_type.trim().is_empty() {
        return Err(anyhow!("{field}.molecule_type is required"));
    }
    Ok(())
}

fn validate_positive_usize(value: Option<usize>, field: &str) -> Result<()> {
    if value == Some(0) {
        return Err(anyhow!("{field} must be greater than zero"));
    }
    Ok(())
}
