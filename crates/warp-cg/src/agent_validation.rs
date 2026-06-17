use anyhow::{anyhow, Result};

#[path = "agent_validation_candidate_extraction.rs"]
mod candidate_extraction;

use candidate_extraction::{validate_candidate_trajectory_extraction, validate_sasa_request};

use crate::forcefield::validate_request_forcefield;

use super::{
    validate_positive, BoTrainingSetPolicyRequest, BoTuningRequest, BondedClassingRequest,
    BondedTermSource, BondingPolicyRequest, CgRequest, CgSource, ChemistryHintRequest,
    ChemistryPolicyRequest, CoordinateEmissionRequest, DihedralEmissionRequest,
    ExclusionEmissionRequest, JsonFileEvaluatorRequest, MetricScoringRequest,
    ObjectiveEvaluatorRequest, ParameterTuningRequest, PolymerPolicyRequest,
    PrecomputedReferenceRequest, ReferenceMetricSourceRequest, ReferenceTransformRequest,
    SimulationRunnerRequest, XtbRequest, AGENT_SCHEMA_VERSION,
};

pub(super) fn validate_request(request: CgRequest) -> Result<CgRequest> {
    if request.schema_version != AGENT_SCHEMA_VERSION {
        return Err(anyhow!(
            "schema_version must be {AGENT_SCHEMA_VERSION}, got {}",
            request.schema_version
        ));
    }
    if request.name.trim().is_empty() {
        return Err(anyhow!("name is required"));
    }
    if request
        .smiles
        .as_ref()
        .is_some_and(|smiles| smiles.trim().is_empty())
    {
        return Err(anyhow!("smiles must not be empty"));
    }
    if request
        .repeat_smiles
        .as_ref()
        .is_some_and(|smiles| smiles.trim().is_empty())
    {
        return Err(anyhow!("repeat_smiles must not be empty"));
    }
    let identity_count = [
        request.smiles.as_ref().map(|_| "smiles"),
        request.repeat_smiles.as_ref().map(|_| "repeat_smiles"),
        request.source.as_ref().map(|_| "source"),
    ]
    .into_iter()
    .flatten()
    .count();
    let trajectory_ndx_reference = request.trajectory_source.is_some()
        && request
            .mapping
            .as_ref()
            .is_some_and(|mapping| mapping.mode == "ndx");
    if identity_count == 0 && !trajectory_ndx_reference {
        return Err(anyhow!(
            "request requires one of smiles, repeat_smiles, source, or trajectory_source with mapping.mode=ndx"
        ));
    }
    if let Some(source) = &request.source {
        validate_source_shape(source)?;
    }
    if let Some(bonding) = &request.bonding {
        validate_bonding_policy(bonding)?;
    }
    for hint in &request.chemistry_hints {
        validate_chemistry_hint(hint)?;
    }
    if let Some(policy) = &request.chemistry_policy {
        validate_chemistry_policy(policy)?;
    }
    if let Some(polymer) = &request.polymer {
        validate_polymer_policy(polymer)?;
    }
    if let Some(mapping) = &request.mapping {
        if mapping.mode.trim().is_empty() {
            return Err(anyhow!("mapping.mode must not be empty"));
        }
        if !matches!(mapping.mode.as_str(), "auto" | "template" | "ndx") {
            return Err(anyhow!("mapping.mode must be auto, template, or ndx"));
        }
        if mapping.mode == "template" && mapping.template.is_none() && request.source.is_some() {
            return Err(anyhow!("mapping.mode=template requires mapping.template"));
        }
        if let Some(policy) = mapping.template_policy.as_deref() {
            if !matches!(policy, "strict_graph" | "assignment_only") {
                return Err(anyhow!(
                    "mapping.template_policy must be strict_graph or assignment_only"
                ));
            }
        }
        if let Some(on_mismatch) = mapping.on_bead_count_mismatch.as_deref() {
            if !matches!(on_mismatch, "warn" | "error") {
                return Err(anyhow!(
                    "mapping.on_bead_count_mismatch must be warn or error"
                ));
            }
        }
        for (role, count) in &mapping.expected_beads_per_role {
            if !matches!(role.as_str(), "head" | "middle" | "tail" | "standalone") {
                return Err(anyhow!(
                    "mapping.expected_beads_per_role keys must be head, middle, tail, or standalone"
                ));
            }
            if *count == 0 {
                return Err(anyhow!(
                    "mapping.expected_beads_per_role values must be greater than zero"
                ));
            }
        }
        if mapping.mode == "ndx" && mapping.ndx.is_none() {
            return Err(anyhow!("mapping.mode=ndx requires mapping.ndx"));
        }
        if mapping.mode == "ndx" && request.source.is_none() && request.trajectory_source.is_none()
        {
            return Err(anyhow!(
                "mapping.mode=ndx requires source or trajectory_source"
            ));
        }
        if mapping
            .template
            .as_ref()
            .is_some_and(|template| template.trim().is_empty())
        {
            return Err(anyhow!("mapping.template must not be empty"));
        }
        if mapping
            .ndx
            .as_ref()
            .is_some_and(|ndx| ndx.trim().is_empty())
        {
            return Err(anyhow!("mapping.ndx must not be empty"));
        }
        if mapping.target_bead_size.is_some_and(|size| size == 0) {
            return Err(anyhow!(
                "mapping.target_bead_size must be greater than zero"
            ));
        }
        if mapping
            .repeat_unit_hint
            .as_ref()
            .is_some_and(|hint| hint.trim().is_empty())
        {
            return Err(anyhow!("mapping.repeat_unit_hint must not be empty"));
        }
        if let Some(classing) = &mapping.bonded_classing {
            validate_bonded_classing(classing)?;
        }
    }
    if request
        .topology
        .as_ref()
        .is_some_and(|topology| topology.trim().is_empty())
    {
        return Err(anyhow!("topology must not be empty"));
    }
    if request.output.out_dir.trim().is_empty() {
        return Err(anyhow!("output.out_dir must not be empty"));
    }
    if request
        .output
        .mapped_trajectory
        .as_ref()
        .is_some_and(|path| path.trim().is_empty())
    {
        return Err(anyhow!("output.mapped_trajectory must not be empty"));
    }
    if request
        .output
        .cg_pdb
        .as_ref()
        .is_some_and(|path| path.trim().is_empty())
    {
        return Err(anyhow!("output.cg_pdb must not be empty"));
    }
    if request.output.write_topology_top && !request.output.write_topology_itp {
        return Err(anyhow!(
            "output.write_topology_top requires output.write_topology_itp because the .top includes the generated .itp"
        ));
    }
    if let Some(exclusions) = &request.output.exclusions {
        validate_exclusion_emission(exclusions)?;
    }
    if let Some(dihedrals) = &request.output.dihedrals {
        validate_dihedral_emission(dihedrals)?;
    }
    if let Some(coordinates) = &request.output.coordinates {
        validate_coordinate_emission(coordinates)?;
    }
    if let Some(forcefield) = &request.forcefield {
        validate_request_forcefield(forcefield)?;
    }
    if request.output.mapped_trajectory.is_some() {
        let has_xtb_reference = request
            .reference_source
            .as_ref()
            .is_some_and(|source| source.kind == "xtb");
        let has_source_trajectory = request
            .source
            .as_ref()
            .and_then(|source| source.trajectory.as_ref())
            .is_some();
        if request.trajectory_source.is_none() && !has_source_trajectory && !has_xtb_reference {
            return Err(anyhow!(
                "output.mapped_trajectory requires trajectory_source, source.trajectory, or reference_source.kind=xtb"
            ));
        }
    }
    if let Some(source) = &request.trajectory_source {
        if source.path.trim().is_empty() {
            return Err(anyhow!("trajectory_source.path is required"));
        }
        if source
            .topology
            .as_ref()
            .is_some_and(|topology| topology.trim().is_empty())
        {
            return Err(anyhow!("trajectory_source.topology must not be empty"));
        }
        if source.kind != "external" {
            return Err(anyhow!("trajectory_source.kind must be external"));
        }
        if source.stride == Some(0) {
            return Err(anyhow!(
                "trajectory_source.stride must be greater than zero"
            ));
        }
        if source.stop.is_some() && source.start.is_some() && source.stop <= source.start {
            return Err(anyhow!("trajectory_source.stop must be greater than start"));
        }
        if source
            .atom_indices
            .as_ref()
            .is_some_and(|indices| indices.is_empty())
        {
            return Err(anyhow!("trajectory_source.atom_indices must not be empty"));
        }
        if source.target_selection.is_some() && source.atom_indices.is_some() {
            return Err(anyhow!(
                "use either trajectory_source.target_selection or atom_indices, not both"
            ));
        }
        if source
            .target_selection
            .as_ref()
            .is_some_and(|selection| selection.trim().is_empty())
        {
            return Err(anyhow!(
                "trajectory_source.target_selection must not be empty"
            ));
        }
        if source
            .environment_selection
            .as_ref()
            .is_some_and(|selection| selection.trim().is_empty())
        {
            return Err(anyhow!(
                "trajectory_source.environment_selection must not be empty"
            ));
        }
        if source.mass_weighted == Some(true)
            && source.topology.is_none()
            && request.topology.is_none()
        {
            return Err(anyhow!(
                "trajectory_source.mass_weighted requires trajectory_source.topology or top-level topology"
            ));
        }
        if let Some(sasa) = &source.sasa {
            validate_sasa_request(sasa, "trajectory_source.sasa")?;
        }
    }
    if let Some(reference) = &request.reference_source {
        if reference.kind != "external"
            && reference.kind != "xtb"
            && reference.kind != "precomputed"
        {
            return Err(anyhow!(
                "reference_source.kind must be external, xtb, or precomputed"
            ));
        }
        if let Some(xtb) = &reference.xtb {
            validate_xtb_request(xtb, "reference_source.xtb")?;
        }
        if let Some(precomputed) = &reference.precomputed {
            validate_precomputed_reference(precomputed)?;
        }
        if let Some(source) = &reference.bonded_terms {
            validate_bonded_term_source(source)?;
        }
        for source in &reference.metrics {
            validate_reference_metric_source(source)?;
        }
        if let Some(transform) = &reference.transform {
            validate_reference_transform(transform)?;
        }
    }
    if let Some(tuning) = &request.optimization {
        validate_tuning_request(tuning, "optimization", &request)?;
    }
    Ok(request)
}

fn validate_exclusion_emission(exclusions: &ExclusionEmissionRequest) -> Result<()> {
    if !matches!(
        exclusions.mode.as_str(),
        "none" | "nrexcl" | "explicit_nhop" | "explicit_all_intra"
    ) {
        return Err(anyhow!(
            "output.exclusions.mode must be none, nrexcl, explicit_nhop, or explicit_all_intra"
        ));
    }
    if exclusions.n_hops.is_some_and(|value| value == 0) {
        return Err(anyhow!(
            "output.exclusions.n_hops must be greater than zero"
        ));
    }
    Ok(())
}

fn validate_dihedral_emission(_dihedrals: &DihedralEmissionRequest) -> Result<()> {
    Ok(())
}

fn validate_coordinate_emission(_coordinates: &CoordinateEmissionRequest) -> Result<()> {
    Ok(())
}

fn validate_bonded_term_source(source: &BondedTermSource) -> Result<()> {
    if source.kind != "gromacs_topology" && source.kind != "gromacs_itp" {
        return Err(anyhow!(
            "reference_source.bonded_terms.kind must be gromacs_topology or gromacs_itp"
        ));
    }
    if source.path.trim().is_empty() {
        return Err(anyhow!("reference_source.bonded_terms.path is required"));
    }
    if source.molecule_type.trim().is_empty() {
        return Err(anyhow!(
            "reference_source.bonded_terms.molecule_type is required"
        ));
    }
    Ok(())
}

fn validate_precomputed_reference(source: &PrecomputedReferenceRequest) -> Result<()> {
    if source.target_set.trim().is_empty() {
        return Err(anyhow!(
            "reference_source.precomputed.target_set is required"
        ));
    }
    if source
        .source_kind
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(anyhow!(
            "reference_source.precomputed.source_kind must not be empty"
        ));
    }
    Ok(())
}

fn validate_bonded_classing(classing: &BondedClassingRequest) -> Result<()> {
    if !matches!(classing.mode.as_str(), "auto" | "explicit" | "patch") {
        return Err(anyhow!(
            "mapping.bonded_classing.mode must be auto, explicit, or patch"
        ));
    }
    if let Some(source) = classing.source.as_deref() {
        if source.trim().is_empty() {
            return Err(anyhow!("mapping.bonded_classing.source must not be empty"));
        }
    }
    if let Some(base) = classing.base.as_deref() {
        if base != "auto" {
            return Err(anyhow!("mapping.bonded_classing.base must be auto"));
        }
    }
    if let Some(policy) = classing.on_unclassified.as_deref() {
        if !matches!(policy, "auto" | "singleton" | "drop" | "error") {
            return Err(anyhow!(
                "mapping.bonded_classing.on_unclassified must be auto, singleton, drop, or error"
            ));
        }
    }
    if let Some(policy) = classing.on_duplicate_member.as_deref() {
        if !matches!(policy, "error" | "allow") {
            return Err(anyhow!(
                "mapping.bonded_classing.on_duplicate_member must be error or allow"
            ));
        }
    }
    for (idx, class) in classing.bonds.iter().enumerate() {
        validate_class_label(
            &class.label,
            &format!("mapping.bonded_classing.bonds[{idx}].label"),
        )?;
        if class.members.is_empty() {
            return Err(anyhow!(
                "mapping.bonded_classing.bonds[{idx}].members must not be empty"
            ));
        }
    }
    for (idx, class) in classing.angles.iter().enumerate() {
        validate_class_label(
            &class.label,
            &format!("mapping.bonded_classing.angles[{idx}].label"),
        )?;
        if class.members.is_empty() {
            return Err(anyhow!(
                "mapping.bonded_classing.angles[{idx}].members must not be empty"
            ));
        }
    }
    for (idx, class) in classing.dihedrals.iter().enumerate() {
        validate_class_label(
            &class.label,
            &format!("mapping.bonded_classing.dihedrals[{idx}].label"),
        )?;
        if class.members.is_empty() {
            return Err(anyhow!(
                "mapping.bonded_classing.dihedrals[{idx}].members must not be empty"
            ));
        }
    }
    for (idx, merge) in classing.merge.iter().enumerate() {
        validate_class_label(
            &merge.label,
            &format!("mapping.bonded_classing.merge[{idx}].label"),
        )?;
        if merge.from.is_empty() {
            return Err(anyhow!(
                "mapping.bonded_classing.merge[{idx}].from must not be empty"
            ));
        }
    }
    for (idx, rename) in classing.rename.iter().enumerate() {
        validate_class_label(
            &rename.from,
            &format!("mapping.bonded_classing.rename[{idx}].from"),
        )?;
        validate_class_label(
            &rename.to,
            &format!("mapping.bonded_classing.rename[{idx}].to"),
        )?;
    }
    for (idx, split) in classing.split.iter().enumerate() {
        validate_class_label(
            &split.from,
            &format!("mapping.bonded_classing.split[{idx}].from"),
        )?;
        if split.into.is_empty() {
            return Err(anyhow!(
                "mapping.bonded_classing.split[{idx}].into must not be empty"
            ));
        }
        for (target_idx, target) in split.into.iter().enumerate() {
            validate_class_label(
                &target.label,
                &format!("mapping.bonded_classing.split[{idx}].into[{target_idx}].label"),
            )?;
            if target.members.is_empty() {
                return Err(anyhow!(
                    "mapping.bonded_classing.split[{idx}].into[{target_idx}].members must not be empty"
                ));
            }
        }
    }
    Ok(())
}

fn validate_class_label(label: &str, field: &str) -> Result<()> {
    if label.trim().is_empty() {
        return Err(anyhow!("{field} must not be empty"));
    }
    Ok(())
}

fn validate_reference_metric_source(source: &ReferenceMetricSourceRequest) -> Result<()> {
    if source.kind != "json" {
        return Err(anyhow!("reference_source.metrics.kind must be json"));
    }
    if source.path.trim().is_empty() {
        return Err(anyhow!("reference_source.metrics.path is required"));
    }
    Ok(())
}

fn validate_reference_transform(transform: &ReferenceTransformRequest) -> Result<()> {
    validate_positive(
        transform.bond_scaling,
        "reference_source.transform.bond_scaling",
    )?;
    validate_positive(
        transform.min_bond_length_nm,
        "reference_source.transform.min_bond_length_nm",
    )?;
    for (key, value) in &transform.specific_bond_lengths_nm {
        if key.trim().is_empty() {
            return Err(anyhow!(
                "reference_source.transform.specific_bond_lengths_nm keys must not be empty"
            ));
        }
        validate_positive(
            Some(*value),
            "reference_source.transform.specific_bond_lengths_nm values",
        )?;
    }
    if transform
        .rg_offset_nm
        .is_some_and(|value| !value.is_finite())
    {
        return Err(anyhow!(
            "reference_source.transform.rg_offset_nm must be finite"
        ));
    }
    Ok(())
}

fn validate_source_shape(source: &CgSource) -> Result<()> {
    if source.kind.trim().is_empty() {
        return Err(anyhow!("source.kind is required"));
    }
    let valid_kind = matches!(
        source.kind.as_str(),
        "structure"
            | "polymer_build_manifest"
            | "polymer_pack_manifest"
            | "coordinates_topology"
            | "coordinates_topology_charge_manifest"
            | "source_manifest"
    );
    if !valid_kind {
        return Err(anyhow!(
            "source.kind must be structure, polymer_build_manifest, polymer_pack_manifest, coordinates_topology, coordinates_topology_charge_manifest, or source_manifest"
        ));
    }
    for (field, value) in [
        ("source.path", source.path.as_ref()),
        ("source.coordinates", source.coordinates.as_ref()),
        ("source.topology", source.topology.as_ref()),
        ("source.charge_manifest", source.charge_manifest.as_ref()),
        ("source.trajectory", source.trajectory.as_ref()),
        ("source.target_selection", source.target_selection.as_ref()),
        ("source.selection", source.selection.as_ref()),
    ] {
        if value.is_some_and(|item| item.trim().is_empty()) {
            return Err(anyhow!("{field} must not be empty"));
        }
    }
    if source.selection.is_some() && source.target_selection.is_some() {
        return Err(anyhow!(
            "use either source.selection or source.target_selection, not both"
        ));
    }
    match source.kind.as_str() {
        "structure" => {
            if source.coordinates.is_none() {
                return Err(anyhow!("source.coordinates is required for structure"));
            }
        }
        "polymer_build_manifest" | "polymer_pack_manifest" | "source_manifest" => {
            if source.path.is_none() {
                return Err(anyhow!("source.path is required for {}", source.kind));
            }
        }
        "coordinates_topology" => {
            if source.coordinates.is_none() || source.topology.is_none() {
                return Err(anyhow!(
                    "source.coordinates and source.topology are required for coordinates_topology"
                ));
            }
        }
        "coordinates_topology_charge_manifest" => {
            if source.coordinates.is_none()
                || source.topology.is_none()
                || source.charge_manifest.is_none()
            {
                return Err(anyhow!(
                    "source.coordinates, source.topology, and source.charge_manifest are required for coordinates_topology_charge_manifest"
                ));
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_bonding_policy(policy: &BondingPolicyRequest) -> Result<()> {
    if let Some(source) = policy.source.as_deref() {
        if !matches!(
            source,
            "explicit_topology" | "infer_from_coordinates" | "template"
        ) {
            return Err(anyhow!(
                "bonding.source must be explicit_topology, infer_from_coordinates, or template"
            ));
        }
    }
    if let Some(on_ambiguous) = policy.on_ambiguous.as_deref() {
        if !matches!(on_ambiguous, "warn" | "error") {
            return Err(anyhow!("bonding.on_ambiguous must be warn or error"));
        }
    }
    Ok(())
}

fn validate_chemistry_hint(hint: &ChemistryHintRequest) -> Result<()> {
    if !matches!(hint.kind.as_str(), "smiles" | "template" | "inline_graph") {
        return Err(anyhow!(
            "chemistry_hints[].kind must be smiles, template, or inline_graph"
        ));
    }
    if !matches!(
        hint.scope.as_str(),
        "molecule" | "repeat_unit" | "residue" | "residue_role"
    ) {
        return Err(anyhow!(
            "chemistry_hints[].scope must be molecule, repeat_unit, residue, or residue_role"
        ));
    }
    match hint.kind.as_str() {
        "smiles" | "inline_graph" if hint.value.as_deref().unwrap_or("").trim().is_empty() => Err(
            anyhow!("chemistry_hints[] with kind {} requires value", hint.kind),
        ),
        "template" if hint.path.as_deref().unwrap_or("").trim().is_empty() => Err(anyhow!(
            "chemistry_hints[] with kind template requires path"
        )),
        _ => Ok(()),
    }
}

fn validate_chemistry_policy(policy: &ChemistryPolicyRequest) -> Result<()> {
    if let Some(mode) = policy.hint_mode.as_deref() {
        if !matches!(
            mode,
            "validate" | "fill_missing" | "prefer_hint" | "prefer_geometry"
        ) {
            return Err(anyhow!(
                "chemistry_policy.hint_mode must be validate, fill_missing, prefer_hint, or prefer_geometry"
            ));
        }
    }
    if let Some(on_conflict) = policy.on_conflict.as_deref() {
        if !matches!(on_conflict, "warn" | "error") {
            return Err(anyhow!(
                "chemistry_policy.on_conflict must be warn or error"
            ));
        }
    }
    Ok(())
}

fn validate_polymer_policy(policy: &PolymerPolicyRequest) -> Result<()> {
    if let Some(mode) = policy.role_mode.as_deref() {
        if !matches!(mode, "infer" | "explicit") {
            return Err(anyhow!("polymer.role_mode must be infer or explicit"));
        }
    }
    if let Some(end_group_policy) = policy.end_group_policy.as_deref() {
        if !matches!(end_group_policy, "preserve" | "map_as_repeat") {
            return Err(anyhow!(
                "polymer.end_group_policy must be preserve or map_as_repeat"
            ));
        }
    }
    Ok(())
}

fn validate_xtb_request(xtb: &XtbRequest, field: &str) -> Result<()> {
    if xtb
        .mode
        .as_ref()
        .is_some_and(|mode| mode != "optimize" && mode != "optimize_and_md")
    {
        return Err(anyhow!("{field}.mode must be optimize or optimize_and_md"));
    }
    validate_positive(xtb.temperature_k, &format!("{field}.temperature_k"))?;
    validate_positive(xtb.time_ps, &format!("{field}.time_ps"))?;
    validate_positive(xtb.timestep_fs, &format!("{field}.timestep_fs"))?;
    validate_positive(xtb.dump_fs, &format!("{field}.dump_fs"))?;
    if xtb.gfn.as_ref().is_some_and(|gfn| gfn.trim().is_empty()) {
        return Err(anyhow!("{field}.gfn must not be empty"));
    }
    Ok(())
}

fn validate_tuning_request(
    tuning: &ParameterTuningRequest,
    field: &str,
    request: &CgRequest,
) -> Result<()> {
    if !is_bo_method(&tuning.method) && tuning.method != "pso" {
        return Err(anyhow!(
            "{field}.method must be bayesian_optimization, bo, or pso"
        ));
    }
    if let Some(mode) = tuning.fitting_mode.as_deref() {
        if !matches!(
            mode,
            "auto"
                | "direct_statistics"
                | "distribution_fit"
                | "external_evaluator"
                | "simulation_fit"
        ) {
            return Err(anyhow!(
                "{field}.fitting_mode must be auto, direct_statistics, distribution_fit, external_evaluator, or simulation_fit"
            ));
        }
    }
    if tuning
        .min_samples_per_term
        .is_some_and(|samples| samples == 0)
    {
        return Err(anyhow!(
            "{field}.min_samples_per_term must be greater than zero"
        ));
    }
    if let Some(policy) = tuning.on_insufficient_samples.as_deref() {
        if !matches!(policy, "warn" | "error") {
            return Err(anyhow!(
                "{field}.on_insufficient_samples must be warn or error"
            ));
        }
    }
    if tuning.max_evaluations == Some(0) {
        return Err(anyhow!("{field}.max_evaluations must be greater than zero"));
    }
    if tuning.swarm_size == Some(0) {
        return Err(anyhow!("{field}.swarm_size must be greater than zero"));
    }
    for (name, value) in &tuning.initial_parameters {
        if name.trim().is_empty() {
            return Err(anyhow!("{field}.initial_parameters keys must not be empty"));
        }
        if !value.is_finite() {
            return Err(anyhow!("{field}.initial_parameters.{name} must be finite"));
        }
    }
    if tuning.pso.is_some() && tuning.method != "pso" {
        return Err(anyhow!("{field}.pso requires method pso"));
    }
    if tuning.bo.is_some() && !is_bo_method(&tuning.method) {
        return Err(anyhow!(
            "{field}.bo requires method bayesian_optimization or bo"
        ));
    }
    if let Some(pso) = &tuning.pso {
        if pso.reboot_after_local_stall_iterations == Some(0) {
            return Err(anyhow!(
                "{field}.pso.reboot_after_local_stall_iterations must be greater than zero"
            ));
        }
        if pso
            .restart_strategy
            .as_ref()
            .is_some_and(|strategy| strategy != "random" && strategy != "recombine")
        {
            return Err(anyhow!(
                "{field}.pso.restart_strategy must be random or recombine"
            ));
        }
        if pso.max_iterations_without_global_best == Some(0) {
            return Err(anyhow!(
                "{field}.pso.max_iterations_without_global_best must be greater than zero"
            ));
        }
        if pso.checkpoint_interval_evaluations == Some(0) {
            return Err(anyhow!(
                "{field}.pso.checkpoint_interval_evaluations must be greater than zero"
            ));
        }
        if pso
            .checkpoint_path
            .as_ref()
            .is_some_and(|path| path.trim().is_empty())
        {
            return Err(anyhow!("{field}.pso.checkpoint_path must not be empty"));
        }
        if pso.resume_from_checkpoint == Some(true) && pso.checkpoint_path.is_none() {
            return Err(anyhow!(
                "{field}.pso.resume_from_checkpoint requires checkpoint_path"
            ));
        }
        validate_positive(
            pso.discrete_probability_dilation_alpha,
            &format!("{field}.pso.discrete_probability_dilation_alpha"),
        )?;
    }
    if let Some(bo) = &tuning.bo {
        validate_bo_options(bo, &format!("{field}.bo"))?;
    }
    if tuning.source != "external_trajectory"
        && tuning.source != "aa_trajectory"
        && tuning.source != "xtb"
    {
        return Err(anyhow!(
            "{field}.source must be external_trajectory, aa_trajectory, or xtb"
        ));
    }
    if let Some(terms) = &tuning.target_terms {
        if terms.is_empty() {
            return Err(anyhow!("{field}.target_terms must not be empty"));
        }
        for term in terms {
            if term != "constraints" && term != "bonds" && term != "angles" && term != "dihedrals" {
                return Err(anyhow!(
                    "{field}.target_terms entries must be constraints, bonds, angles, or dihedrals"
                ));
            }
        }
    }
    if let Some(xtb) = &tuning.xtb {
        validate_xtb_request(xtb, &format!("{field}.xtb"))?;
    }
    if let Some(metric_scoring) = &tuning.metric_scoring {
        validate_metric_scoring(metric_scoring, &format!("{field}.metric_scoring"))?;
    }
    if let Some(evaluator) = &tuning.evaluator {
        validate_objective_evaluator(evaluator, field)?;
    }
    if let Some(runner) = &tuning.runner {
        validate_simulation_runner(runner, &format!("{field}.runner"))?;
    }
    if tuning.evaluator.is_some() && tuning.runner.is_some() {
        return Err(anyhow!(
            "{field} accepts either evaluator or runner, not both"
        ));
    }
    if tuning.fitting_mode.as_deref() == Some("simulation_fit") {
        let evaluator_has_candidate_extraction = tuning
            .evaluator
            .as_ref()
            .and_then(|evaluator| evaluator.json_file.as_ref())
            .and_then(|json_file| json_file.candidate_extraction.as_ref())
            .is_some();
        let runner_has_candidate_extraction = tuning
            .runner
            .as_ref()
            .and_then(|runner| runner.candidate_extraction.as_ref())
            .is_some();
        if !evaluator_has_candidate_extraction && !runner_has_candidate_extraction {
            return Err(anyhow!(
                "{field}.fitting_mode=simulation_fit requires evaluator.json_file.candidate_extraction or runner.candidate_extraction"
            ));
        }
        if tuning.runner.as_ref().is_some_and(|runner| {
            runner
                .protocol
                .as_ref()
                .and_then(|protocol| protocol.trajectory_format.as_deref())
                == Some("none")
        }) {
            return Err(anyhow!(
                "{field}.fitting_mode=simulation_fit requires optimization.runner.protocol.trajectory_format to be xtc or dcd"
            ));
        }
    }
    if tuning.enabled
        && (tuning.source == "external_trajectory" || tuning.source == "aa_trajectory")
        && tuning.evaluator.is_none()
        && tuning.runner.is_none()
        && request.trajectory_source.is_none()
        && request
            .source
            .as_ref()
            .and_then(|source| source.trajectory.as_ref())
            .is_none()
    {
        return Err(anyhow!(
            "{field} trajectory tuning requires trajectory_source or source.trajectory"
        ));
    }
    if tuning.enabled
        && tuning.source == "xtb"
        && !request
            .reference_source
            .as_ref()
            .is_some_and(|source| source.kind == "xtb")
    {
        return Err(anyhow!(
            "{field} xtb parameter tuning requires reference_source.kind=xtb"
        ));
    }
    Ok(())
}

fn is_bo_method(method: &str) -> bool {
    matches!(method, "bayesian_optimization" | "bo")
}

fn validate_bo_options(bo: &BoTuningRequest, field: &str) -> Result<()> {
    if bo.n_startup_trials == Some(0) {
        return Err(anyhow!(
            "{field}.n_startup_trials must be greater than zero"
        ));
    }
    if bo.n_candidates == Some(0) {
        return Err(anyhow!("{field}.n_candidates must be greater than zero"));
    }
    validate_positive(bo.noise_variance, &format!("{field}.noise_variance"))?;
    if let Some(policy) = &bo.training_set_policy {
        validate_bo_training_set_policy(policy, &format!("{field}.training_set_policy"))?;
    }
    if let Some(policy) = bo.failure_handling.as_deref() {
        if !matches!(
            policy,
            "penalize"
                | "exclude_from_gp_but_keep_in_history"
                | "exclude"
                | "model_as_constraint_later"
                | "constraint"
        ) {
            return Err(anyhow!(
                "{field}.failure_handling must be penalize, exclude_from_gp_but_keep_in_history, exclude, model_as_constraint_later, or constraint"
            ));
        }
    }
    validate_positive(bo.failure_penalty, &format!("{field}.failure_penalty"))?;
    if bo.checkpoint_interval_evaluations == Some(0) {
        return Err(anyhow!(
            "{field}.checkpoint_interval_evaluations must be greater than zero"
        ));
    }
    if bo
        .checkpoint_path
        .as_ref()
        .is_some_and(|path| path.trim().is_empty())
    {
        return Err(anyhow!("{field}.checkpoint_path must not be empty"));
    }
    if bo.resume_from_checkpoint == Some(true) && bo.checkpoint_path.is_none() {
        return Err(anyhow!(
            "{field}.resume_from_checkpoint requires checkpoint_path"
        ));
    }
    if bo
        .evaluator_signature
        .as_ref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(anyhow!("{field}.evaluator_signature must not be empty"));
    }
    Ok(())
}

fn validate_bo_training_set_policy(policy: &BoTrainingSetPolicyRequest, field: &str) -> Result<()> {
    if policy.max_points == 0 {
        return Err(anyhow!("{field}.max_points must be greater than zero"));
    }
    if policy.keep_best == 0 && policy.keep_recent == 0 && policy.keep_diverse == 0 {
        return Err(anyhow!(
            "{field} must retain at least one of keep_best, keep_recent, or keep_diverse"
        ));
    }
    Ok(())
}

fn validate_metric_scoring(source: &MetricScoringRequest, field: &str) -> Result<()> {
    for (name, value) in [
        ("rg_weight", source.rg_weight),
        ("sasa_weight", source.sasa_weight),
    ] {
        if let Some(value) = value {
            if !value.is_finite() || value < 0.0 {
                return Err(anyhow!("{field}.{name} must be finite and non-negative"));
            }
        }
    }
    validate_positive(
        source.missing_metric_penalty,
        &format!("{field}.missing_metric_penalty"),
    )?;
    Ok(())
}

fn validate_objective_evaluator(evaluator: &ObjectiveEvaluatorRequest, field: &str) -> Result<()> {
    if evaluator.kind != "json_file" {
        return Err(anyhow!("{field}.evaluator.kind must be json_file"));
    }
    let source = evaluator
        .json_file
        .as_ref()
        .ok_or_else(|| anyhow!("{field}.evaluator.json_file is required"))?;
    validate_json_file_evaluator(source, field)
}

fn validate_json_file_evaluator(source: &JsonFileEvaluatorRequest, field: &str) -> Result<()> {
    if source.work_dir.trim().is_empty() {
        return Err(anyhow!("{field}.evaluator.json_file.work_dir is required"));
    }
    if source
        .request_filename
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(anyhow!(
            "{field}.evaluator.json_file.request_filename must not be empty"
        ));
    }
    if source
        .result_filename
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(anyhow!(
            "{field}.evaluator.json_file.result_filename must not be empty"
        ));
    }
    if let Some(command) = &source.command {
        if command.program.trim().is_empty() {
            return Err(anyhow!(
                "{field}.evaluator.json_file.command.program is required"
            ));
        }
    }
    if let Some(extraction) = &source.candidate_extraction {
        validate_candidate_trajectory_extraction(
            extraction,
            &format!("{field}.evaluator.json_file.candidate_extraction"),
        )?;
    }
    Ok(())
}

fn validate_simulation_runner(source: &SimulationRunnerRequest, field: &str) -> Result<()> {
    if source.kind != "martini_openmm" {
        return Err(anyhow!("{field}.kind must be martini_openmm"));
    }
    if source
        .work_dir
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(anyhow!("{field}.work_dir must not be empty"));
    }
    if source
        .python
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(anyhow!("{field}.python must not be empty"));
    }
    if source.gro.trim().is_empty() {
        return Err(anyhow!("{field}.gro is required"));
    }
    if source.top.trim().is_empty() {
        return Err(anyhow!("{field}.top is required"));
    }
    if source
        .template_dir
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(anyhow!("{field}.template_dir must not be empty"));
    }
    for (idx, replacement) in source.replacements.iter().enumerate() {
        let replacement_field = format!("{field}.replacements[{idx}]");
        if replacement.path.trim().is_empty() {
            return Err(anyhow!("{replacement_field}.path is required"));
        }
        if replacement.parameter.trim().is_empty() {
            return Err(anyhow!("{replacement_field}.parameter is required"));
        }
        if replacement
            .placeholder
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(anyhow!("{replacement_field}.placeholder must not be empty"));
        }
        if replacement
            .format
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(anyhow!("{replacement_field}.format must not be empty"));
        }
    }
    if let Some(protocol) = &source.protocol {
        validate_positive(
            protocol.temperature,
            &format!("{field}.protocol.temperature"),
        )?;
        validate_positive(protocol.pressure, &format!("{field}.protocol.pressure"))?;
        if protocol.friction.is_some_and(|value| value < 0.0) {
            return Err(anyhow!("{field}.protocol.friction must be non-negative"));
        }
        validate_positive(
            protocol.eq_timestep_fs,
            &format!("{field}.protocol.eq_timestep_fs"),
        )?;
        validate_positive(
            protocol.prod_timestep_fs,
            &format!("{field}.protocol.prod_timestep_fs"),
        )?;
        validate_positive(protocol.cutoff_nm, &format!("{field}.protocol.cutoff_nm"))?;
        if protocol.eq_ns.is_some_and(|value| value < 0.0) {
            return Err(anyhow!("{field}.protocol.eq_ns must be non-negative"));
        }
        if protocol.prod_ns.is_some_and(|value| value < 0.0) {
            return Err(anyhow!("{field}.protocol.prod_ns must be non-negative"));
        }
        if let Some(ensemble) = protocol.production_ensemble.as_deref() {
            if !matches!(ensemble, "npt" | "nvt") {
                return Err(anyhow!(
                    "{field}.protocol.production_ensemble must be npt or nvt"
                ));
            }
        }
        if let Some(precision) = protocol.precision.as_deref() {
            if !matches!(precision, "single" | "mixed" | "double") {
                return Err(anyhow!(
                    "{field}.protocol.precision must be single, mixed, or double"
                ));
            }
        }
        if let Some(format) = protocol.trajectory_format.as_deref() {
            if !matches!(format, "xtc" | "dcd" | "none") {
                return Err(anyhow!(
                    "{field}.protocol.trajectory_format must be xtc, dcd, or none"
                ));
            }
        }
        for (name, value) in [
            ("cpu_threads", protocol.cpu_threads),
            ("minimize_iterations", protocol.minimize_iterations),
            ("barostat_frequency", protocol.barostat_frequency),
            ("report_interval_steps", protocol.report_interval_steps),
            (
                "trajectory_interval_steps",
                protocol.trajectory_interval_steps,
            ),
            (
                "checkpoint_interval_steps",
                protocol.checkpoint_interval_steps,
            ),
            ("energy_interval_steps", protocol.energy_interval_steps),
            ("status_interval_steps", protocol.status_interval_steps),
        ] {
            if value == Some(0) {
                return Err(anyhow!("{field}.protocol.{name} must be greater than zero"));
            }
        }
        for define in &protocol.defines {
            if define.trim().is_empty() {
                return Err(anyhow!(
                    "{field}.protocol.defines entries must not be empty"
                ));
            }
        }
        if protocol
            .defines_file
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(anyhow!("{field}.protocol.defines_file must not be empty"));
        }
    }
    if let Some(extraction) = &source.candidate_extraction {
        validate_candidate_trajectory_extraction(
            extraction,
            &format!("{field}.candidate_extraction"),
        )?;
    }
    Ok(())
}
