use super::*;

pub(super) fn resolved_ion_species(
    species: &[IonComponent],
    fallback_name: &str,
    fallback_charge: i32,
) -> Vec<ResolvedIonSpecies> {
    if species.is_empty() {
        let library = lookup_ion_library(fallback_name);
        return vec![ResolvedIonSpecies {
            name: fallback_name.to_string(),
            residue_name: library
                .as_ref()
                .map(|entry| entry.name.to_string())
                .unwrap_or_else(|| ion_residue_name(fallback_name)),
            atom_name: library
                .as_ref()
                .map(|entry| entry.atom_name.to_string())
                .unwrap_or_else(|| ion_atom_name(fallback_name)),
            ratio: 1.0,
            charge_e: library
                .as_ref()
                .filter(|entry| fallback_charge == entry.default_charge)
                .map(|entry| entry.charge_e)
                .unwrap_or(fallback_charge),
        }];
    }
    species
        .iter()
        .map(|ion| {
            let library = lookup_ion_library(&ion.name);
            ResolvedIonSpecies {
                name: ion.name.clone(),
                residue_name: library
                    .as_ref()
                    .map(|entry| entry.name.to_string())
                    .unwrap_or_else(|| ion_residue_name(&ion.name)),
                atom_name: library
                    .as_ref()
                    .map(|entry| entry.atom_name.to_string())
                    .unwrap_or_else(|| ion_atom_name(&ion.name)),
                ratio: ion.ratio,
                charge_e: library
                    .as_ref()
                    .filter(|_| ion.charge_e == 0)
                    .map(|entry| entry.charge_e)
                    .unwrap_or(ion.charge_e),
            }
        })
        .collect()
}

fn coarse_molarity_counts(
    volume_nm3: f32,
    molarity_mol_l: f32,
    ratio_mapping: &[(f32, f32)],
) -> Vec<usize> {
    if volume_nm3 <= 0.0 || molarity_mol_l <= 0.0 || ratio_mapping.is_empty() {
        return vec![0; ratio_mapping.len()];
    }
    let ratio_sum = ratio_mapping.iter().map(|(ratio, _)| *ratio).sum::<f32>();
    if ratio_sum <= 0.0 {
        return vec![0; ratio_mapping.len()];
    }
    let ratios = ratio_mapping
        .iter()
        .map(|(ratio, _)| ((ratio / ratio_sum) * 10_000.0).round() / 10_000.0)
        .collect::<Vec<_>>();
    let weighted_mapping = ratio_mapping
        .iter()
        .zip(ratios.iter())
        .map(|((_, mapping), ratio)| mapping * ratio)
        .sum::<f32>();
    if weighted_mapping <= 0.0 {
        return vec![0; ratio_mapping.len()];
    }
    let number_of_particles = AVOGADRO * volume_nm3 as f64 * 1.0e-24 * molarity_mol_l as f64;
    ratios
        .iter()
        .map(|ratio| {
            (number_of_particles * *ratio as f64 / weighted_mapping as f64 + 0.5).floor() as usize
        })
        .collect()
}

fn mixed_solvent_material_volume_nm3(counts: &[usize], species: &[ResolvedSolventSpecies]) -> f32 {
    counts
        .iter()
        .zip(species.iter())
        .map(|(count, species)| {
            let molar_mass_kg_mol = species.molar_mass_g_mol as f64 * 1.0e-3;
            let volume_m3 = *count as f64 * species.mapping_ratio as f64 * molar_mass_kg_mol
                / (AVOGADRO * species.density_kg_m3 as f64);
            volume_m3 * 1.0e27
        })
        .sum::<f64>() as f32
}

fn ratio_target_counts(target_total: f32, ratios: &[f32]) -> Vec<usize> {
    if target_total <= 0.0 || !target_total.is_finite() || ratios.is_empty() {
        return vec![0; ratios.len()];
    }
    let ratio_sum = ratios.iter().copied().sum::<f32>();
    if ratio_sum <= 0.0 {
        return vec![0; ratios.len()];
    }
    ratios
        .iter()
        .map(|ratio| (target_total * (*ratio / ratio_sum) + 0.5).floor() as usize)
        .collect()
}

pub(super) fn emit_solvent_and_ions(
    request: &BuildRequest,
    net_charge: Option<f32>,
    neutralization: &mut NeutralizationSummary,
    emitted_beads: &mut Vec<EmittedBead>,
    next_residue_id: &mut i32,
) -> Result<SolventEmission> {
    let solvent = &request.environment.solvent;
    let ions = &request.environment.ions;
    if !solvent.zones.is_empty() {
        let mut aggregate = SolventEmission {
            counts: BTreeMap::new(),
            solvent_charge_e: 0.0,
            baseline_ion_charge_e: 0.0,
            neutralization_input_charge_e: 0.0,
            summary: SolventPlacementSummary {
                algorithm: solvent_algorithm_name(&request.system, true),
                mode: request.system.placement.mode.clone(),
                random_seed: request.system.placement.random_seed,
                box_volume_nm3: 0.0,
                excluded_volume_nm3: 0.0,
                free_volume_nm3: 0.0,
                solvent_material_volume_nm3: 0.0,
                grid_point_count: 0,
                inserted_count: 0,
                grid_squeeze_pass_count: 0,
                squeezed_candidate_count: 0,
                min_grid_spacing_angstrom: None,
                kick_attempt_count: 0,
                kicked_inserted_count: 0,
                density: PlacementPhaseDensitySummary::default(),
            },
        };
        for (zone_idx, zone) in solvent.zones.iter().enumerate() {
            let zone_solvent = solvent_policy_for_zone(solvent, zone);
            let mut zone_ions = ions.clone();
            if let Some(salt_molarity) = zone.salt_molarity_mol_l {
                zone_ions.salt_molarity_mol_l = salt_molarity;
            }
            let mut zone_neutralization = if zone_idx == 0 {
                neutralization.clone()
            } else {
                NeutralizationSummary {
                    enabled: false,
                    salt_method: ions.salt_method.clone(),
                    counterion: None,
                    counterion_count: 0,
                    counterion_charge_e: None,
                    cation_delta: 0,
                    anion_delta: 0,
                    residual_charge_e: None,
                }
            };
            let zone_emission = emit_solvent_zone_and_ions(
                request,
                &zone_solvent,
                &zone_ions,
                if zone_idx == 0 { net_charge } else { None },
                &mut zone_neutralization,
                emitted_beads,
                next_residue_id,
            )?;
            if zone_idx == 0 {
                *neutralization = zone_neutralization;
            }
            for (name, count) in zone_emission.counts {
                *aggregate.counts.entry(name).or_insert(0) += count;
            }
            aggregate.solvent_charge_e += zone_emission.solvent_charge_e;
            aggregate.baseline_ion_charge_e += zone_emission.baseline_ion_charge_e;
            aggregate.neutralization_input_charge_e += zone_emission.neutralization_input_charge_e;
            aggregate.summary.box_volume_nm3 += zone_emission.summary.box_volume_nm3;
            aggregate.summary.excluded_volume_nm3 += zone_emission.summary.excluded_volume_nm3;
            aggregate.summary.free_volume_nm3 += zone_emission.summary.free_volume_nm3;
            aggregate.summary.solvent_material_volume_nm3 +=
                zone_emission.summary.solvent_material_volume_nm3;
            aggregate.summary.grid_point_count += zone_emission.summary.grid_point_count;
            aggregate.summary.inserted_count += zone_emission.summary.inserted_count;
            aggregate.summary.grid_squeeze_pass_count +=
                zone_emission.summary.grid_squeeze_pass_count;
            aggregate.summary.squeezed_candidate_count +=
                zone_emission.summary.squeezed_candidate_count;
            aggregate.summary.min_grid_spacing_angstrom = match (
                aggregate.summary.min_grid_spacing_angstrom,
                zone_emission.summary.min_grid_spacing_angstrom,
            ) {
                (Some(current), Some(next)) => Some(current.min(next)),
                (None, Some(next)) => Some(next),
                (current, None) => current,
            };
            aggregate.summary.kick_attempt_count += zone_emission.summary.kick_attempt_count;
            aggregate.summary.kicked_inserted_count += zone_emission.summary.kicked_inserted_count;
            aggregate.summary.density.target_count += zone_emission.summary.density.target_count;
            aggregate.summary.density.placed_count += zone_emission.summary.density.placed_count;
            aggregate.summary.density.initial_candidate_count +=
                zone_emission.summary.density.initial_candidate_count;
            aggregate.summary.density.final_candidate_count +=
                zone_emission.summary.density.final_candidate_count;
            finalize_phase_density(
                &mut aggregate.summary.density,
                aggregate.summary.grid_squeeze_pass_count,
            );
        }
        return Ok(aggregate);
    }
    emit_solvent_zone_and_ions(
        request,
        solvent,
        ions,
        net_charge,
        neutralization,
        emitted_beads,
        next_residue_id,
    )
}

fn solvent_policy_for_zone(base: &SolventPolicy, zone: &SolventZone) -> SolventPolicy {
    let mut solvent = base.clone();
    solvent.zones.clear();
    if let Some(box_size) = zone.box_size_angstrom {
        solvent.box_size_angstrom = Some(box_size);
    }
    if let Some(center) = zone.center_angstrom {
        solvent.center_angstrom = Some(center);
    }
    if let Some(molarity) = zone.molarity_mol_l {
        solvent.molarity_mol_l = molarity;
    }
    if let Some(solvent_per_lipid) = zone.solvent_per_lipid {
        solvent.solvent_per_lipid = Some(solvent_per_lipid);
    }
    if let Some(cutoff) = zone.solvent_per_lipid_cutoff {
        solvent.solvent_per_lipid_cutoff = cutoff;
    }
    if !zone.species.is_empty() {
        solvent.species = zone.species.clone();
    }
    solvent
}

fn emit_solvent_zone_and_ions(
    request: &BuildRequest,
    solvent: &SolventPolicy,
    ions: &IonPolicy,
    net_charge: Option<f32>,
    neutralization: &mut NeutralizationSummary,
    emitted_beads: &mut Vec<EmittedBead>,
    next_residue_id: &mut i32,
) -> Result<SolventEmission> {
    let solvent_species = resolved_solvent_species(solvent);
    let cation_species = resolved_ion_species(&ions.cations, &ions.cation, ions.cation_charge_e);
    let anion_species = resolved_ion_species(&ions.anions, &ions.anion, ions.anion_charge_e);
    let solvent_box_size = solvent_box_size_angstrom(&request.system, solvent);
    let box_volume_nm3 = box_volume_nm3(solvent_box_size);
    let excluded_volume_nm3 = excluded_volume_nm3(
        &beads_inside_solvent_zone(emitted_beads, request, solvent),
        solvent.excluded_bead_radius_angstrom,
    );
    let free_volume_nm3 = (box_volume_nm3 - excluded_volume_nm3).max(0.0);
    let solvent_counts_by_species = if let Some(solvent_per_lipid) = solvent.solvent_per_lipid {
        let lipid_count = lipid_count_inside_solvent_zone(emitted_beads, request, solvent);
        ratio_target_counts(
            solvent_per_lipid * lipid_count as f32,
            &solvent_species
                .iter()
                .map(|species| species.ratio)
                .collect::<Vec<_>>(),
        )
    } else {
        coarse_molarity_counts(
            free_volume_nm3,
            solvent.molarity_mol_l,
            &solvent_species
                .iter()
                .map(|species| (species.ratio, species.mapping_ratio))
                .collect::<Vec<_>>(),
        )
    };
    let solvent_count = solvent_counts_by_species.iter().sum::<usize>();
    let solvent_material_volume_nm3 =
        mixed_solvent_material_volume_nm3(&solvent_counts_by_species, &solvent_species);
    let mut cation_counts = coarse_molarity_counts(
        solvent_material_volume_nm3,
        ions.salt_molarity_mol_l,
        &cation_species
            .iter()
            .map(|species| (species.ratio, 1.0))
            .collect::<Vec<_>>(),
    )
    .into_iter()
    .map(|count| count as isize)
    .collect::<Vec<_>>();
    let mut anion_counts = coarse_molarity_counts(
        solvent_material_volume_nm3,
        ions.salt_molarity_mol_l,
        &anion_species
            .iter()
            .map(|species| (species.ratio, 1.0))
            .collect::<Vec<_>>(),
    )
    .into_iter()
    .map(|count| count as isize)
    .collect::<Vec<_>>();

    let solvent_charge = solvent_species
        .iter()
        .zip(solvent_counts_by_species.iter())
        .map(|(species, count)| species.charge_e * *count as f32)
        .sum::<f32>();
    let baseline_ion_charge = ion_charge_sum(&cation_species, &cation_counts)
        + ion_charge_sum(&anion_species, &anion_counts);
    let current_charge = net_charge.unwrap_or(0.0) + solvent_charge + baseline_ion_charge;
    if neutralization.enabled {
        let (cation_delta, anion_delta, counterion, counterion_charge) =
            solvent_neutralization_deltas(ions, &cation_species, &anion_species, current_charge);
        apply_delta_to_ions(&mut cation_counts, &cation_species, cation_delta);
        apply_delta_to_ions(&mut anion_counts, &anion_species, anion_delta);
        neutralization.cation_delta = cation_delta;
        neutralization.anion_delta = anion_delta;
        neutralization.counterion_count = cation_delta.unsigned_abs() + anion_delta.unsigned_abs();
        neutralization.counterion = counterion;
        neutralization.counterion_charge_e = counterion_charge;
        neutralization.residual_charge_e = Some(
            current_charge
                + cation_delta as f32 * representative_charge(&cation_species).unwrap_or(1) as f32
                + anion_delta as f32 * representative_charge(&anion_species).unwrap_or(-1) as f32,
        );
    }
    let cation_counts = cation_counts
        .into_iter()
        .map(|count| count.max(0) as usize)
        .collect::<Vec<_>>();
    let anion_counts = anion_counts
        .into_iter()
        .map(|count| count.max(0) as usize)
        .collect::<Vec<_>>();
    let cation_count = cation_counts.iter().sum::<usize>();
    let anion_count = anion_counts.iter().sum::<usize>();

    let total_inserted = solvent_count + cation_count + anion_count;
    let placement_plan = solvent_placement_plan(request, emitted_beads, solvent, total_inserted);
    let candidates = &placement_plan.candidates;
    if candidates.len() < total_inserted {
        return Err(anyhow!(
            "environment.solvent generated {} valid grid points but needs {total_inserted} molecules; decrease grid spacing or solvent count",
            candidates.len()
        ));
    }

    let mut cursor = 0usize;
    for (species, count) in solvent_species.iter().zip(solvent_counts_by_species.iter()) {
        emit_solvent_molecules(
            emitted_beads,
            next_residue_id,
            candidates,
            &mut cursor,
            species,
            *count,
        );
    }
    for (species, count) in cation_species.iter().zip(cation_counts.iter()) {
        emit_simple_molecules(
            emitted_beads,
            next_residue_id,
            candidates,
            &mut cursor,
            &species.residue_name,
            &species.atom_name,
            species.charge_e as f32,
            *count,
        );
    }
    for (species, count) in anion_species.iter().zip(anion_counts.iter()) {
        emit_simple_molecules(
            emitted_beads,
            next_residue_id,
            candidates,
            &mut cursor,
            &species.residue_name,
            &species.atom_name,
            species.charge_e as f32,
            *count,
        );
    }

    let mut counts = BTreeMap::new();
    for (species, count) in solvent_species.iter().zip(solvent_counts_by_species.iter()) {
        *counts.entry(species.name.clone()).or_insert(0) += *count;
    }
    for (species, count) in cation_species.iter().zip(cation_counts.iter()) {
        *counts.entry(species.residue_name.clone()).or_insert(0) += *count;
    }
    for (species, count) in anion_species.iter().zip(anion_counts.iter()) {
        *counts.entry(species.residue_name.clone()).or_insert(0) += *count;
    }

    Ok(SolventEmission {
        counts,
        solvent_charge_e: solvent_charge,
        baseline_ion_charge_e: baseline_ion_charge,
        neutralization_input_charge_e: current_charge,
        summary: SolventPlacementSummary {
            algorithm: solvent_algorithm_name(&request.system, false),
            mode: request.system.placement.mode.clone(),
            random_seed: request.system.placement.random_seed,
            box_volume_nm3,
            excluded_volume_nm3,
            free_volume_nm3,
            solvent_material_volume_nm3,
            grid_point_count: placement_plan.grid_point_count,
            inserted_count: total_inserted,
            grid_squeeze_pass_count: placement_plan.grid_squeeze_pass_count,
            squeezed_candidate_count: placement_plan.squeezed_candidate_count,
            min_grid_spacing_angstrom: placement_plan.min_grid_spacing_angstrom,
            kick_attempt_count: placement_plan.kick_attempt_count,
            kicked_inserted_count: placement_plan.kicked_inserted_count,
            density: phase_density_summary(
                total_inserted,
                total_inserted,
                placement_plan.grid_point_count,
                placement_plan.final_candidate_count,
                placement_plan.grid_squeeze_pass_count,
            ),
        },
    })
}

pub(super) fn solvent_box_size_angstrom(system: &BuildSystem, solvent: &SolventPolicy) -> [f32; 3] {
    solvent
        .box_size_angstrom
        .unwrap_or_else(|| placement_box_size_angstrom(system))
}

fn emit_simple_molecules(
    emitted_beads: &mut Vec<EmittedBead>,
    next_residue_id: &mut i32,
    candidates: &[[f32; 3]],
    cursor: &mut usize,
    residue_name: &str,
    atom_name: &str,
    charge_e: f32,
    count: usize,
) {
    for _ in 0..count {
        let position_angstrom = candidates[*cursor];
        *cursor += 1;
        emitted_beads.push(EmittedBead {
            residue_id: *next_residue_id,
            residue_name: residue_name.to_string(),
            atom_name: atom_name.to_string(),
            charge_e,
            position_angstrom,
            excluded_volume_factor: 0.0,
        });
        *next_residue_id += 1;
    }
}

fn emit_solvent_molecules(
    emitted_beads: &mut Vec<EmittedBead>,
    next_residue_id: &mut i32,
    candidates: &[[f32; 3]],
    cursor: &mut usize,
    species: &ResolvedSolventSpecies,
    count: usize,
) {
    for _ in 0..count {
        let center = candidates[*cursor];
        *cursor += 1;
        let residue_id = *next_residue_id;
        *next_residue_id += 1;
        for bead in &species.beads {
            emitted_beads.push(EmittedBead {
                residue_id,
                residue_name: species.name.clone(),
                atom_name: bead.atom_name.clone(),
                charge_e: bead.charge_e,
                position_angstrom: [
                    center[0] + bead.offset_angstrom[0],
                    center[1] + bead.offset_angstrom[1],
                    center[2] + bead.offset_angstrom[2],
                ],
                excluded_volume_factor: 0.0,
            });
        }
    }
}

fn beads_inside_solvent_zone(
    beads: &[EmittedBead],
    request: &BuildRequest,
    solvent: &SolventPolicy,
) -> Vec<EmittedBead> {
    beads
        .iter()
        .filter(|bead| point_inside_solvent_zone(bead.position_angstrom, request, solvent))
        .cloned()
        .collect()
}

fn lipid_count_inside_solvent_zone(
    beads: &[EmittedBead],
    request: &BuildRequest,
    solvent: &SolventPolicy,
) -> usize {
    let mut lipids: BTreeMap<i32, (usize, usize)> = BTreeMap::new();
    for bead in beads
        .iter()
        .filter(|bead| (bead.excluded_volume_factor - 1.0).abs() < 1.0e-5)
    {
        let entry = lipids.entry(bead.residue_id).or_insert((0, 0));
        entry.0 += 1;
        if point_inside_solvent_zone(bead.position_angstrom, request, solvent) {
            entry.1 += 1;
        }
    }
    lipids
        .values()
        .filter(|(total, inside)| {
            *total > 0 && (*inside as f32) > (*total as f32 * solvent.solvent_per_lipid_cutoff)
        })
        .count()
}

pub(super) fn point_inside_solvent_zone(
    position: [f32; 3],
    request: &BuildRequest,
    solvent: &SolventPolicy,
) -> bool {
    if solvent.box_size_angstrom.is_none() {
        if let Some(vectors) = distance_box_vectors(&request.system) {
            return point_inside_vector_cell(
                position,
                solvent.center_angstrom.unwrap_or([0.0, 0.0, 0.0]),
                vectors,
                0.0,
            );
        }
    }
    let [box_x, box_y, box_z] = solvent_box_size_angstrom(&request.system, solvent);
    let [center_x, center_y, center_z] = solvent.center_angstrom.unwrap_or([0.0, 0.0, 0.0]);
    let min_x = center_x - box_x * 0.5;
    let max_x = center_x + box_x * 0.5;
    let min_y = center_y - box_y * 0.5;
    let max_y = center_y + box_y * 0.5;
    let min_z = center_z - box_z * 0.5;
    let max_z = center_z + box_z * 0.5;
    let [x, y, z] = position;
    x >= min_x && x <= max_x && y >= min_y && y <= max_y && z >= min_z && z <= max_z
}

fn box_volume_nm3(box_size_angstrom: [f32; 3]) -> f32 {
    box_size_angstrom[0] * box_size_angstrom[1] * box_size_angstrom[2] * 1.0e-3
}

fn excluded_volume_nm3(beads: &[EmittedBead], radius_angstrom: f32) -> f32 {
    let radius_nm = radius_angstrom as f64 * 0.1;
    let bead_volume_nm3 = (4.0 / 3.0) * std::f64::consts::PI * radius_nm.powi(3);
    let factor_sum = beads
        .iter()
        .map(|bead| bead.excluded_volume_factor.max(0.0) as f64)
        .sum::<f64>();
    (factor_sum * bead_volume_nm3) as f32
}
