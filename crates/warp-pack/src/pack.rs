use std::path::Path;
use std::time::{Duration, Instant};

use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

use crate::atom_params::{build_atom_params, max_radius, scale_atom_params, AtomParams};
use crate::config::PackConfig;
use crate::error::{PackError, PackResult};
use crate::gencan::{evaluate_state, optimize_gencan, GencanResult};
use crate::gencan_objective::ObjectiveBuffer;
use crate::geom::{Quaternion, Vec3};
use crate::io::write_output;
use crate::movebad::{build_movebad_index, run_movebad_pass};
use crate::pack_ops::{
    append_bonds, append_ter_after, apply_chain, apply_resnumbers, build_order, load_template,
    maybe_write_snapshot, place_fixed, random_center_for_structure,
    satisfies_structure_constraints, transform_atoms, transform_positions_into,
    validate_min_distance, write_bad_snapshot, PlacementSeed, TemplateEntry,
};
use crate::pbc::PbcBox;
use crate::placement::PlacementRecord;
use crate::relax::relax_overlaps;
use crate::restart::{read_restart, write_restart, RestartEntry};
use crate::spatial_hash::{SpatialHash, SpatialHashParamsExt, SpatialHashV2};
use crate::streaming::{PackingPhase, StreamEmitter};

#[derive(Clone, Debug)]
pub struct AtomRecord {
    pub record_kind: AtomRecordKind,
    pub name: String,
    pub element: String,
    pub resname: String,
    pub resid: i32,
    pub chain: char,
    pub segid: String,
    pub charge: f32,
    pub position: Vec3,
    pub mol_id: i32,
}

#[derive(Clone, Debug)]
pub struct PackOutput {
    pub atoms: Vec<AtomRecord>,
    pub bonds: Vec<(usize, usize)>,
    pub box_size: [f32; 3],
    pub ter_after: Vec<usize>,
}

#[derive(Default)]
struct PackProfile {
    templates: Duration,
    place_core: Duration,
    movebad: Duration,
    gencan: Duration,
    relax: Duration,
}

impl PackProfile {
    fn enabled() -> bool {
        std::env::var("WARP_PACK_PROFILE").is_ok()
    }

    fn report(&self) {
        let secs = |d: Duration| d.as_secs_f64();
        let total = self.templates + self.place_core + self.movebad + self.gencan + self.relax;
        eprintln!(
            "warp-pack profile (s): total={:.3} templates={:.3} place_core={:.3} movebad={:.3} gencan={:.3} relax={:.3}",
            secs(total),
            secs(self.templates),
            secs(self.place_core),
            secs(self.movebad),
            secs(self.gencan),
            secs(self.relax),
        );
    }
}

fn rebuild_hash_from_positions(hash: &mut SpatialHashV2, positions: &[Vec3]) {
    hash.clear();
    for (idx, pos) in positions.iter().copied().enumerate() {
        hash.insert(idx, pos);
    }
}

fn maybe_write_gencan_snapshot(
    cfg: &PackConfig,
    atoms: &[AtomRecord],
    bonds: &[(usize, usize)],
    ter_after: &[usize],
    box_size: [f32; 3],
    last_write: &mut Instant,
    current_fx: f32,
    best_fx: &mut f32,
) -> PackResult<()> {
    let Some(interval) = cfg.writeout else {
        return Ok(());
    };
    if interval <= 0.0 {
        return Ok(());
    }
    if last_write.elapsed().as_secs_f32() < interval {
        return Ok(());
    }
    let Some(spec) = cfg.output.as_ref() else {
        return Ok(());
    };
    let improved = current_fx < *best_fx;
    if improved || cfg.writebad {
        let out = PackOutput {
            atoms: atoms.to_vec(),
            bonds: bonds.to_vec(),
            box_size,
            ter_after: ter_after.to_vec(),
        };
        let add_box_sides = cfg.add_box_sides || cfg.pbc;
        let box_fix = cfg.add_box_sides_fix.unwrap_or(0.0);
        let write_conect = !cfg.ignore_conect;
        write_output(
            &out,
            spec,
            add_box_sides,
            box_fix,
            write_conect,
            cfg.hexadecimal_indices,
        )?;
        if improved {
            *best_fx = current_fx;
        }
    }
    *last_write = Instant::now();
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomRecordKind {
    Atom,
    HetAtom,
}

pub fn run(config: &PackConfig) -> PackResult<PackOutput> {
    run_with_stream(config, StreamEmitter::disabled())
}

pub fn run_with_stream(config: &PackConfig, emitter: StreamEmitter) -> PackResult<PackOutput> {
    let cfg = config.normalized()?;
    let seed = cfg.seed.unwrap_or(1_234_567);
    let out = run_once(&cfg, seed, 0, emitter)?;
    if cfg.check {
        validate_min_distance(&out.atoms, cfg.min_distance.unwrap_or(2.0))?;
    }
    Ok(out)
}

fn run_once(
    cfg: &PackConfig,
    seed: u64,
    _attempt: usize,
    emitter: StreamEmitter,
) -> PackResult<PackOutput> {
    let mut profile = if PackProfile::enabled() {
        Some(PackProfile::default())
    } else {
        None
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let box_size = cfg.box_.size;
    let box_origin = Vec3::from_array(cfg.pbc_min.unwrap_or([0.0, 0.0, 0.0]));
    let pbc = if cfg.pbc {
        if let (Some(min), Some(max)) = (cfg.pbc_min, cfg.pbc_max) {
            Some(PbcBox::from_bounds(min, max)?)
        } else {
            Some(PbcBox::from_size(box_size)?)
        }
    } else {
        None
    };
    let dist_scale = 1.0f32;
    let total_mols: usize = cfg.structures.iter().map(|s| s.count).sum();

    // Emit pack started event
    if emitter.is_enabled() {
        emitter.emit_pack_started(&crate::streaming::PackStartedEvent {
            total_molecules: total_mols,
            box_size,
            box_origin: [box_origin.x, box_origin.y, box_origin.z],
            output_path: cfg.output.as_ref().map(|spec| spec.path.clone()),
        });
    }

    let restart_all = if let Some(path) = &cfg.restart_from {
        Some(read_restart(Path::new(path))?)
    } else {
        None
    };
    let mut restart_per_spec = Vec::with_capacity(cfg.structures.len());
    for spec in &cfg.structures {
        if let Some(path) = &spec.restart_from {
            restart_per_spec.push(Some(read_restart(Path::new(path))?));
        } else {
            restart_per_spec.push(None);
        }
    }
    if let Some(entries) = &restart_all {
        if entries.len() != total_mols {
            return Err(PackError::Invalid(
                "restart_from entry count does not match total molecule count".into(),
            ));
        }
    }
    for (idx, spec) in cfg.structures.iter().enumerate() {
        if let Some(entries) = &restart_per_spec[idx] {
            if entries.len() != spec.count {
                return Err(PackError::Invalid(format!(
                    "restart_from entry count does not match count for structure {}",
                    spec.path
                )));
            }
        }
    }
    let mut restart_offsets = vec![0usize; cfg.structures.len()];
    if restart_all.is_some() {
        let mut offset = 0usize;
        for (idx, spec) in cfg.structures.iter().enumerate() {
            restart_offsets[idx] = offset;
            offset += spec.count;
        }
    }
    let restart_entry = |spec_index: usize, local_index: usize| -> Option<RestartEntry> {
        if let Some(entries) = &restart_per_spec[spec_index] {
            return entries.get(local_index).copied();
        }
        if let Some(entries) = &restart_all {
            let offset = restart_offsets[spec_index];
            return entries.get(offset + local_index).copied();
        }
        None
    };
    let mut global_resid_counter = 1i32;
    let mut atoms = Vec::new();
    let mut bonds = Vec::new();
    let mut ter_after = Vec::new();
    let mut positions = Vec::new();
    let mut atom_params: Vec<AtomParams> = Vec::new();
    let fbins = cfg.fbins.unwrap_or(3.0_f32.sqrt());
    let mut mol_id = 1i32;
    let mut last_write = Instant::now();
    let mut mol_spec: Vec<usize> = Vec::with_capacity(total_mols);
    let mut placements = vec![PlacementRecord::default(); total_mols];
    let mut placement_filled = vec![false; total_mols];
    let short_tol_dist = cfg
        .short_tol_dist
        .unwrap_or(cfg.min_distance.unwrap_or(2.0) * 0.5)
        * dist_scale;
    let short_tol_scale = cfg.short_tol_scale.unwrap_or(3.0);
    let mut use_short_tol = cfg.use_short_tol;

    let mut templates = Vec::with_capacity(cfg.structures.len());
    let mut max_atom_radius = 0.0f32;
    let t_templates = Instant::now();

    if emitter.is_enabled() {
        emitter.emit_phase_started(&crate::streaming::PhaseStartedEvent {
            phase: PackingPhase::TemplateLoad,
            total_molecules: Some(total_mols),
            max_iterations: None,
        });
    }

    for (idx, spec) in cfg.structures.iter().enumerate() {
        let format = spec.format.as_deref().or(cfg.filetype.as_deref());
        let template = load_template(
            spec,
            mol_id + idx as i32,
            format,
            cfg.ignore_conect,
            cfg.non_standard_conect,
        )?;
        for atom_constraint in &spec.atom_constraints {
            for &idx1 in &atom_constraint.indices {
                if idx1 == 0 || idx1 > template.atoms.len() {
                    return Err(PackError::Invalid(format!(
                        "atom constraint index {} out of range for structure {} ({} atoms)",
                        idx1,
                        spec.path,
                        template.atoms.len()
                    )));
                }
            }
        }
        let atom_params = build_atom_params(cfg, spec, template.atoms.len(), dist_scale)?;
        max_atom_radius = max_atom_radius.max(max_radius(&atom_params));
        let res_mode = spec.resnumbers.unwrap_or(1);
        let connect = spec.connect && !cfg.ignore_conect;
        let maxmove = spec.maxmove.or(cfg.maxmove).unwrap_or(spec.count.max(1));
        templates.push(TemplateEntry {
            template,
            res_mode,
            connect,
            maxmove,
            atom_params,
        });
    }
    if let Some(ref mut prof) = profile {
        prof.templates += t_templates.elapsed();
    }
    if emitter.is_enabled() {
        emitter.emit_phase_complete(&crate::streaming::PhaseCompleteEvent {
            phase: PackingPhase::TemplateLoad,
            elapsed_ms: crate::streaming::duration_ms(t_templates.elapsed()),
            iterations: None,
            final_obj_value: None,
        });
    }
    if !use_short_tol {
        use_short_tol = templates
            .iter()
            .any(|entry| entry.atom_params.iter().any(|p| p.use_short));
    }
    let short_tol = if use_short_tol {
        Some((short_tol_dist, short_tol_scale))
    } else {
        None
    };
    let cell_size = ((2.0 * max_atom_radius) * fbins * dist_scale).max(1.0e-6);
    let box_max = box_origin.add(Vec3::from_array(box_size));
    let mut hash = SpatialHashV2::new(cell_size, box_origin, box_max);

    for (idx, spec) in cfg.structures.iter().enumerate() {
        if !spec.fixed {
            continue;
        }
        let entry = &templates[idx];
        let mut seeds = Vec::with_capacity(spec.count);
        let mut seed_eulers = Vec::with_capacity(spec.count);
        for i in 0..spec.count {
            let (center, rotation, euler) = if let Some(positions) = &spec.positions {
                let pos = positions.get(i).copied().unwrap_or([0.0, 0.0, 0.0]);
                let euler = spec
                    .fixed_eulers
                    .as_ref()
                    .and_then(|vals| vals.get(i).copied())
                    .unwrap_or([0.0, 0.0, 0.0]);
                let rotation = Quaternion::from_packmol_euler(euler[0], euler[1], euler[2]);
                (Vec3::from_array(pos), rotation, euler)
            } else if let Some(entry) = restart_entry(idx, i) {
                let rotation =
                    Quaternion::from_packmol_euler(entry.euler[0], entry.euler[1], entry.euler[2]);
                (entry.center, rotation, entry.euler)
            } else {
                return Err(PackError::Invalid(
                    "fixed structure missing positions and restart_from".into(),
                ));
            };
            seeds.push(PlacementSeed { center, rotation });
            seed_eulers.push(euler);
        }
        place_fixed(
            spec,
            &entry.template,
            &entry.atom_params,
            &seeds,
            &mut atoms,
            &mut positions,
            &mut atom_params,
            &mut hash,
            mol_id,
            pbc,
            entry.res_mode,
            &mut global_resid_counter,
            &mut bonds,
            &mut ter_after,
            entry.connect,
            cfg.add_amber_ter,
            cfg.amber_ter_preserve,
            cfg.avoid_overlap,
        )?;
        let mol_id_start = mol_id;
        for i in 0..spec.count {
            let mol_index = (mol_id_start + i as i32 - 1) as usize;
            placements[mol_index] = PlacementRecord::new(seeds[i].center, seed_eulers[i]);
            placement_filled[mol_index] = true;
            mol_spec.push(idx);
        }
        mol_id += spec.count as i32;
        maybe_write_snapshot(&cfg, &atoms, &bonds, &ter_after, box_size, &mut last_write)?;
    }

    let mut order = build_order(cfg, &templates);
    if cfg.packall {
        order.shuffle(&mut rng);
    }

    if cfg.randominitialpoint {
        order.shuffle(&mut rng);
    }

    if emitter.is_enabled() {
        emitter.emit_phase_started(&crate::streaming::PhaseStartedEvent {
            phase: PackingPhase::CorePlacement,
            total_molecules: Some(total_mols),
            max_iterations: None,
        });
    }
    let t_placement = Instant::now();
    let mut molecules_placed = 0usize;

    struct Candidate {
        penalty: f32,
        center: Vec3,
        euler: [f32; 3],
    }

    for placement in order {
        let t_place = Instant::now();
        let spec = &cfg.structures[placement.spec_index];
        let entry = &templates[placement.spec_index];
        let mut trial_positions = Vec::with_capacity(entry.template.atoms.len());
        // Packmol-style flow allows non-zero overlap in initial placement and relies on
        // global optimization to resolve residual clashes.
        let mut best: Option<Candidate> = None;
        if let Some(restart) = restart_entry(
            placement.spec_index,
            (placement.local_index.max(1) - 1) as usize,
        ) {
            let rotation = Quaternion::from_packmol_euler(
                restart.euler[0],
                restart.euler[1],
                restart.euler[2],
            );
            transform_positions_into(
                &entry.template,
                rotation,
                restart.center,
                &mut trial_positions,
            );
            if satisfies_structure_constraints(&trial_positions, spec, pbc) {
                if short_tol.is_some() {
                    if let Some(penalty) = hash.overlaps_short_tol_params(
                        &trial_positions,
                        &entry.atom_params,
                        &positions,
                        &atom_params,
                    ) {
                        if best
                            .as_ref()
                            .map(|best| penalty < best.penalty)
                            .unwrap_or(true)
                        {
                            best = Some(Candidate {
                                penalty,
                                center: restart.center,
                                euler: restart.euler,
                            });
                        }
                    }
                } else {
                    let penalty = hash.overlap_penalty_params(
                        &trial_positions,
                        &entry.atom_params,
                        &positions,
                        &atom_params,
                    );
                    best = Some(Candidate {
                        penalty,
                        center: restart.center,
                        euler: restart.euler,
                    });
                }
            }
        }
        if !best
            .as_ref()
            .map(|cand| cand.penalty == 0.0)
            .unwrap_or(false)
        {
            let max_attempts = cfg.max_attempts.unwrap_or(10000);
            let placed = (mol_id as usize).saturating_sub(1);
            let total = placements.len().max(1);
            let progress = (placed as f32 / total as f32).clamp(0.0, 1.0);
            let dense_factor = 1.0 - progress;
            let scaled = 32.0 + dense_factor * dense_factor * 224.0;
            let fallback_budget = max_attempts.min(scaled.round() as usize).max(1);
            for attempt in 0..max_attempts {
                let (rotation, euler) = if spec.rotate {
                    if let Some(bounds) = spec.rot_bounds {
                        let bx = rng.gen_range(bounds[0][0]..=bounds[0][1]);
                        let gy = rng.gen_range(bounds[1][0]..=bounds[1][1]);
                        let tz = rng.gen_range(bounds[2][0]..=bounds[2][1]);
                        (Quaternion::from_packmol_euler(bx, gy, tz), [bx, gy, tz])
                    } else {
                        let rot = Quaternion::random(&mut rng);
                        let (b, g, t) = rot.to_packmol_euler();
                        (rot, [b, g, t])
                    }
                } else {
                    (Quaternion::identity(), [0.0, 0.0, 0.0])
                };
                let center = random_center_for_structure(
                    &mut rng,
                    spec,
                    box_size,
                    box_origin,
                    entry.template.radius,
                )?;
                transform_positions_into(&entry.template, rotation, center, &mut trial_positions);
                if !satisfies_structure_constraints(&trial_positions, spec, pbc) {
                    continue;
                }
                if short_tol.is_some() {
                    if let Some(penalty) = hash.overlaps_short_tol_params(
                        &trial_positions,
                        &entry.atom_params,
                        &positions,
                        &atom_params,
                    ) {
                        if best
                            .as_ref()
                            .map(|best| penalty < best.penalty)
                            .unwrap_or(true)
                        {
                            best = Some(Candidate {
                                penalty,
                                center,
                                euler,
                            });
                        }
                        if penalty == 0.0 {
                            break;
                        }
                    }
                } else {
                    let penalty = hash.overlap_penalty_params(
                        &trial_positions,
                        &entry.atom_params,
                        &positions,
                        &atom_params,
                    );
                    if best
                        .as_ref()
                        .map(|best| penalty < best.penalty)
                        .unwrap_or(true)
                    {
                        best = Some(Candidate {
                            penalty,
                            center,
                            euler,
                        });
                    }
                    if penalty == 0.0 {
                        break;
                    }
                    if best.is_some() && attempt + 1 >= fallback_budget {
                        break;
                    }
                }
            }
        }
        let Some(best) = best else {
            if cfg.writebad {
                write_bad_snapshot(cfg, &atoms, &bonds, &ter_after, box_size)?;
            }
            return Err(PackError::Placement(format!(
                "failed to place structure {}",
                spec.path
            )));
        };
        let rotation = Quaternion::from_packmol_euler(best.euler[0], best.euler[1], best.euler[2]);
        let mut candidate = transform_atoms(&entry.template, rotation, best.center, mol_id);
        apply_chain(&mut candidate, spec, mol_id);
        apply_resnumbers(
            &mut candidate,
            entry.res_mode,
            entry.template.resid_min,
            global_resid_counter,
            placement.local_index,
            mol_id,
        )?;
        global_resid_counter += entry.template.resid_count;
        let base = atoms.len();
        if entry.connect {
            append_bonds(&mut bonds, &entry.template.bonds, base);
        }
        if cfg.amber_ter_preserve {
            append_ter_after(&mut ter_after, &entry.template.ter_after, base);
        }
        for (offset, atom) in candidate.into_iter().enumerate() {
            let idx = positions.len();
            positions.push(atom.position);
            hash.insert(idx, atom.position);
            atom_params.push(entry.atom_params[offset]);
            atoms.push(atom);
        }
        if cfg.add_amber_ter {
            if let Some(last) = atoms.len().checked_sub(1) {
                ter_after.push(last);
            }
        }
        let mol_index = (mol_id - 1) as usize;
        placements[mol_index] = PlacementRecord::new(best.center, best.euler);
        placement_filled[mol_index] = true;
        mol_spec.push(placement.spec_index);

        molecules_placed += 1;
        if emitter.is_enabled() {
            emitter.emit_molecule_placed(&crate::streaming::MoleculePlacedEvent {
                molecule_index: molecules_placed,
                total_molecules: total_mols,
                molecule_name: spec.path.clone(),
                successful: true,
            });
        }

        if best.penalty > 0.0 {
            // Avoid expensive per-molecule GENCAN during initial placement; a
            // constrained overlap fallback should remain cheap and rely on the
            // normal global optimization/movebad passes to resolve residuals.
        }
        mol_id += 1;
        maybe_write_snapshot(cfg, &atoms, &bonds, &ter_after, box_size, &mut last_write)?;
        if let Some(ref mut prof) = profile {
            prof.place_core += t_place.elapsed();
        }

        // Packmol does not run movebad after each individual molecule placement.
        // Keep movebad in the later optimization loops only.
    }

    if emitter.is_enabled() {
        emitter.emit_phase_complete(&crate::streaming::PhaseCompleteEvent {
            phase: PackingPhase::CorePlacement,
            elapsed_ms: crate::streaming::duration_ms(t_placement.elapsed()),
            iterations: None,
            final_obj_value: None,
        });
    }

    if placement_filled.iter().any(|filled| !filled) {
        return Err(PackError::Placement(
            "internal error: missing placement records".into(),
        ));
    }
    if mol_spec.len() != placements.len() {
        return Err(PackError::Placement(
            "internal error: molecule mapping mismatch".into(),
        ));
    }

    let movebad_index = build_movebad_index(&atoms, &mol_spec, templates.len());
    let precision = cfg.precision.unwrap_or(1.0e-2);
    let use_short_tol = cfg.use_short_tol;
    let trace_outer = std::env::var("WARP_PACK_TRACE_OUTER").is_ok();
    let mut eval_objective_buffer = ObjectiveBuffer::default();
    let initial_stats = evaluate_state(
        cfg,
        &atoms,
        &positions,
        &atom_params,
        &mol_spec,
        &templates,
        pbc,
        use_short_tol,
        true,
        None,
        None,
        &mut eval_objective_buffer,
    )?;
    let initial_solution =
        initial_stats.max_overlap <= precision && initial_stats.max_constraint <= precision;
    if initial_solution && trace_outer {
        eprintln!("initial approximation is a solution. nothing to do.");
    }

    if !initial_solution {
        let t_gencan = Instant::now();
        if emitter.is_enabled() {
            emitter.emit_phase_started(&crate::streaming::PhaseStartedEvent {
                phase: PackingPhase::GencanOptimization,
                total_molecules: Some(total_mols),
                max_iterations: cfg.maxit,
            });
        }
        let nloop0_default = cfg.nloop0.unwrap_or(0);
        let nloop_default = cfg.nloop.unwrap_or(1);
        let snapshot_enabled = cfg.writeout.is_some() || cfg.writebad;
        let radscale_start = cfg.discale.unwrap_or(1.1).max(1.0);
        let mut gencan_last_write = Instant::now();
        let mut fprint = if snapshot_enabled {
            initial_stats.value
        } else {
            f32::INFINITY
        };

        // Initial structure-restricted passes used by Packmol to satisfy constraints.
        for (spec_index, spec) in cfg.structures.iter().enumerate() {
            let loops0 = spec.nloop0.unwrap_or(nloop0_default);
            if loops0 == 0 {
                continue;
            }
            let mut radscale = radscale_start;
            eval_objective_buffer = ObjectiveBuffer::default();
            let mut base_stats = evaluate_state(
                cfg,
                &atoms,
                &positions,
                &atom_params,
                &mol_spec,
                &templates,
                pbc,
                use_short_tol,
                true,
                Some(spec_index),
                None,
                &mut eval_objective_buffer,
            )?;
            if base_stats.max_constraint <= precision {
                continue;
            }
            for _ in 0..loops0 {
                let scaled = scale_atom_params(&atom_params, radscale);
                let res = optimize_gencan(
                    cfg,
                    &mut atoms,
                    &mut positions,
                    &scaled,
                    &mut placements,
                    &mol_spec,
                    &templates,
                    box_size,
                    box_origin.to_array(),
                    pbc,
                    false,
                    Some(spec_index),
                    None,
                    Some(emitter),
                )?;
                base_stats = GencanResult {
                    value: res.value,
                    max_overlap: base_stats.max_overlap,
                    max_constraint: res.max_constraint,
                };
                if !cfg.disable_movebad && base_stats.max_constraint > precision {
                    rebuild_hash_from_positions(&mut hash, &positions);
                    let t_movebad = Instant::now();
                    run_movebad_pass(
                        cfg,
                        &templates,
                        &mut atoms,
                        &mut positions,
                        &mut atom_params,
                        &mut hash,
                        &mut placements,
                        box_size,
                        box_origin,
                        pbc,
                        &mut rng,
                        &mol_spec,
                        &movebad_index,
                        Some(spec_index),
                        true,
                        cell_size,
                    )?;
                    if let Some(ref mut prof) = profile {
                        prof.movebad += t_movebad.elapsed();
                    }
                    eval_objective_buffer = ObjectiveBuffer::default();
                    base_stats = evaluate_state(
                        cfg,
                        &atoms,
                        &positions,
                        &atom_params,
                        &mol_spec,
                        &templates,
                        pbc,
                        use_short_tol,
                        true,
                        Some(spec_index),
                        None,
                        &mut eval_objective_buffer,
                    )?;
                }
                if base_stats.max_constraint <= precision {
                    break;
                }
                if radscale > 1.0 {
                    radscale = (0.9 * radscale).max(1.0);
                }
            }
        }

        let mut stages: Vec<Option<usize>> = if cfg.packall {
            vec![None]
        } else {
            let mut stage_list = (0..cfg.structures.len()).map(Some).collect::<Vec<_>>();
            stage_list.push(None);
            stage_list
        };
        'stage_loop: for active_spec in stages.drain(..) {
            let stage_nloop = active_spec
                .and_then(|idx| cfg.structures.get(idx).and_then(|s| s.nloop))
                .unwrap_or(nloop_default);
            if stage_nloop == 0 {
                continue;
            }

            let mut radscale = radscale_start;
            eval_objective_buffer = ObjectiveBuffer::default();
            let mut stats = evaluate_state(
                cfg,
                &atoms,
                &positions,
                &atom_params,
                &mol_spec,
                &templates,
                pbc,
                use_short_tol,
                true,
                active_spec,
                None,
                &mut eval_objective_buffer,
            )?;
            let mut bestf = stats.value;
            let mut flast = stats.value;
            let mut fimp = 99.99f32;

            if stats.max_overlap <= precision && stats.max_constraint <= precision {
                if active_spec.is_none() {
                    break 'stage_loop;
                }
                continue;
            }

            for loop_idx in 0..stage_nloop {
                if !cfg.disable_movebad && radscale <= 1.0 + 1.0e-6 && fimp <= 10.0 {
                    rebuild_hash_from_positions(&mut hash, &positions);
                    let t_movebad = Instant::now();
                    if emitter.is_enabled() {
                        emitter.emit_phase_started(&crate::streaming::PhaseStartedEvent {
                            phase: PackingPhase::MoveBad,
                            total_molecules: Some(total_mols),
                            max_iterations: None,
                        });
                    }
                    run_movebad_pass(
                        cfg,
                        &templates,
                        &mut atoms,
                        &mut positions,
                        &mut atom_params,
                        &mut hash,
                        &mut placements,
                        box_size,
                        box_origin,
                        pbc,
                        &mut rng,
                        &mol_spec,
                        &movebad_index,
                        active_spec,
                        true,
                        cell_size,
                    )?;
                    if let Some(ref mut prof) = profile {
                        prof.movebad += t_movebad.elapsed();
                    }
                    if emitter.is_enabled() {
                        emitter.emit_phase_complete(&crate::streaming::PhaseCompleteEvent {
                            phase: PackingPhase::MoveBad,
                            elapsed_ms: crate::streaming::duration_ms(t_movebad.elapsed()),
                            iterations: None,
                            final_obj_value: None,
                        });
                    }
                    eval_objective_buffer = ObjectiveBuffer::default();
                    stats = evaluate_state(
                        cfg,
                        &atoms,
                        &positions,
                        &atom_params,
                        &mol_spec,
                        &templates,
                        pbc,
                        use_short_tol,
                        true,
                        active_spec,
                        None,
                        &mut eval_objective_buffer,
                    )?;
                    flast = stats.value;
                }

                let scaled = scale_atom_params(&atom_params, radscale);
                optimize_gencan(
                    cfg,
                    &mut atoms,
                    &mut positions,
                    &scaled,
                    &mut placements,
                    &mol_spec,
                    &templates,
                    box_size,
                    box_origin.to_array(),
                    pbc,
                    true,
                    active_spec,
                    None,
                    Some(emitter),
                )?;

                // Packmol computes loop statistics and convergence with user radii (unscaled).
                eval_objective_buffer = ObjectiveBuffer::default();
                stats = evaluate_state(
                    cfg,
                    &atoms,
                    &positions,
                    &atom_params,
                    &mol_spec,
                    &templates,
                    pbc,
                    use_short_tol,
                    true,
                    active_spec,
                    None,
                    &mut eval_objective_buffer,
                )?;
                if flast > 0.0 {
                    fimp = (-100.0 * (stats.value - flast) / flast)
                        .max(-99.99)
                        .min(99.99);
                } else {
                    fimp = 100.0;
                }
                if stats.value < bestf {
                    bestf = stats.value;
                }
                flast = stats.value;

                if trace_outer {
                    if let Some(spec_index) = active_spec {
                        eprintln!(
                            "outer spec={} loop={} f={:.6} overlap={:.6} constraint={:.6} fimp={:.3} radscale={:.3}",
                            spec_index,
                            loop_idx,
                            stats.value,
                            stats.max_overlap,
                            stats.max_constraint,
                            fimp,
                            radscale
                        );
                    } else {
                        eprintln!(
                            "outer spec=all loop={} f={:.6} overlap={:.6} constraint={:.6} fimp={:.3} radscale={:.3}",
                            loop_idx,
                            stats.value,
                            stats.max_overlap,
                            stats.max_constraint,
                            fimp,
                            radscale
                        );
                    }
                }

                if snapshot_enabled {
                    let snapshot_value = if active_spec.is_none() {
                        stats.value
                    } else {
                        eval_objective_buffer = ObjectiveBuffer::default();
                        evaluate_state(
                            cfg,
                            &atoms,
                            &positions,
                            &atom_params,
                            &mol_spec,
                            &templates,
                            pbc,
                            use_short_tol,
                            true,
                            None,
                            None,
                            &mut eval_objective_buffer,
                        )?
                        .value
                    };
                    maybe_write_gencan_snapshot(
                        cfg,
                        &atoms,
                        &bonds,
                        &ter_after,
                        box_size,
                        &mut gencan_last_write,
                        snapshot_value,
                        &mut fprint,
                    )?;
                }

                if stats.max_overlap <= precision && stats.max_constraint <= precision {
                    if active_spec.is_none() {
                        break 'stage_loop;
                    }
                    break;
                }

                if radscale > 1.0 {
                    if (stats.max_overlap <= precision && fimp < 10.0) || fimp < 2.0 {
                        radscale = (0.9 * radscale).max(1.0);
                    }
                }
            }
        }

        if let Some(ref mut prof) = profile {
            prof.gencan += t_gencan.elapsed();
        }
        if emitter.is_enabled() {
            emitter.emit_phase_complete(&crate::streaming::PhaseCompleteEvent {
                phase: PackingPhase::GencanOptimization,
                elapsed_ms: crate::streaming::duration_ms(t_gencan.elapsed()),
                iterations: None,
                final_obj_value: None,
            });
        }
    }

    if cfg.relax_steps.unwrap_or(0) > 0 {
        let t_relax = Instant::now();
        if emitter.is_enabled() {
            emitter.emit_phase_started(&crate::streaming::PhaseStartedEvent {
                phase: PackingPhase::Relax,
                total_molecules: Some(total_mols),
                max_iterations: Some(cfg.relax_steps.unwrap_or(0)),
            });
        }
        relax_overlaps(
            cfg,
            &mut atoms,
            &mut positions,
            &mut atom_params,
            &mut placements,
            &mol_spec,
            &templates,
            box_size,
            pbc,
            dist_scale,
        )?;
        if let Some(ref mut prof) = profile {
            prof.relax += t_relax.elapsed();
        }
        if emitter.is_enabled() {
            emitter.emit_phase_complete(&crate::streaming::PhaseCompleteEvent {
                phase: PackingPhase::Relax,
                elapsed_ms: crate::streaming::duration_ms(t_relax.elapsed()),
                iterations: Some(cfg.relax_steps.unwrap_or(0)),
                final_obj_value: None,
            });
        }
    }

    if cfg.restart_to.is_some() || cfg.structures.iter().any(|spec| spec.restart_to.is_some()) {
        write_restart_outputs(cfg, &placements, &mol_spec)?;
    }

    ter_after.sort_unstable();
    ter_after.dedup();

    if let Some(prof) = profile.as_ref() {
        prof.report();
    }

    Ok(PackOutput {
        atoms,
        bonds,
        box_size,
        ter_after,
    })
}

fn write_restart_outputs(
    cfg: &PackConfig,
    placements: &[PlacementRecord],
    mol_spec: &[usize],
) -> PackResult<()> {
    if let Some(path) = &cfg.restart_to {
        let entries: Vec<RestartEntry> = placements
            .iter()
            .map(|p| RestartEntry {
                center: p.center,
                euler: p.euler,
            })
            .collect();
        write_restart(Path::new(path), &entries)?;
    }
    for (spec_index, spec) in cfg.structures.iter().enumerate() {
        let Some(path) = &spec.restart_to else {
            continue;
        };
        let mut entries = Vec::new();
        for (mol_idx, &spec_idx) in mol_spec.iter().enumerate() {
            if spec_idx == spec_index {
                let placement = placements.get(mol_idx).copied().unwrap_or_default();
                entries.push(RestartEntry {
                    center: placement.center,
                    euler: placement.euler,
                });
            }
        }
        write_restart(Path::new(path), &entries)?;
    }
    Ok(())
}
