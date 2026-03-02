use std::collections::VecDeque;
use std::time::Instant;

use crate::atom_params::AtomParams;
use crate::config::PackConfig;
use crate::error::{PackError, PackResult};
use crate::gencan_objective::{
    build_bounds, compute_gradient_from_grad_pos, compute_gradient_from_grad_pos_into,
    compute_objective_only, compute_objective_with_buffer, grad_norm, pack_variables, project_step,
    update_positions, validate_molecule_counts, Bounds, MolInfo, ObjectiveBuffer,
};
use crate::geom::Vec3;
use crate::pack::AtomRecord;
use crate::pack_ops::TemplateEntry;
use crate::pbc::PbcBox;
use crate::placement::PlacementRecord;
use crate::streaming::StreamEmitter;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GencanResult {
    pub(crate) value: f32,
    pub(crate) max_overlap: f32,
    pub(crate) max_constraint: f32,
}

struct ProjectedStats {
    pg: Vec<f32>,
    gpsupn: f32,
    gpeucn2: f32,
    gieucn2: f32,
    nind: usize,
}

#[derive(Default)]
struct GencanCounters {
    obj_calls: usize,
    obj_only_calls: usize,
    eval_grad_calls: usize,
    hess_vec_calls: usize,
    cg_iters: usize,
    ls_trials: usize,
}

pub(crate) fn optimize_gencan(
    cfg: &PackConfig,
    atoms: &mut Vec<AtomRecord>,
    positions: &mut Vec<Vec3>,
    atom_params: &[AtomParams],
    placements: &mut [PlacementRecord],
    mol_spec: &[usize],
    templates: &[TemplateEntry],
    box_size: [f32; 3],
    box_origin: [f32; 3],
    pbc: Option<PbcBox>,
    include_overlap: bool,
    active_spec: Option<usize>,
    active_mol: Option<usize>,
    emitter: Option<StreamEmitter>,
) -> PackResult<GencanResult> {
    let total_mols = placements.len();
    if total_mols == 0 {
        return Ok(GencanResult::default());
    }
    let mol_info = build_mol_info(cfg, atoms, mol_spec, templates, active_spec, active_mol)?;

    let mut x = pack_variables(placements, &mol_info);
    let bounds = build_bounds(&mol_info, placements, box_size, box_origin);
    if std::env::var("WARP_PACK_TRACE_CALLS").is_ok() {
        eprintln!(
            "gencan call active_spec={:?} active_mol={:?} include_overlap={} nvars={}",
            active_spec,
            active_mol,
            include_overlap,
            x.len()
        );
    }
    let maxit = cfg.gencan_maxit.unwrap_or(20);
    let trace_stats = std::env::var("WARP_PACK_TRACE_GENCAN_STATS").is_ok();
    let gencan_step = cfg.gencan_step;
    let precision = cfg.precision.unwrap_or(1.0e-2);
    let use_short_tol = cfg.use_short_tol;
    let short_dist = cfg.short_tol_dist.unwrap_or(0.0);
    let short_scale = cfg.short_tol_scale.unwrap_or(1.0);
    let fbins = cfg.fbins.unwrap_or(3.0f32.sqrt());
    let radmax = atom_params.iter().fold(0.0f32, |acc, p| acc.max(p.radius));

    let nonmon_m = 10usize;
    let gamma = 1.0e-4f32;
    let beta = 0.5f32;
    let theta = 1.0e-6f32;
    let sigma1 = 0.1f32;
    let sigma2 = 0.9f32;
    let nint = 2.0f32;
    let next = 2.0f32;
    let min_interp = 4usize;
    let max_ls = 40usize;
    let epsgpsn = 1.0e-6f32;
    let lspg_min = 1.0e-10f32;
    let lspg_max = 1.0e10f32;
    // EASYGENCAN internal default used by Packmol.
    let delmin = 1.0e-2f32;
    let eta = 0.9f32;
    let ometa2 = (1.0 - eta) * (1.0 - eta);
    let cgepsi = 1.0e-1f32;
    let cgepsf = 1.0e-5f32;
    let cggpnf = 1.0e-4f32.max(epsgpsn);

    let mut f_hist = VecDeque::with_capacity(nonmon_m);
    let mut prev_x: Option<Vec<f32>> = None;
    let mut prev_g: Option<Vec<f32>> = None;
    let mut gpsupn0: Option<f32> = None;
    let mut acgeps = 0.0f32;
    let mut bcgeps = cgepsf.log10();
    let mut counters = GencanCounters::default();
    let mut iters_used = 0usize;
    let mut objective_buffer = ObjectiveBuffer::default();
    let overlap_atoms = collect_selected_overlap_atoms(atoms, &mol_info);

    if cfg.chkgrad {
        update_positions(&x, placements, &mol_info, atoms, positions);
        counters.obj_calls += 1;
        let (_obj_value, _obj_overlap, _obj_constraint) = compute_objective_with_buffer(
            atoms,
            positions,
            atom_params,
            &mol_info,
            overlap_atoms.as_deref(),
            pbc,
            use_short_tol,
            include_overlap,
            short_dist,
            short_scale,
            fbins,
            radmax,
            &mut objective_buffer,
        );
        let g = compute_gradient_from_grad_pos(&objective_buffer.grad_pos, &x, &mol_info);
        let eps = 1.0e-4f32;
        let mut max_rel = 0.0f32;
        let ncheck = x.len().min(20);
        for i in 0..ncheck {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += eps;
            x_minus[i] -= eps;
            update_positions(&x_plus, placements, &mol_info, atoms, positions);
            counters.obj_only_calls += 1;
            let f_plus = compute_objective_only(
                atoms,
                positions,
                atom_params,
                &mol_info,
                overlap_atoms.as_deref(),
                pbc,
                use_short_tol,
                include_overlap,
                short_dist,
                short_scale,
                fbins,
                radmax,
                &mut objective_buffer,
            );
            update_positions(&x_minus, placements, &mol_info, atoms, positions);
            counters.obj_only_calls += 1;
            let f_minus = compute_objective_only(
                atoms,
                positions,
                atom_params,
                &mol_info,
                overlap_atoms.as_deref(),
                pbc,
                use_short_tol,
                include_overlap,
                short_dist,
                short_scale,
                fbins,
                radmax,
                &mut objective_buffer,
            );
            let fd = (f_plus - f_minus) / (2.0 * eps);
            let denom = fd.abs().max(g[i].abs()).max(1.0e-6);
            let rel = (fd - g[i]).abs() / denom;
            if rel > max_rel {
                max_rel = rel;
            }
        }
        if max_rel > 1.0e-2 {
            return Err(PackError::Invalid(format!(
                "chkgrad failed: max relative error {:.3e}",
                max_rel
            )));
        }
        update_positions(&x, placements, &mol_info, atoms, positions);
    }

    let gencan_start = Instant::now();
    for iter in 0..maxit {
        iters_used = iter + 1;
        update_positions(&x, placements, &mol_info, atoms, positions);
        counters.obj_calls += 1;
        let elapsed_ms = crate::streaming::duration_ms(gencan_start.elapsed());
        let (obj_value, obj_overlap, obj_constraint) = compute_objective_with_buffer(
            atoms,
            positions,
            atom_params,
            &mol_info,
            overlap_atoms.as_deref(),
            pbc,
            use_short_tol,
            include_overlap,
            short_dist,
            short_scale,
            fbins,
            radmax,
            &mut objective_buffer,
        );
        if obj_overlap <= precision && obj_constraint <= precision {
            break;
        }
        let g = compute_gradient_from_grad_pos(&objective_buffer.grad_pos, &x, &mol_info);
        let proj = projected_stats(&x, &g, &bounds);
        let _pg = proj.pg;
        let pg_norm = proj.gpeucn2.sqrt();
        let pg_sup = proj.gpsupn;
        let gpeucn2 = proj.gpeucn2;
        let gieucn2 = proj.gieucn2;
        let nind = proj.nind;

        // Emit iteration progress event
        if let Some(ref emitter) = emitter {
            if emitter.is_enabled() {
                emitter.emit_gencan_iteration(&crate::streaming::GencanIterationEvent {
                    iteration: iter,
                    max_iterations: maxit,
                    obj_value,
                    obj_overlap,
                    obj_constraint,
                    pg_sup,
                    pg_norm,
                    elapsed_ms,
                });
            }
        }

        if gpsupn0.is_none() {
            let gps0 = pg_sup.max(1.0e-12);
            gpsupn0 = Some(gps0);
            if gps0 > 0.0 && cggpnf > 0.0 {
                let denom = (cggpnf / gps0).log10();
                if denom.is_finite() && denom.abs() > 1.0e-12 {
                    acgeps = (cgepsf / cgepsi).log10() / denom;
                    bcgeps = cgepsi.log10() - acgeps * gps0.log10();
                } else {
                    acgeps = 0.0;
                    bcgeps = cgepsf.log10();
                }
            } else {
                acgeps = 0.0;
                bcgeps = cgepsf.log10();
            }
        }
        if pg_sup <= epsgpsn {
            break;
        }
        if pg_norm < 1.0e-6 {
            break;
        }

        let xnorm = norm2(&x).sqrt();
        let (sts, sty) = match (&prev_x, &prev_g) {
            (Some(px), Some(pg_prev)) => {
                let mut s_dot_s = 0.0f32;
                let mut s_dot_y = 0.0f32;
                let n = x.len().min(px.len()).min(g.len()).min(pg_prev.len());
                for i in 0..n {
                    let s = x[i] - px[i];
                    let y = g[i] - pg_prev[i];
                    s_dot_s += s * s;
                    s_dot_y += s * y;
                }
                (s_dot_s, s_dot_y)
            }
            _ => (0.0, 0.0),
        };
        let use_spg = gieucn2 <= ometa2 * gpeucn2;

        let mut lamspg = if iter == 0 || sty <= 0.0 {
            let denom = gpeucn2.max(1.0e-12).sqrt();
            (1.0f32.max(xnorm)) / denom
        } else {
            (sts / sty).abs()
        };
        if iter == 0 {
            if let Some(step) = gencan_step {
                lamspg = step.abs();
            }
        }
        let lamspg = lamspg.clamp(lspg_min, lspg_max);
        let spg_dir = spg_direction(&x, &g, &bounds, lamspg);

        let delta = if !use_spg {
            if iter == 0 {
                delmin.max(0.1 * 1.0f32.max(xnorm))
            } else {
                delmin.max(10.0 * sts.max(0.0).sqrt())
            }
        } else {
            0.0
        };

        let mut cgmaxit = 20usize;
        let mut cgeps = cgepsi;
        if !use_spg && nind > 0 {
            let gps0 = gpsupn0.unwrap_or(pg_sup.max(1.0e-12)).max(1.0e-12);
            let mut kappa = if pg_sup > 0.0 {
                let denom = (epsgpsn / gps0).log10();
                if denom.is_finite() && denom.abs() > 1.0e-12 {
                    ((pg_sup / gps0).log10() / denom).clamp(0.0, 1.0)
                } else {
                    1.0
                }
            } else {
                1.0
            };
            if !kappa.is_finite() {
                kappa = 1.0;
            }
            let base = (10.0 * (nind as f32).max(1.0).log10()).max(1.0);
            let cg_est = ((1.0 - kappa) * base + kappa * nind as f32)
                .max(1.0)
                .min(nind as f32);
            cgmaxit = (cg_est as usize).clamp(1, 20);
            let cgeps_est = if pg_sup > 0.0 {
                10.0f32.powf(acgeps * pg_sup.log10() + bcgeps)
            } else {
                cgepsf
            };
            if cgeps_est.is_finite() {
                cgeps = cgeps_est.clamp(cgepsf, cgepsi);
            }
        }

        let mut direction = if use_spg {
            spg_dir.clone()
        } else {
            truncated_newton_direction(
                &x,
                &g,
                &bounds,
                atoms,
                positions,
                atom_params,
                placements,
                &mol_info,
                overlap_atoms.as_deref(),
                pbc,
                use_short_tol,
                include_overlap,
                short_dist,
                short_scale,
                fbins,
                radmax,
                cgmaxit,
                cgeps,
                delta,
                &mut counters,
                &mut objective_buffer,
            )
        };
        let mut gtd = dot(&g, &direction);
        if gtd >= 0.0 {
            direction = spg_dir.clone();
            gtd = dot(&g, &direction);
            if gtd >= 0.0 {
                break;
            }
        }

        if !use_spg {
            let gnorm = grad_norm(&g);
            let dnorm = grad_norm(&direction);
            if gtd > -theta * gnorm * dnorm {
                direction = spg_dir.clone();
                gtd = dot(&g, &direction);
                if gtd >= 0.0 {
                    break;
                }
            }
            let dsup = sup_norm(&direction);
            if dsup > delta && dsup > 0.0 {
                let scale = delta / dsup;
                for v in direction.iter_mut() {
                    *v *= scale;
                }
                gtd = dot(&g, &direction);
            }
        }

        let f_ref = f_hist.iter().cloned().fold(obj_value, |a, b| a.max(b));
        let mut ls = if use_spg {
            line_search_spg(
                &x,
                &direction,
                gtd,
                &bounds,
                obj_value,
                f_ref,
                gamma,
                sigma1,
                sigma2,
                nint,
                min_interp,
                max_ls,
                atoms,
                positions,
                atom_params,
                placements,
                &mol_info,
                overlap_atoms.as_deref(),
                pbc,
                use_short_tol,
                include_overlap,
                short_dist,
                short_scale,
                fbins,
                radmax,
                &mut counters,
                &mut objective_buffer,
            )
        } else {
            line_search_tn(
                &x,
                &direction,
                gtd,
                &bounds,
                obj_value,
                f_ref,
                gamma,
                beta,
                sigma1,
                sigma2,
                nint,
                next,
                min_interp,
                max_ls,
                atoms,
                positions,
                atom_params,
                placements,
                &mol_info,
                overlap_atoms.as_deref(),
                pbc,
                use_short_tol,
                include_overlap,
                short_dist,
                short_scale,
                fbins,
                radmax,
                &mut counters,
                &mut objective_buffer,
            )
        };
        // Match Packmol: if TN line-search fails due to tiny step, force an SPG iteration.
        if ls.is_none() && !use_spg {
            let spg_gtd = dot(&g, &spg_dir);
            if spg_gtd < 0.0 {
                ls = line_search_spg(
                    &x,
                    &spg_dir,
                    spg_gtd,
                    &bounds,
                    obj_value,
                    f_ref,
                    gamma,
                    sigma1,
                    sigma2,
                    nint,
                    min_interp,
                    max_ls,
                    atoms,
                    positions,
                    atom_params,
                    placements,
                    &mol_info,
                    overlap_atoms.as_deref(),
                    pbc,
                    use_short_tol,
                    include_overlap,
                    short_dist,
                    short_scale,
                    fbins,
                    radmax,
                    &mut counters,
                    &mut objective_buffer,
                );
            }
        }
        let (x_new, f_new) = match ls {
            Some(v) => v,
            None => break,
        };

        if f_hist.len() == nonmon_m {
            f_hist.pop_front();
        }
        f_hist.push_back(f_new);

        prev_x = Some(x);
        prev_g = Some(g);
        x = x_new;

        if let Some(freq) = cfg.iprint1 {
            if freq > 0 && (iter as i32) % freq == 0 {
                eprintln!("gencan iter {} f={:.6} pg={:.6}", iter, f_new, pg_norm);
            }
        }
    }

    update_positions(&x, placements, &mol_info, atoms, positions);
    counters.obj_calls += 1;
    let (obj_value, obj_overlap, obj_constraint) = compute_objective_with_buffer(
        atoms,
        positions,
        atom_params,
        &mol_info,
        overlap_atoms.as_deref(),
        pbc,
        use_short_tol,
        include_overlap,
        short_dist,
        short_scale,
        fbins,
        radmax,
        &mut objective_buffer,
    );
    if trace_stats {
        eprintln!(
            "gencan stats active_spec={:?} active_mol={:?} include_overlap={} iters={} obj={} obj_only={} eval_grad={} hess_vec={} cg_iters={} ls_trials={}",
            active_spec,
            active_mol,
            include_overlap,
            iters_used,
            counters.obj_calls,
            counters.obj_only_calls,
            counters.eval_grad_calls,
            counters.hess_vec_calls,
            counters.cg_iters,
            counters.ls_trials
        );
    }
    Ok(GencanResult {
        value: obj_value,
        max_overlap: obj_overlap,
        max_constraint: obj_constraint,
    })
}

pub(crate) fn evaluate_state(
    cfg: &PackConfig,
    atoms: &[AtomRecord],
    positions: &[Vec3],
    atom_params: &[AtomParams],
    mol_spec: &[usize],
    templates: &[TemplateEntry],
    pbc: Option<PbcBox>,
    use_short_tol: bool,
    include_overlap: bool,
    active_spec: Option<usize>,
    active_mol: Option<usize>,
    objective_buffer: &mut ObjectiveBuffer,
) -> PackResult<GencanResult> {
    let mol_info = build_mol_info(cfg, atoms, mol_spec, templates, active_spec, active_mol)?;
    let overlap_atoms = collect_selected_overlap_atoms(atoms, &mol_info);
    let radmax = atom_params.iter().fold(0.0f32, |acc, p| acc.max(p.radius));
    let short_dist = if use_short_tol {
        cfg.short_tol_dist.unwrap_or(0.0)
    } else {
        0.0
    };
    let short_scale = if use_short_tol {
        cfg.short_tol_scale.unwrap_or(1.0)
    } else {
        1.0
    };
    let (value, max_overlap, max_constraint) = compute_objective_with_buffer(
        atoms,
        positions,
        atom_params,
        &mol_info,
        overlap_atoms.as_deref(),
        pbc,
        use_short_tol,
        include_overlap,
        short_dist,
        short_scale,
        cfg.fbins.unwrap_or(3.0f32.sqrt()),
        radmax,
        objective_buffer,
    );
    Ok(GencanResult {
        value,
        max_overlap,
        max_constraint,
    })
}

fn collect_selected_overlap_atoms(
    atoms: &[AtomRecord],
    mol_info: &[MolInfo],
) -> Option<Vec<usize>> {
    let selected_subset =
        mol_info.iter().any(|m| m.selected) && mol_info.iter().any(|m| !m.selected);
    if !selected_subset {
        return None;
    }
    let mut indices = Vec::new();
    for (idx, atom) in atoms.iter().enumerate() {
        let mol_idx = atom.mol_id.max(1) as usize - 1;
        if mol_idx < mol_info.len() && mol_info[mol_idx].selected {
            indices.push(idx);
        }
    }
    Some(indices)
}

fn build_mol_info(
    cfg: &PackConfig,
    atoms: &[AtomRecord],
    mol_spec: &[usize],
    templates: &[TemplateEntry],
    active_spec: Option<usize>,
    active_mol: Option<usize>,
) -> PackResult<Vec<MolInfo>> {
    let total_mols = mol_spec.len();
    let mut mol_atoms: Vec<Vec<usize>> = vec![Vec::new(); total_mols];
    for (idx, atom) in atoms.iter().enumerate() {
        let mol_idx = atom.mol_id.max(1) as usize - 1;
        if mol_idx < total_mols {
            mol_atoms[mol_idx].push(idx);
        }
    }
    let mut mol_info = Vec::with_capacity(total_mols);
    for (mol_idx, &spec_index) in mol_spec.iter().enumerate() {
        let spec = &cfg.structures[spec_index];
        let entry = templates
            .get(spec_index)
            .ok_or_else(|| PackError::Placement("missing template entry".into()))?;
        let active_spec_ok = active_spec.map_or(true, |idx| idx == spec_index);
        let active_mol_ok = active_mol.map_or(true, |idx| idx == mol_idx);
        let active = active_spec_ok && active_mol_ok;
        let selected = if active_mol.is_some() {
            active_mol_ok
        } else {
            active_spec_ok
        };
        let atom_indices = mol_atoms[mol_idx].clone();
        validate_molecule_counts(&atom_indices, entry.template.atoms.len())?;
        let local_positions = entry
            .template
            .atoms
            .iter()
            .map(|a| a.position)
            .collect::<Vec<_>>();
        let atom_constraints = if active {
            let mut constraints = Vec::new();
            for atom_constraint in &spec.atom_constraints {
                for &idx in atom_constraint.indices.iter() {
                    if let Some(local_idx) = idx.checked_sub(1) {
                        constraints.push((local_idx, atom_constraint.constraint.clone()));
                    }
                }
            }
            constraints
        } else {
            Vec::new()
        };
        mol_info.push(MolInfo {
            atom_indices,
            local_positions,
            radius: entry.template.radius,
            selected,
            movable: !spec.fixed && active,
            rotatable: spec.rotate && active,
            constraints: if active {
                spec.constraints.clone()
            } else {
                Vec::new()
            },
            atom_constraints,
            rot_bounds: spec.rot_bounds,
        });
    }
    Ok(mol_info)
}

fn spg_direction(x: &[f32], g: &[f32], bounds: &Bounds, lamspg: f32) -> Vec<f32> {
    let trial = project_step(x, g, -lamspg, bounds);
    let mut dir = Vec::with_capacity(x.len());
    let n = x.len().min(trial.len());
    for i in 0..n {
        dir.push(trial[i] - x[i]);
    }
    if n < x.len() {
        dir.resize(x.len(), 0.0);
    }
    dir
}

fn max_step_to_bounds(x: &[f32], direction: &[f32], bounds: &Bounds) -> f32 {
    let n = x
        .len()
        .min(direction.len())
        .min(bounds.lower.len())
        .min(bounds.upper.len());
    let mut amax = f32::INFINITY;
    for i in 0..n {
        let di = direction[i];
        if di > 0.0 {
            let a = (bounds.upper[i] - x[i]) / di;
            if a < amax {
                amax = a;
            }
        } else if di < 0.0 {
            let a = (bounds.lower[i] - x[i]) / di;
            if a < amax {
                amax = a;
            }
        }
    }
    if amax.is_finite() {
        amax.max(0.0)
    } else {
        1.0
    }
}

fn line_search_spg(
    x: &[f32],
    direction: &[f32],
    gtd: f32,
    bounds: &Bounds,
    f0: f32,
    f_ref: f32,
    gamma: f32,
    sigma1: f32,
    sigma2: f32,
    nint: f32,
    min_interp: usize,
    max_ls: usize,
    atoms: &mut [AtomRecord],
    positions: &mut [Vec3],
    atom_params: &[AtomParams],
    placements: &mut [PlacementRecord],
    mols: &[MolInfo],
    overlap_atoms: Option<&[usize]>,
    pbc: Option<PbcBox>,
    use_short_tol: bool,
    include_overlap: bool,
    short_dist: f32,
    short_scale: f32,
    fbins: f32,
    radmax: f32,
    counters: &mut GencanCounters,
    objective_buffer: &mut ObjectiveBuffer,
) -> Option<(Vec<f32>, f32)> {
    if gtd >= 0.0 {
        return None;
    }
    let mut step = 1.0f32;
    let mut interp = 0usize;
    for _ in 0..max_ls {
        counters.ls_trials += 1;
        let x_new = project_step(x, direction, step, bounds);
        update_positions(&x_new, placements, mols, atoms, positions);
        counters.obj_only_calls += 1;
        let f_new = compute_objective_only(
            atoms,
            positions,
            atom_params,
            mols,
            overlap_atoms,
            pbc,
            use_short_tol,
            include_overlap,
            short_dist,
            short_scale,
            fbins,
            radmax,
            objective_buffer,
        );
        if f_new <= f_ref + gamma * step * gtd {
            return Some((x_new, f_new));
        }
        interp += 1;
        if interp >= min_interp && step < 1.0e-12 {
            return None;
        }
        let denom = 2.0 * (f_new - f0 - step * gtd);
        let mut step_new = if denom > 0.0 {
            -gtd * step * step / denom
        } else {
            step / nint
        };
        let lower = sigma1 * step;
        let upper = sigma2 * step;
        if step_new < lower || step_new > upper {
            step_new = step / nint;
        }
        step = step_new;
    }
    None
}

fn line_search_tn(
    x: &[f32],
    direction: &[f32],
    gtd: f32,
    bounds: &Bounds,
    f0: f32,
    _f_ref: f32,
    gamma: f32,
    beta: f32,
    sigma1: f32,
    sigma2: f32,
    nint: f32,
    next: f32,
    min_interp: usize,
    max_ls: usize,
    atoms: &mut [AtomRecord],
    positions: &mut [Vec3],
    atom_params: &[AtomParams],
    placements: &mut [PlacementRecord],
    mols: &[MolInfo],
    overlap_atoms: Option<&[usize]>,
    pbc: Option<PbcBox>,
    use_short_tol: bool,
    include_overlap: bool,
    short_dist: f32,
    short_scale: f32,
    fbins: f32,
    radmax: f32,
    counters: &mut GencanCounters,
    objective_buffer: &mut ObjectiveBuffer,
) -> Option<(Vec<f32>, f32)> {
    if gtd >= 0.0 {
        return None;
    }
    let amax = max_step_to_bounds(x, direction, bounds);
    if !amax.is_finite() || amax <= 0.0 {
        return None;
    }

    let mut alpha = amax.min(1.0);
    let mut x_plus = project_step(x, direction, alpha, bounds);
    counters.ls_trials += 1;
    counters.obj_only_calls += 1;
    update_positions(&x_plus, placements, mols, atoms, positions);
    let mut f_plus = compute_objective_only(
        atoms,
        positions,
        atom_params,
        mols,
        overlap_atoms,
        pbc,
        use_short_tol,
        include_overlap,
        short_dist,
        short_scale,
        fbins,
        radmax,
        objective_buffer,
    );
    let mut fevals = 1usize;

    // Packmol TNLS logic: beta-condition only on the first interior Armijo trial.
    if amax > 1.0 {
        if f_plus <= f0 + gamma * alpha * gtd {
            let mut g_plus = Vec::<f32>::new();
            eval_grad(
                &x_plus,
                atoms,
                positions,
                atom_params,
                placements,
                mols,
                overlap_atoms,
                pbc,
                use_short_tol,
                include_overlap,
                short_dist,
                short_scale,
                fbins,
                radmax,
                counters,
                objective_buffer,
                &mut g_plus,
            );
            let gptd = dot(&g_plus, direction);
            if gptd < beta * gtd {
                while fevals < max_ls {
                    let atmp = if alpha < amax && next * alpha > amax {
                        amax
                    } else {
                        next * alpha
                    };
                    let x_tmp = project_step(x, direction, atmp, bounds);
                    let mut samep = true;
                    let n = x_tmp.len().min(x_plus.len());
                    for i in 0..n {
                        let thr = (1.0e-10f32 * x_plus[i].abs()).max(1.0e-20);
                        if (x_tmp[i] - x_plus[i]).abs() > thr {
                            samep = false;
                            break;
                        }
                    }
                    if samep {
                        break;
                    }
                    counters.ls_trials += 1;
                    counters.obj_only_calls += 1;
                    update_positions(&x_tmp, placements, mols, atoms, positions);
                    let f_tmp = compute_objective_only(
                        atoms,
                        positions,
                        atom_params,
                        mols,
                        overlap_atoms,
                        pbc,
                        use_short_tol,
                        include_overlap,
                        short_dist,
                        short_scale,
                        fbins,
                        radmax,
                        objective_buffer,
                    );
                    fevals += 1;
                    if f_tmp < f_plus {
                        alpha = atmp;
                        x_plus = x_tmp;
                        f_plus = f_tmp;
                    } else {
                        break;
                    }
                }
            }
            return Some((x_plus, f_plus));
        }
    } else if f_plus < f0 {
        while fevals < max_ls {
            let atmp = if alpha < amax && next * alpha > amax {
                amax
            } else {
                next * alpha
            };
            let x_tmp = project_step(x, direction, atmp, bounds);
            let mut samep = true;
            let n = x_tmp.len().min(x_plus.len());
            for i in 0..n {
                let thr = (1.0e-10f32 * x_plus[i].abs()).max(1.0e-20);
                if (x_tmp[i] - x_plus[i]).abs() > thr {
                    samep = false;
                    break;
                }
            }
            if samep {
                break;
            }
            counters.ls_trials += 1;
            counters.obj_only_calls += 1;
            update_positions(&x_tmp, placements, mols, atoms, positions);
            let f_tmp = compute_objective_only(
                atoms,
                positions,
                atom_params,
                mols,
                overlap_atoms,
                pbc,
                use_short_tol,
                include_overlap,
                short_dist,
                short_scale,
                fbins,
                radmax,
                objective_buffer,
            );
            fevals += 1;
            if f_tmp < f_plus {
                alpha = atmp;
                x_plus = x_tmp;
                f_plus = f_tmp;
            } else {
                break;
            }
        }
        return Some((x_plus, f_plus));
    }

    // Interpolation branch.
    let epsrel = 1.0e-10f32;
    let epsabs = 1.0e-20f32;
    let mut interp = 0usize;
    loop {
        if f_plus <= f0 + gamma * alpha * gtd {
            return Some((x_plus, f_plus));
        }
        interp += 1;
        if fevals >= max_ls {
            return None;
        }
        let alpha_new = if alpha < sigma1 {
            alpha / nint
        } else {
            let denom = 2.0 * (f_plus - f0 - alpha * gtd);
            let mut atmp = if denom > 0.0 {
                -gtd * alpha * alpha / denom
            } else {
                alpha / nint
            };
            if atmp < sigma1 || atmp > sigma2 * alpha {
                atmp = alpha / nint;
            }
            atmp
        };
        if !alpha_new.is_finite() || alpha_new <= 0.0 {
            return None;
        }
        let mut samep = true;
        let n = direction.len().min(x.len());
        for i in 0..n {
            let thr = (epsrel * x[i].abs()).max(epsabs);
            if (alpha_new * direction[i]).abs() > thr {
                samep = false;
                break;
            }
        }
        if interp >= min_interp && samep {
            return None;
        }
        alpha = alpha_new;
        x_plus = project_step(x, direction, alpha, bounds);
        counters.ls_trials += 1;
        counters.obj_only_calls += 1;
        update_positions(&x_plus, placements, mols, atoms, positions);
        f_plus = compute_objective_only(
            atoms,
            positions,
            atom_params,
            mols,
            overlap_atoms,
            pbc,
            use_short_tol,
            include_overlap,
            short_dist,
            short_scale,
            fbins,
            radmax,
            objective_buffer,
        );
        fevals += 1;
    }
}

fn truncated_newton_direction(
    x: &[f32],
    g: &[f32],
    bounds: &Bounds,
    atoms: &mut [AtomRecord],
    positions: &mut [Vec3],
    atom_params: &[AtomParams],
    placements: &mut [PlacementRecord],
    mols: &[MolInfo],
    overlap_atoms: Option<&[usize]>,
    pbc: Option<PbcBox>,
    use_short_tol: bool,
    include_overlap: bool,
    short_dist: f32,
    short_scale: f32,
    fbins: f32,
    radmax: f32,
    cg_maxit: usize,
    cgeps: f32,
    delta: f32,
    counters: &mut GencanCounters,
    objective_buffer: &mut ObjectiveBuffer,
) -> Vec<f32> {
    let n = x.len();
    let mask = active_mask(x, bounds);
    let mut d = vec![0.0f32; n];
    let mut d_prev = vec![0.0f32; n];
    let mut r = vec![0.0f32; n];
    let mut p = vec![0.0f32; n];
    let mut ap = vec![0.0f32; n];
    let mut x2 = vec![0.0f32; n];
    let mut g2 = vec![0.0f32; n];
    let gnorm2 = dot_mask(g, g, &mask);
    let theta = 1.0e-6f32;
    let epsrel = 1.0e-10f32;
    let epsabs = 1.0e-20f32;
    let epsnqmp = 1.0e-4f32;
    let maxitnqmp = 5usize;
    let mut q = 0.0f32;
    let mut bestprog = 0.0f32;
    let mut itnqmp = 0usize;
    for i in 0..n {
        if mask[i] {
            r[i] = -g[i];
            p[i] = r[i];
        }
    }
    let mut rs_old = dot_mask(&r, &r, &mask);
    if rs_old.sqrt() < 1.0e-6 {
        return d;
    }
    let max_cg = cg_maxit.max(1);
    let tol = cgeps.clamp(1.0e-8, 1.0).max(1.0e-8) * rs_old.sqrt();

    for _ in 0..max_cg {
        counters.cg_iters += 1;
        d_prev.copy_from_slice(&d);
        let qprev = q;
        hess_vec(
            x,
            g,
            &p,
            atoms,
            positions,
            atom_params,
            placements,
            mols,
            overlap_atoms,
            pbc,
            use_short_tol,
            include_overlap,
            short_dist,
            short_scale,
            fbins,
            radmax,
            counters,
            objective_buffer,
            &mut ap,
            &mut x2,
            &mut g2,
        );
        let p_ap = dot_mask(&p, &ap, &mask);
        if p_ap <= 1.0e-8 {
            break;
        }
        let r_dot_p = dot_mask(&r, &p, &mask);
        let alpha = rs_old / p_ap;
        for i in 0..n {
            if mask[i] {
                d[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
        }
        q = q + 0.5 * alpha * alpha * p_ap - alpha * r_dot_p;
        if delta > 0.0 {
            let dsup = sup_norm_mask(&d, &mask);
            if dsup >= delta {
                if dsup > 0.0 {
                    let scale = delta / dsup;
                    for i in 0..n {
                        if mask[i] {
                            d[i] *= scale;
                        }
                    }
                }
                break;
            }
        }
        let rs_new = dot_mask(&r, &r, &mask);
        if rs_new.sqrt() < tol {
            break;
        }
        let gts = dot_mask(g, &d, &mask);
        let snorm2 = dot_mask(&d, &d, &mask);
        if snorm2 > 0.0 && (gts > 0.0 || gts * gts < theta * theta * gnorm2.max(0.0) * snorm2) {
            d.copy_from_slice(&d_prev);
            break;
        }
        let mut samep = true;
        for i in 0..n {
            if !mask[i] {
                continue;
            }
            if (alpha * p[i]).abs() > (epsrel * d[i].abs()).max(epsabs) {
                samep = false;
                break;
            }
        }
        if samep {
            break;
        }
        let currprog = qprev - q;
        if currprog.is_finite() {
            bestprog = bestprog.max(currprog);
            if bestprog > 0.0 && currprog <= epsnqmp * bestprog {
                itnqmp += 1;
                if itnqmp >= maxitnqmp {
                    break;
                }
            } else {
                itnqmp = 0;
            }
        }
        let beta = rs_new / rs_old;
        for i in 0..n {
            if mask[i] {
                p[i] = r[i] + beta * p[i];
            }
        }
        rs_old = rs_new;
    }
    d
}

fn hess_vec(
    x: &[f32],
    grad_at_x: &[f32],
    v: &[f32],
    atoms: &mut [AtomRecord],
    positions: &mut [Vec3],
    atom_params: &[AtomParams],
    placements: &mut [PlacementRecord],
    mols: &[MolInfo],
    overlap_atoms: Option<&[usize]>,
    pbc: Option<PbcBox>,
    use_short_tol: bool,
    include_overlap: bool,
    short_dist: f32,
    short_scale: f32,
    fbins: f32,
    radmax: f32,
    counters: &mut GencanCounters,
    objective_buffer: &mut ObjectiveBuffer,
    hv: &mut Vec<f32>,
    x2: &mut Vec<f32>,
    g2: &mut Vec<f32>,
) {
    counters.hess_vec_calls += 1;
    let n = x.len();
    let mut x_sup = 0.0f32;
    let mut d_sup = 0.0f32;
    let m = n.min(v.len());
    for i in 0..m {
        x_sup = x_sup.max(x[i].abs());
        d_sup = d_sup.max(v[i].abs());
    }
    if d_sup < 1.0e-20 {
        d_sup = 1.0e-20;
    }
    let eps = (1.0e-7f32 * x_sup).max(1.0e-10f32) / d_sup;
    if x2.len() != n {
        x2.resize(n, 0.0);
    }
    for i in 0..n {
        x2[i] = x[i] + v.get(i).copied().unwrap_or(0.0) * eps;
    }
    eval_grad(
        x2,
        atoms,
        positions,
        atom_params,
        placements,
        mols,
        overlap_atoms,
        pbc,
        use_short_tol,
        include_overlap,
        short_dist,
        short_scale,
        fbins,
        radmax,
        counters,
        objective_buffer,
        g2,
    );
    if hv.len() != n {
        hv.resize(n, 0.0);
    }
    let m = n.min(g2.len()).min(grad_at_x.len());
    for i in 0..m {
        hv[i] = (g2[i] - grad_at_x[i]) / eps;
    }
}

fn eval_grad(
    x: &[f32],
    atoms: &mut [AtomRecord],
    positions: &mut [Vec3],
    atom_params: &[AtomParams],
    placements: &mut [PlacementRecord],
    mols: &[MolInfo],
    overlap_atoms: Option<&[usize]>,
    pbc: Option<PbcBox>,
    use_short_tol: bool,
    include_overlap: bool,
    short_dist: f32,
    short_scale: f32,
    fbins: f32,
    radmax: f32,
    counters: &mut GencanCounters,
    objective_buffer: &mut ObjectiveBuffer,
    grad_out: &mut Vec<f32>,
) {
    counters.eval_grad_calls += 1;
    update_positions(x, placements, mols, atoms, positions);
    counters.obj_calls += 1;
    let (_value, _max_overlap, _max_constraint) = compute_objective_with_buffer(
        atoms,
        positions,
        atom_params,
        mols,
        overlap_atoms,
        pbc,
        use_short_tol,
        include_overlap,
        short_dist,
        short_scale,
        fbins,
        radmax,
        objective_buffer,
    );
    compute_gradient_from_grad_pos_into(&objective_buffer.grad_pos, x, mols, grad_out);
}

fn active_mask(x: &[f32], bounds: &Bounds) -> Vec<bool> {
    let mut mask = vec![true; x.len()];
    let n = x.len().min(bounds.lower.len()).min(bounds.upper.len());
    for i in 0..n {
        let lo = bounds.lower[i];
        let hi = bounds.upper[i];
        mask[i] = x[i] > lo && x[i] < hi;
    }
    mask
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let n = a.len().min(b.len());
    for i in 0..n {
        sum += a[i] * b[i];
    }
    sum
}

fn norm2(a: &[f32]) -> f32 {
    dot(a, a)
}

fn dot_mask(a: &[f32], b: &[f32], mask: &[bool]) -> f32 {
    let mut sum = 0.0f32;
    let n = a.len().min(b.len()).min(mask.len());
    for i in 0..n {
        if mask[i] {
            sum += a[i] * b[i];
        }
    }
    sum
}

fn sup_norm(a: &[f32]) -> f32 {
    let mut out = 0.0f32;
    for &v in a {
        out = out.max(v.abs());
    }
    out
}

fn sup_norm_mask(a: &[f32], mask: &[bool]) -> f32 {
    let mut out = 0.0f32;
    let n = a.len().min(mask.len());
    for i in 0..n {
        if mask[i] {
            out = out.max(a[i].abs());
        }
    }
    out
}

fn projected_stats(x: &[f32], grad: &[f32], bounds: &Bounds) -> ProjectedStats {
    let mut pg = Vec::with_capacity(x.len());
    let n = x
        .len()
        .min(grad.len())
        .min(bounds.lower.len())
        .min(bounds.upper.len());
    let mut gpsupn = 0.0f32;
    let mut gpeucn2 = 0.0f32;
    let mut gieucn2 = 0.0f32;
    let mut nind = 0usize;
    for i in 0..n {
        let lo = bounds.lower[i];
        let hi = bounds.upper[i];
        let proj = (x[i] - grad[i]).clamp(lo, hi);
        let gpi = proj - x[i];
        pg.push(gpi);
        gpsupn = gpsupn.max(gpi.abs());
        gpeucn2 += gpi * gpi;
        if x[i] > lo && x[i] < hi {
            gieucn2 += gpi * gpi;
            nind += 1;
        }
    }
    if n < x.len() {
        pg.resize(x.len(), 0.0);
    }
    ProjectedStats {
        pg,
        gpsupn,
        gpeucn2,
        gieucn2,
        nind,
    }
}
