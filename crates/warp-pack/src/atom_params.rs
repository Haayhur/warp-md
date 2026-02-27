use crate::config::{AtomOverride, PackConfig, StructureSpec};
use crate::error::{PackError, PackResult};

#[derive(Clone, Copy, Debug)]
pub(crate) struct AtomParams {
    pub(crate) radius: f32,
    pub(crate) fscale: f32,
    pub(crate) short_radius: f32,
    pub(crate) short_scale: f32,
    pub(crate) use_short: bool,
}

pub(crate) fn build_atom_params(
    cfg: &PackConfig,
    spec: &StructureSpec,
    atom_count: usize,
    dist_scale: f32,
) -> PackResult<Vec<AtomParams>> {
    let base_radius = spec
        .radius
        .or(spec.min_distance.map(|v| v * 0.5))
        .or(cfg.min_distance.map(|v| v * 0.5))
        .unwrap_or(1.0)
        .abs();
    let base_fscale = spec.fscale.unwrap_or(1.0);
    let mut use_short = cfg.use_short_tol;
    let mut base_short_radius = cfg.short_tol_dist.map(|v| v * 0.5).unwrap_or(0.0).abs();
    let mut base_short_scale = cfg.short_tol_scale.unwrap_or(1.0);
    if let Some(v) = spec.short_radius {
        base_short_radius = v.abs();
        use_short = true;
    }
    if let Some(v) = spec.short_radius_scale {
        base_short_scale = v.abs();
        use_short = true;
    }
    let radius = base_radius * dist_scale;
    let short_radius = base_short_radius * dist_scale;
    let mut params = vec![
        AtomParams {
            radius,
            fscale: base_fscale,
            short_radius,
            short_scale: base_short_scale,
            use_short,
        };
        atom_count
    ];

    apply_overrides(&mut params, &spec.atom_overrides)?;

    for (idx, p) in params.iter_mut().enumerate() {
        if p.radius <= 0.0 {
            return Err(PackError::Invalid(format!(
                "atom radius must be positive (index {})",
                idx + 1
            )));
        }
        if p.use_short {
            if p.short_radius <= 0.0 {
                return Err(PackError::Invalid(format!(
                    "short radius must be positive (index {})",
                    idx + 1
                )));
            }
            if p.short_radius >= p.radius {
                return Err(PackError::Invalid(format!(
                    "short radius must be smaller than radius (index {})",
                    idx + 1
                )));
            }
        }
        if p.fscale <= 0.0 {
            return Err(PackError::Invalid(format!(
                "fscale must be positive (index {})",
                idx + 1
            )));
        }
        if p.short_scale <= 0.0 {
            p.short_scale = 1.0;
        }
    }

    Ok(params)
}

fn apply_overrides(params: &mut [AtomParams], overrides: &[AtomOverride]) -> PackResult<()> {
    for ov in overrides {
        for &idx1 in ov.indices.iter() {
            if idx1 == 0 || idx1 > params.len() {
                return Err(PackError::Invalid(format!(
                    "atom override index {} out of range",
                    idx1
                )));
            }
            let idx = idx1 - 1;
            if let Some(v) = ov.radius {
                params[idx].radius = v.abs();
            }
            if let Some(v) = ov.fscale {
                params[idx].fscale = v;
            }
            if let Some(v) = ov.short_radius {
                params[idx].short_radius = v.abs();
                params[idx].use_short = true;
            }
            if let Some(v) = ov.short_radius_scale {
                params[idx].short_scale = v.abs();
                params[idx].use_short = true;
            }
        }
    }
    Ok(())
}

pub(crate) fn max_radius(params: &[AtomParams]) -> f32 {
    params.iter().fold(0.0f32, |acc, p| acc.max(p.radius))
}

pub(crate) fn scale_atom_params(params: &[AtomParams], scale: f32) -> Vec<AtomParams> {
    if (scale - 1.0).abs() < f32::EPSILON {
        return params.to_vec();
    }
    params
        .iter()
        .map(|p| AtomParams {
            radius: p.radius * scale,
            fscale: p.fscale,
            use_short: p.use_short,
            short_radius: p.short_radius * scale,
            short_scale: p.short_scale,
        })
        .collect()
}
