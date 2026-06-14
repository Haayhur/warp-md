use anyhow::{anyhow, Result};

use crate::bonded_terms::{BondedTermSet, VirtualSiteTerm};

pub(super) fn coordinate_count_with_virtual_sites(
    mapped_bead_count: usize,
    terms: &BondedTermSet,
) -> usize {
    terms
        .virtual_sites
        .iter()
        .map(|site| site.site + 1)
        .max()
        .unwrap_or(mapped_bead_count)
        .max(mapped_bead_count)
}

pub(super) fn apply_virtual_sites(
    coords: &mut Vec<[f32; 3]>,
    terms: &BondedTermSet,
    masses: Option<&[f64]>,
) -> Result<()> {
    let target_count = coordinate_count_with_virtual_sites(coords.len(), terms);
    if coords.len() < target_count {
        coords.resize(target_count, [0.0, 0.0, 0.0]);
    }
    for site in &terms.virtual_sites {
        validate_site(site, coords.len())?;
        coords[site.site] = virtual_site_coord(coords, site, masses)?;
    }
    Ok(())
}

fn validate_site(site: &VirtualSiteTerm, coord_count: usize) -> Result<()> {
    if site.site >= coord_count {
        return Err(anyhow!(
            "virtual site {} exceeds mapped coordinate count {}",
            site.site + 1,
            coord_count
        ));
    }
    for &idx in &site.defining_beads {
        if idx >= coord_count {
            return Err(anyhow!(
                "virtual site {} uses defining bead {} but mapped coordinate count is {}",
                site.site + 1,
                idx + 1,
                coord_count
            ));
        }
    }
    Ok(())
}

fn virtual_site_coord(
    coords: &[[f32; 3]],
    site: &VirtualSiteTerm,
    masses: Option<&[f64]>,
) -> Result<[f32; 3]> {
    match (site.kind.as_str(), site.function) {
        ("2", 1) => vs2_func_1(coords, site),
        ("2", 2) => vs2_func_2(coords, site),
        ("3", 1) => vs3_func_1(coords, site),
        ("3", 2) => vs3_func_2(coords, site),
        ("3", 3) => vs3_func_3(coords, site),
        ("3", 4) => vs3_func_4(coords, site),
        ("4", 2) => vs4_func_2(coords, site),
        ("n", 1) => vsn_func_1(coords, site),
        ("n", 2) => vsn_func_2(coords, site, masses),
        ("n", 3) => vsn_func_3(coords, site, masses),
        (kind, function) => Err(anyhow!(
            "unsupported virtual site kind virtual_sites{kind} function {function}"
        )),
    }
}

fn vs2_func_1(coords: &[[f32; 3]], site: &VirtualSiteTerm) -> Result<[f32; 3]> {
    require_defs(site, 2)?;
    require_params(site, 1)?;
    let i = coords[site.defining_beads[0]];
    let j = coords[site.defining_beads[1]];
    let a = site.parameters[0] as f32;
    Ok(add(scale(i, 1.0 - a), scale(j, a)))
}

fn vs2_func_2(coords: &[[f32; 3]], site: &VirtualSiteTerm) -> Result<[f32; 3]> {
    require_defs(site, 2)?;
    require_params(site, 1)?;
    let i = coords[site.defining_beads[0]];
    let j = coords[site.defining_beads[1]];
    Ok(add(i, scale(unit(sub(j, i))?, site.parameters[0] as f32)))
}

fn vs3_func_1(coords: &[[f32; 3]], site: &VirtualSiteTerm) -> Result<[f32; 3]> {
    require_defs(site, 3)?;
    require_params(site, 2)?;
    let i = coords[site.defining_beads[0]];
    let j = coords[site.defining_beads[1]];
    let k = coords[site.defining_beads[2]];
    let a = site.parameters[0] as f32;
    let b = site.parameters[1] as f32;
    Ok(add(
        i,
        add(
            scale(unit(sub(j, i))?, a / 2.0),
            scale(unit(sub(k, i))?, b / 2.0),
        ),
    ))
}

fn vs3_func_2(coords: &[[f32; 3]], site: &VirtualSiteTerm) -> Result<[f32; 3]> {
    require_defs(site, 3)?;
    require_params(site, 2)?;
    let i = coords[site.defining_beads[0]];
    let j = coords[site.defining_beads[1]];
    let k = coords[site.defining_beads[2]];
    let a = site.parameters[0] as f32;
    let b = site.parameters[1] as f32;
    let combined = add(scale(sub(j, i), 1.0 - a), scale(sub(k, j), a));
    Ok(add(i, scale(unit(combined)?, b)))
}

fn vs3_func_3(coords: &[[f32; 3]], site: &VirtualSiteTerm) -> Result<[f32; 3]> {
    require_defs(site, 3)?;
    require_params(site, 2)?;
    let i = coords[site.defining_beads[0]];
    let j = coords[site.defining_beads[1]];
    let k = coords[site.defining_beads[2]];
    let angle = (site.parameters[0] as f32).to_radians();
    let distance = site.parameters[1] as f32;
    let rij = sub(j, i);
    let rjk = sub(k, j);
    let projection = scale(rij, dot(rij, rjk) / dot(rij, rij));
    let in_plane = sub(rjk, projection);
    Ok(add(
        i,
        add(
            scale(unit(rij)?, distance * angle.cos()),
            scale(unit(in_plane)?, distance * angle.sin()),
        ),
    ))
}

fn vs3_func_4(coords: &[[f32; 3]], site: &VirtualSiteTerm) -> Result<[f32; 3]> {
    require_defs(site, 3)?;
    require_params(site, 3)?;
    let i = coords[site.defining_beads[0]];
    let j = coords[site.defining_beads[1]];
    let k = coords[site.defining_beads[2]];
    let a = site.parameters[0] as f32;
    let b = site.parameters[1] as f32;
    let c = site.parameters[2] as f32;
    let rij = sub(j, i);
    let rik = sub(k, i);
    Ok(add(
        i,
        sub(
            add(scale(rij, a), scale(rik, b)),
            scale(cross(unit(rij)?, unit(rik)?), c),
        ),
    ))
}

fn vs4_func_2(coords: &[[f32; 3]], site: &VirtualSiteTerm) -> Result<[f32; 3]> {
    require_defs(site, 4)?;
    require_params(site, 3)?;
    let i = coords[site.defining_beads[0]];
    let j = coords[site.defining_beads[1]];
    let k = coords[site.defining_beads[2]];
    let l = coords[site.defining_beads[3]];
    let a = site.parameters[0] as f32;
    let b = site.parameters[1] as f32;
    let c = site.parameters[2] as f32;
    let rij = sub(j, i);
    let rik = sub(k, i);
    let ril = sub(l, i);
    let rja = sub(scale(rik, a), rij);
    let rjb = sub(scale(ril, b), rij);
    Ok(sub(i, scale(unit(cross(rja, rjb))?, c)))
}

fn vsn_func_1(coords: &[[f32; 3]], site: &VirtualSiteTerm) -> Result<[f32; 3]> {
    require_any_defs(site)?;
    weighted_center(coords, &site.defining_beads, None)
}

fn vsn_func_2(
    coords: &[[f32; 3]],
    site: &VirtualSiteTerm,
    masses: Option<&[f64]>,
) -> Result<[f32; 3]> {
    require_any_defs(site)?;
    weighted_center(coords, &site.defining_beads, masses)
}

fn vsn_func_3(
    coords: &[[f32; 3]],
    site: &VirtualSiteTerm,
    masses: Option<&[f64]>,
) -> Result<[f32; 3]> {
    require_any_defs(site)?;
    if site.parameters.len() != site.defining_beads.len() {
        return Err(anyhow!(
            "virtual_sitesn function 3 for site {} requires one weight per defining bead",
            site.site + 1
        ));
    }
    let weights = site
        .defining_beads
        .iter()
        .enumerate()
        .map(|(idx, &bead)| {
            let mass = masses
                .and_then(|values| values.get(bead))
                .copied()
                .filter(|value| value.is_finite() && *value > 0.0)
                .unwrap_or(1.0);
            mass * site.parameters[idx]
        })
        .collect::<Vec<_>>();
    weighted_center_values(coords, &site.defining_beads, &weights)
}

fn weighted_center(
    coords: &[[f32; 3]],
    indices: &[usize],
    masses: Option<&[f64]>,
) -> Result<[f32; 3]> {
    let weights = indices
        .iter()
        .map(|&idx| {
            masses
                .and_then(|values| values.get(idx))
                .copied()
                .filter(|value| value.is_finite() && *value > 0.0)
                .unwrap_or(1.0)
        })
        .collect::<Vec<_>>();
    weighted_center_values(coords, indices, &weights)
}

fn weighted_center_values(
    coords: &[[f32; 3]],
    indices: &[usize],
    weights: &[f64],
) -> Result<[f32; 3]> {
    let mut center = [0.0f64; 3];
    let mut total = 0.0;
    for (&idx, &weight) in indices.iter().zip(weights) {
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }
        total += weight;
        center[0] += coords[idx][0] as f64 * weight;
        center[1] += coords[idx][1] as f64 * weight;
        center[2] += coords[idx][2] as f64 * weight;
    }
    if total <= 0.0 {
        return Err(anyhow!("virtual site center has zero total weight"));
    }
    Ok([
        (center[0] / total) as f32,
        (center[1] / total) as f32,
        (center[2] / total) as f32,
    ])
}

fn require_defs(site: &VirtualSiteTerm, expected: usize) -> Result<()> {
    if site.defining_beads.len() != expected {
        return Err(anyhow!(
            "virtual site {} requires {expected} defining beads",
            site.site + 1
        ));
    }
    Ok(())
}

fn require_any_defs(site: &VirtualSiteTerm) -> Result<()> {
    if site.defining_beads.is_empty() {
        return Err(anyhow!(
            "virtual site {} requires defining beads",
            site.site + 1
        ));
    }
    Ok(())
}

fn require_params(site: &VirtualSiteTerm, expected: usize) -> Result<()> {
    if site.parameters.len() < expected {
        return Err(anyhow!(
            "virtual site {} requires {expected} parameters",
            site.site + 1
        ));
    }
    Ok(())
}

fn add(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn sub(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn scale(value: [f32; 3], factor: f32) -> [f32; 3] {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}

fn dot(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn unit(value: [f32; 3]) -> Result<[f32; 3]> {
    let norm = dot(value, value).sqrt();
    if norm <= f32::EPSILON {
        return Err(anyhow!("virtual site definition uses a zero-length vector"));
    }
    Ok(scale(value, 1.0 / norm))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn site(function: usize, params: Vec<f64>) -> VirtualSiteTerm {
        VirtualSiteTerm {
            site: 2,
            kind: "n".to_string(),
            function,
            defining_beads: vec![0, 1],
            parameters: params,
        }
    }

    #[test]
    fn virtual_sitesn_function_1_uses_center_of_geometry() {
        let mut coords = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let terms = BondedTermSet {
            virtual_sites: vec![site(1, Vec::new())],
            ..BondedTermSet::default()
        };

        apply_virtual_sites(&mut coords, &terms, None).unwrap();

        assert_eq!(coords.len(), 3);
        assert!((coords[2][0] - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn virtual_sitesn_function_2_uses_center_of_mass() {
        let mut coords = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let terms = BondedTermSet {
            virtual_sites: vec![site(2, Vec::new())],
            ..BondedTermSet::default()
        };

        apply_virtual_sites(&mut coords, &terms, Some(&[3.0, 1.0])).unwrap();

        assert!((coords[2][0] - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn virtual_sitesn_function_3_uses_mass_weighted_custom_weights() {
        let mut coords = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let terms = BondedTermSet {
            virtual_sites: vec![site(3, vec![1.0, 3.0])],
            ..BondedTermSet::default()
        };

        apply_virtual_sites(&mut coords, &terms, Some(&[2.0, 1.0])).unwrap();

        assert!((coords[2][0] - 1.2).abs() < 1.0e-6);
    }
}
