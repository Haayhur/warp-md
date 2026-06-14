use super::*;

pub(super) fn leaflet_lipid_sequence(
    system: &BuildSystem,
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
    lipids: &[ResolvedLipid],
) -> Vec<usize> {
    let mut sequence = lipids
        .iter()
        .enumerate()
        .flat_map(|(lipid_idx, lipid)| std::iter::repeat_n(lipid_idx, lipid.count))
        .collect::<Vec<_>>();
    if let Some(seed) = placement_seed(system).filter(|_| placement_uses_random_candidates(system))
    {
        let sequence_len = sequence.len();
        shuffle_usize(
            &mut sequence,
            mix_seed(
                seed,
                "leaflet_lipid_sequence",
                format!("{}:{}", membrane.name, leaflet.name).as_bytes(),
                sequence_len,
            ),
        );
    }
    sequence
}

pub(super) fn initial_leaflet_grid(
    system: &BuildSystem,
    radii: &[f32],
    apl_angstrom2: f32,
    bounds: LayoutBounds,
    basis: Option<LayoutBasis2D>,
    periodicity: LayoutPeriodicity,
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
    proteins: &[InsertedComponent],
) -> Result<Vec<LayoutPoint>> {
    if placement_uses_random_candidates(system) {
        let seed = placement_seed(system).expect("validated seeded random candidate source");
        if let Some(points) = random_leaflet_grid(
            seed,
            radii,
            bounds,
            basis,
            periodicity,
            membrane,
            leaflet,
            proteins,
        )? {
            return Ok(points);
        }
    }
    if !membrane_has_spatial_constraints(membrane, leaflet) {
        if let Some(basis) = basis {
            return Ok(basis_leaflet_grid(radii, basis));
        }
        return Ok(rectangular_leaflet_grid(radii, apl_angstrom2, bounds));
    }
    let count = radii.len();
    if count == 0 {
        return Ok(Vec::new());
    }
    if let Some(basis) = basis {
        if let Some(points) = basis_constrained_leaflet_grid(
            radii,
            apl_angstrom2,
            basis,
            bounds,
            periodicity,
            membrane,
            leaflet,
            proteins,
        )? {
            return Ok(points);
        }
    }
    let mut spacing = apl_angstrom2.sqrt().max(0.5);
    for _ in 0..12 {
        let mut points = Vec::with_capacity(count);
        let mut y = bounds.ymin + spacing * 0.5;
        while y <= bounds.ymax && points.len() < count {
            let mut x = bounds.xmax - spacing * 0.5;
            while x >= bounds.xmin && points.len() < count {
                let point = LayoutPoint {
                    x,
                    y,
                    radius: radii[points.len()],
                };
                if membrane_allows_layout_point(
                    membrane,
                    leaflet,
                    proteins,
                    &point,
                    bounds,
                    periodicity,
                    basis,
                )? {
                    points.push(point);
                }
                x -= spacing;
            }
            y += spacing;
        }
        if points.len() == count {
            return Ok(points);
        }
        spacing *= 0.85;
    }
    Err(anyhow!(
        "leaflet {} could not place {} lipids inside requested regions",
        leaflet.name,
        count
    ))
}

fn basis_leaflet_grid(radii: &[f32], basis: LayoutBasis2D) -> Vec<LayoutPoint> {
    let count = radii.len();
    if count == 0 {
        return Vec::new();
    }
    let a_len = basis.a[0].hypot(basis.a[1]).max(1.0);
    let b_len = basis.b[0].hypot(basis.b[1]).max(1.0);
    let aspect = (a_len / b_len).max(0.1);
    let cols = ((count as f32 * aspect).sqrt().ceil() as usize).max(1);
    let rows = count.div_ceil(cols).max(1);
    let mut points = Vec::with_capacity(count);
    for row in 0..rows {
        for col in 0..cols {
            if points.len() == count {
                return points;
            }
            let fractional = [
                (col as f32 + 0.5) / cols as f32,
                (row as f32 + 0.5) / rows as f32,
            ];
            let cartesian = basis.cartesian(fractional);
            points.push(LayoutPoint {
                x: cartesian[0],
                y: cartesian[1],
                radius: radii[points.len()],
            });
        }
    }
    points
}

fn basis_constrained_leaflet_grid(
    radii: &[f32],
    apl_angstrom2: f32,
    basis: LayoutBasis2D,
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
    proteins: &[InsertedComponent],
) -> Result<Option<Vec<LayoutPoint>>> {
    let count = radii.len();
    if count == 0 {
        return Ok(Some(Vec::new()));
    }
    let a_len = basis.a[0].hypot(basis.a[1]).max(1.0);
    let b_len = basis.b[0].hypot(basis.b[1]).max(1.0);
    let mut spacing = apl_angstrom2.sqrt().max(0.5);
    for _ in 0..12 {
        let cols = (a_len / spacing).ceil().max(1.0) as usize;
        let rows = (b_len / spacing).ceil().max(1.0) as usize;
        let mut points = Vec::with_capacity(count);
        for row in 0..rows {
            for col in (0..cols).rev() {
                if points.len() == count {
                    return Ok(Some(points));
                }
                let fractional = [
                    (col as f32 + 0.5) / cols as f32,
                    (row as f32 + 0.5) / rows as f32,
                ];
                let cartesian = basis.cartesian(fractional);
                let point = LayoutPoint {
                    x: cartesian[0],
                    y: cartesian[1],
                    radius: radii[points.len()],
                };
                if point_inside_layout_domain(&point, bounds, periodicity, Some(basis))
                    && membrane_allows_layout_point(
                        membrane,
                        leaflet,
                        proteins,
                        &point,
                        bounds,
                        periodicity,
                        Some(basis),
                    )?
                {
                    points.push(point);
                }
            }
        }
        spacing *= 0.85;
    }
    Ok(None)
}

pub(super) fn random_leaflet_grid(
    seed: u64,
    radii: &[f32],
    bounds: LayoutBounds,
    basis: Option<LayoutBasis2D>,
    periodicity: LayoutPeriodicity,
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
    proteins: &[InsertedComponent],
) -> Result<Option<Vec<LayoutPoint>>> {
    if let Some(basis) = basis {
        if let Some(points) = global_basis_random_leaflet_grid(
            seed,
            radii,
            basis,
            bounds,
            periodicity,
            membrane,
            leaflet,
            proteins,
        )? {
            return Ok(Some(points));
        }
    }
    if let Some(points) = stratified_random_leaflet_grid(
        seed,
        radii,
        bounds,
        periodicity,
        membrane,
        leaflet,
        proteins,
    )? {
        return Ok(Some(points));
    }
    global_random_leaflet_grid(
        seed,
        radii,
        bounds,
        periodicity,
        membrane,
        leaflet,
        proteins,
    )
}

fn global_basis_random_leaflet_grid(
    seed: u64,
    radii: &[f32],
    basis: LayoutBasis2D,
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
    proteins: &[InsertedComponent],
) -> Result<Option<Vec<LayoutPoint>>> {
    if radii.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let mut state = mix_seed(
        seed,
        "leaflet_basis_random_candidates",
        format!("{}:{}", membrane.name, leaflet.name).as_bytes(),
        radii.len(),
    );
    let mut points = Vec::with_capacity(radii.len());
    let max_attempts = (radii.len() * 2048).max(4096).min(500_000);
    let mut attempts = 0usize;
    while points.len() < radii.len() && attempts < max_attempts {
        attempts += 1;
        let radius = radii[points.len()];
        let Some(margins) = basis_fractional_margins(basis, radius) else {
            return Ok(None);
        };
        if margins[0] >= 0.5 || margins[1] >= 0.5 {
            return Ok(None);
        }
        let fractional = [
            margins[0] + (1.0 - 2.0 * margins[0]) * seeded_unit_f32(&mut state),
            margins[1] + (1.0 - 2.0 * margins[1]) * seeded_unit_f32(&mut state),
        ];
        let cartesian = basis.cartesian(fractional);
        let point = LayoutPoint {
            x: cartesian[0],
            y: cartesian[1],
            radius,
        };
        if !membrane_allows_layout_point(
            membrane,
            leaflet,
            proteins,
            &point,
            bounds,
            periodicity,
            Some(basis),
        )? {
            continue;
        }
        points.push(point);
    }
    Ok((points.len() == radii.len()).then_some(points))
}

pub(super) fn basis_fractional_margins(basis: LayoutBasis2D, radius: f32) -> Option<[f32; 2]> {
    let area = (basis.a[0] * basis.b[1] - basis.a[1] * basis.b[0]).abs();
    let a_len = basis.a[0].hypot(basis.a[1]);
    let b_len = basis.b[0].hypot(basis.b[1]);
    if area <= 1.0e-8 || a_len <= 1.0e-8 || b_len <= 1.0e-8 {
        return None;
    }
    Some([radius / (area / b_len), radius / (area / a_len)])
}

#[derive(Clone, Copy, Debug)]
struct LeafletSpatialGroup {
    bounds: LayoutBounds,
    weight: usize,
}

fn stratified_random_leaflet_grid(
    seed: u64,
    radii: &[f32],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
    proteins: &[InsertedComponent],
) -> Result<Option<Vec<LayoutPoint>>> {
    if radii.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let groups =
        leaflet_spatial_groups(bounds, periodicity, membrane, leaflet, proteins, radii[0])?;
    if groups.len() <= 1 {
        return Ok(None);
    }
    let counts = spatial_group_counts(radii.len(), &groups);
    let mut points = Vec::with_capacity(radii.len());
    for (group_idx, (group, count)) in groups.iter().zip(counts.iter()).enumerate() {
        let mut state = mix_seed(
            seed,
            "leaflet_spatial_group",
            format!("{}:{}:{group_idx}", membrane.name, leaflet.name).as_bytes(),
            *count,
        );
        let max_attempts = (*count * 2048).max(1024).min(100_000);
        let start_len = points.len();
        let mut attempts = 0usize;
        while points.len() - start_len < *count && attempts < max_attempts {
            attempts += 1;
            let radius = radii[points.len()];
            let point = LayoutPoint {
                x: group.bounds.xmin
                    + (group.bounds.xmax - group.bounds.xmin) * seeded_unit_f32(&mut state),
                y: group.bounds.ymin
                    + (group.bounds.ymax - group.bounds.ymin) * seeded_unit_f32(&mut state),
                radius,
            };
            if !point_inside_layout_domain(&point, bounds, periodicity, None) {
                continue;
            }
            if !membrane_allows_layout_point(
                membrane,
                leaflet,
                proteins,
                &point,
                bounds,
                periodicity,
                None,
            )? {
                continue;
            }
            points.push(point);
        }
        if points.len() - start_len < *count {
            return Ok(None);
        }
    }
    Ok((points.len() == radii.len()).then_some(points))
}

fn leaflet_spatial_groups(
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
    proteins: &[InsertedComponent],
    probe_radius: f32,
) -> Result<Vec<LeafletSpatialGroup>> {
    let width = bounds.xmax - bounds.xmin;
    let height = bounds.ymax - bounds.ymin;
    if width <= 0.0 || height <= 0.0 {
        return Ok(Vec::new());
    }
    let nx = (width / 50.0).ceil().max(1.0) as usize;
    let ny = (height / 50.0).ceil().max(1.0) as usize;
    let dx = width / nx as f32;
    let dy = height / ny as f32;
    let mut groups = Vec::new();
    for iy in 0..ny {
        for ix in 0..nx {
            let group_bounds = LayoutBounds {
                xmin: bounds.xmin + ix as f32 * dx,
                xmax: if ix + 1 == nx {
                    bounds.xmax
                } else {
                    bounds.xmin + (ix + 1) as f32 * dx
                },
                ymin: bounds.ymin + iy as f32 * dy,
                ymax: if iy + 1 == ny {
                    bounds.ymax
                } else {
                    bounds.ymin + (iy + 1) as f32 * dy
                },
            };
            let weight = leaflet_spatial_group_weight(
                group_bounds,
                bounds,
                periodicity,
                membrane,
                leaflet,
                proteins,
                probe_radius,
            )?;
            if weight > 0 {
                groups.push(LeafletSpatialGroup {
                    bounds: group_bounds,
                    weight,
                });
            }
        }
    }
    Ok(groups)
}

fn leaflet_spatial_group_weight(
    group_bounds: LayoutBounds,
    layout_bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
    proteins: &[InsertedComponent],
    probe_radius: f32,
) -> Result<usize> {
    const PROBES_PER_AXIS: usize = 5;
    let mut weight = 0usize;
    for iy in 0..PROBES_PER_AXIS {
        let y = group_bounds.ymin
            + (iy as f32 + 0.5) * (group_bounds.ymax - group_bounds.ymin) / PROBES_PER_AXIS as f32;
        for ix in 0..PROBES_PER_AXIS {
            let x = group_bounds.xmin
                + (ix as f32 + 0.5) * (group_bounds.xmax - group_bounds.xmin)
                    / PROBES_PER_AXIS as f32;
            let point = LayoutPoint {
                x,
                y,
                radius: probe_radius,
            };
            if point_inside_layout_domain(&point, layout_bounds, periodicity, None)
                && membrane_allows_layout_point(
                    membrane,
                    leaflet,
                    proteins,
                    &point,
                    layout_bounds,
                    periodicity,
                    None,
                )?
            {
                weight += 1;
            }
        }
    }
    Ok(weight)
}

fn spatial_group_counts(total: usize, groups: &[LeafletSpatialGroup]) -> Vec<usize> {
    if total == 0 || groups.is_empty() {
        return vec![0; groups.len()];
    }
    let weight_sum: usize = groups.iter().map(|group| group.weight).sum();
    if weight_sum == 0 {
        return vec![0; groups.len()];
    }
    let mut counts = groups
        .iter()
        .map(|group| total * group.weight / weight_sum)
        .collect::<Vec<_>>();
    let mut assigned: usize = counts.iter().sum();
    while assigned < total {
        let assigned_now = assigned.max(1);
        let best_idx = groups
            .iter()
            .enumerate()
            .max_by(|(left_idx, left), (right_idx, right)| {
                let left_gap = left.weight as f32 / weight_sum as f32
                    - counts[*left_idx] as f32 / assigned_now as f32;
                let right_gap = right.weight as f32 / weight_sum as f32
                    - counts[*right_idx] as f32 / assigned_now as f32;
                left_gap
                    .partial_cmp(&right_gap)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.weight.cmp(&right.weight))
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        counts[best_idx] += 1;
        assigned += 1;
    }
    counts
}

fn global_random_leaflet_grid(
    seed: u64,
    radii: &[f32],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
    proteins: &[InsertedComponent],
) -> Result<Option<Vec<LayoutPoint>>> {
    if radii.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let mut state = mix_seed(
        seed,
        "leaflet_random_candidates",
        format!("{}:{}", membrane.name, leaflet.name).as_bytes(),
        radii.len(),
    );
    let mut points = Vec::with_capacity(radii.len());
    let max_attempts = (radii.len() * 2048).max(4096).min(500_000);
    let mut attempts = 0usize;
    while points.len() < radii.len() && attempts < max_attempts {
        attempts += 1;
        let radius = radii[points.len()];
        let point = LayoutPoint {
            x: bounds.xmin + (bounds.xmax - bounds.xmin) * seeded_unit_f32(&mut state),
            y: bounds.ymin + (bounds.ymax - bounds.ymin) * seeded_unit_f32(&mut state),
            radius,
        };
        if !point_inside_layout_domain(&point, bounds, periodicity, None) {
            continue;
        }
        if !membrane_allows_layout_point(
            membrane,
            leaflet,
            proteins,
            &point,
            bounds,
            periodicity,
            None,
        )? {
            continue;
        }
        points.push(point);
    }
    Ok((points.len() == radii.len()).then_some(points))
}
