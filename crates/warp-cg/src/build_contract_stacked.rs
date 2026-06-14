use super::*;

pub(super) fn expand_stacked_membranes(mut request: BuildRequest) -> Result<BuildRequest> {
    if request.stacked_membranes.is_empty() {
        return Ok(request);
    }

    let stacks = request.stacked_membranes.clone();
    for (stack_idx, stack) in stacks.iter().enumerate() {
        let expanded = expand_stacked_membrane_request(&request.system, stack_idx, stack)?;
        request.system.box_size_angstrom[2] =
            request.system.box_size_angstrom[2].max(expanded.box_z_angstrom);
        request.membranes.extend(expanded.membranes);
        if !expanded.solvent_zones.is_empty() {
            request.environment.solvent.enabled = true;
            request
                .environment
                .solvent
                .zones
                .extend(expanded.solvent_zones);
        }
    }
    Ok(request)
}

#[derive(Clone, Debug)]
struct ExpandedStackedMembranes {
    box_z_angstrom: f32,
    membranes: Vec<MembraneRequest>,
    solvent_zones: Vec<SolventZone>,
}

fn expand_stacked_membrane_request(
    system: &BuildSystem,
    stack_idx: usize,
    stack: &StackedMembranesRequest,
) -> Result<ExpandedStackedMembranes> {
    let layer_count = stack.layers.len();
    if layer_count == 0 {
        return Err(anyhow!(
            "stacked_membranes[{stack_idx}].layers cannot be empty"
        ));
    }
    if !matches!(stack.pbc.as_str(), "split" | "bottom") {
        return Err(anyhow!(
            "stacked_membranes[{stack_idx}].pbc must be split or bottom"
        ));
    }
    let distances = expand_stacked_values(
        &stack.distance_angstrom,
        layer_count,
        50.0,
        &format!("stacked_membranes[{stack_idx}].distance_angstrom"),
    )?;
    if distances
        .iter()
        .any(|value| *value <= 0.0 || !value.is_finite())
    {
        return Err(anyhow!(
            "stacked_membranes[{stack_idx}].distance_angstrom values must be finite and > 0"
        ));
    }
    let distance_types = expand_stacked_distance_types(stack, layer_count, stack_idx)?;

    let extents = stack
        .layers
        .iter()
        .enumerate()
        .map(|(layer_idx, layer)| {
            stacked_membrane_layer_extents(system, &layer.membrane)
                .map_err(|err| anyhow!("stacked_membranes[{stack_idx}].layers[{layer_idx}]: {err}"))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut membrane_centers_abs = Vec::with_capacity(layer_count);
    let mut solvent_intervals_abs = Vec::with_capacity(layer_count);
    let mut current_z = 0.0f32;
    for layer_idx in 0..layer_count {
        let distance = distances[layer_idx];
        let below_upper_extent = extents[(layer_idx + layer_count - 1) % layer_count].upper;
        let above_lower_extent = extents[layer_idx].lower;
        if layer_idx != 0 {
            current_z += below_upper_extent;
        }
        let leaflet_extent = below_upper_extent + above_lower_extent;
        let full_length = match distance_types[layer_idx].as_str() {
            "center" => {
                if distance <= leaflet_extent {
                    return Err(anyhow!(
                        "stacked_membranes[{stack_idx}].distance_angstrom[{layer_idx}] must exceed neighboring leaflet extent when distance_type is center"
                    ));
                }
                distance
            }
            "surface" => leaflet_extent + distance,
            other => {
                return Err(anyhow!(
                    "stacked_membranes[{stack_idx}].distance_type[{layer_idx}] must be surface or center, got {other}"
                ))
            }
        };
        let solvent_gap = full_length - leaflet_extent;
        let lower_spacing = solvent_gap * 0.5 + below_upper_extent;
        let upper_spacing = solvent_gap * 0.5 + above_lower_extent;
        let solvent_mid = solvent_gap * 0.5 + current_z;
        solvent_intervals_abs.push((solvent_mid - lower_spacing, solvent_mid + upper_spacing));
        current_z += upper_spacing + solvent_gap * 0.5;
        membrane_centers_abs.push(current_z);
        if layer_idx == layer_count - 1 {
            current_z += extents[layer_idx].upper;
        }
    }
    let total_height = current_z;
    let pbc_shift = if stack.pbc == "split" {
        let (first_min, first_max) = solvent_intervals_abs[0];
        (first_min + first_max) * 0.5
    } else {
        0.0
    };
    let half_height = total_height * 0.5;

    let mut membranes = Vec::with_capacity(layer_count);
    let mut solvent_zones = Vec::new();
    for (layer_idx, layer) in stack.layers.iter().enumerate() {
        let mut membrane = layer.membrane.clone();
        let stack_name = stack
            .name
            .clone()
            .unwrap_or_else(|| format!("stack_{stack_idx}"));
        membrane.name = format!("{stack_name}:{}", membrane.name);
        membrane.center_z_angstrom = membrane_centers_abs[layer_idx] - pbc_shift - half_height;
        membranes.push(membrane);

        let Some(solvent) = &layer.solvent else {
            continue;
        };
        let intervals = stacked_solvent_intervals_for_pbc(
            stack.pbc.as_str(),
            layer_idx,
            solvent_intervals_abs[layer_idx],
            pbc_shift,
            total_height,
        );
        for (interval_idx, (zmin, zmax)) in intervals.into_iter().enumerate() {
            if zmax <= zmin {
                continue;
            }
            let mut zone = solvent.clone();
            let mut box_size = zone.box_size_angstrom.unwrap_or([
                system.box_size_angstrom[0],
                system.box_size_angstrom[1],
                zmax - zmin,
            ]);
            box_size[2] = zmax - zmin;
            zone.box_size_angstrom = Some(box_size);
            let center_xy = zone.center_angstrom.unwrap_or([0.0, 0.0, 0.0]);
            zone.center_angstrom = Some([
                center_xy[0],
                center_xy[1],
                ((zmin + zmax) * 0.5) - half_height,
            ]);
            if zone.name.is_none() {
                let stack_name = stack
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("stack_{stack_idx}"));
                zone.name = Some(format!("{stack_name}:solvent:{layer_idx}:{interval_idx}"));
            }
            solvent_zones.push(zone);
        }
    }

    Ok(ExpandedStackedMembranes {
        box_z_angstrom: total_height,
        membranes,
        solvent_zones,
    })
}

#[derive(Clone, Copy, Debug)]
struct MembraneZExtents {
    upper: f32,
    lower: f32,
}

fn stacked_membrane_layer_extents(
    system: &BuildSystem,
    membrane: &MembraneRequest,
) -> Result<MembraneZExtents> {
    let mut upper = 1.32f32;
    let mut lower = 1.32f32;
    for leaflet in &membrane.leaflets {
        let mut max_offset = 0.0f32;
        for lipid in &leaflet.composition {
            let template = lookup_lipid_template(&lipid.lipid, &system.force_field);
            let beads = normalized_lipid_beads(lipid, template.as_ref())?;
            for bead in beads {
                max_offset = max_offset.max(bead.offset_angstrom[2].abs());
            }
        }
        let extent = 15.0 + max_offset;
        match leaflet.side.as_str() {
            "upper" => upper = upper.max(extent),
            "lower" => lower = lower.max(extent),
            _ => {}
        }
    }
    Ok(MembraneZExtents { upper, lower })
}

fn expand_stacked_values(
    values: &[f32],
    target_len: usize,
    fallback: f32,
    path: &str,
) -> Result<Vec<f32>> {
    match values.len() {
        0 => Ok(vec![fallback; target_len]),
        1 => Ok(vec![values[0]; target_len]),
        len if len == target_len => Ok(values.to_vec()),
        _ => Err(anyhow!(
            "{path} must contain either one value or one value per stacked layer"
        )),
    }
}

fn expand_stacked_distance_types(
    stack: &StackedMembranesRequest,
    target_len: usize,
    stack_idx: usize,
) -> Result<Vec<String>> {
    let values = match stack.distance_type.len() {
        0 => vec!["surface".to_string(); target_len],
        1 => vec![stack.distance_type[0].clone(); target_len],
        len if len == target_len => stack.distance_type.clone(),
        _ => {
            return Err(anyhow!(
                "stacked_membranes[{stack_idx}].distance_type must contain either one value or one value per stacked layer"
            ))
        }
    };
    Ok(values)
}

fn stacked_solvent_intervals_for_pbc(
    pbc: &str,
    layer_idx: usize,
    interval: (f32, f32),
    pbc_shift: f32,
    total_height: f32,
) -> Vec<(f32, f32)> {
    if pbc == "split" && layer_idx == 0 {
        let shifted_min = interval.0 - pbc_shift;
        let shifted_max = interval.1 - pbc_shift;
        return vec![
            (total_height + shifted_min, total_height),
            (0.0, shifted_max),
        ];
    }
    vec![(interval.0 - pbc_shift, interval.1 - pbc_shift)]
}
