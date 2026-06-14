use anyhow::{anyhow, Result};

use crate::build_contract::{BuildBoxSummary, BuildSystem};

pub(crate) fn validate_box_contract(system: &BuildSystem) -> Result<()> {
    if !matches!(system.box_type.as_str(), "orthorhombic" | "triclinic") {
        return Err(anyhow!("system.box_type must be orthorhombic or triclinic"));
    }
    if !matches!(
        system.pbc.as_str(),
        "xyz" | "xy" | "xz" | "yz" | "x" | "y" | "z" | "none"
    ) {
        return Err(anyhow!(
            "system.pbc must be one of xyz, xy, xz, yz, x, y, z, none"
        ));
    }
    if let Some(unit_cell) = system.unit_cell_angstrom {
        validate_unit_cell(unit_cell)?;
        if system.box_type == "orthorhombic" && !unit_cell_is_orthorhombic(unit_cell) {
            return Err(anyhow!(
                "system.unit_cell_angstrom angles must be 90 for orthorhombic boxes"
            ));
        }
    }
    if let Some(vectors) = system.box_vectors_angstrom {
        validate_box_vectors(vectors)?;
        if system.box_type == "orthorhombic" && !box_vectors_are_orthorhombic(vectors) {
            return Err(anyhow!(
                "system.box_vectors_angstrom must be axis-aligned for orthorhombic boxes"
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_unit_cell(unit_cell: [f32; 6]) -> Result<()> {
    if unit_cell[..3]
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(anyhow!(
            "system.unit_cell_angstrom lengths must be finite and > 0"
        ));
    }
    if unit_cell[3..]
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0 || *value >= 180.0)
    {
        return Err(anyhow!(
            "system.unit_cell_angstrom angles must be finite and between 0 and 180 degrees"
        ));
    }
    unit_cell_to_vectors(unit_cell)?;
    Ok(())
}

pub(crate) fn validate_box_vectors(vectors: [[f32; 3]; 3]) -> Result<()> {
    for vector in vectors {
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(anyhow!("system.box_vectors_angstrom values must be finite"));
        }
    }
    if vectors.iter().any(|vector| vector_norm(*vector) <= 0.0) {
        return Err(anyhow!(
            "system.box_vectors_angstrom vectors must have positive length"
        ));
    }
    let volume = dot(vectors[0], cross(vectors[1], vectors[2])).abs();
    if volume <= 1.0e-6 {
        return Err(anyhow!(
            "system.box_vectors_angstrom vectors must define a non-zero volume"
        ));
    }
    Ok(())
}

pub(crate) fn resolved_box_metadata(system: &BuildSystem) -> Result<BuildBoxSummary> {
    let vectors = if let Some(vectors) = system.box_vectors_angstrom {
        vectors
    } else if let Some(unit_cell) = system.unit_cell_angstrom {
        unit_cell_to_vectors(unit_cell)?
    } else {
        [
            [system.box_size_angstrom[0], 0.0, 0.0],
            [0.0, system.box_size_angstrom[1], 0.0],
            [0.0, 0.0, system.box_size_angstrom[2]],
        ]
    };
    let unit_cell = if let Some(unit_cell) = system.unit_cell_angstrom {
        unit_cell
    } else {
        vectors_to_unit_cell(vectors)?
    };
    Ok(BuildBoxSummary {
        box_type: system.box_type.clone(),
        pbc: system.pbc.clone(),
        box_size_angstrom: system.box_size_angstrom,
        unit_cell_angstrom: unit_cell,
        box_vectors_angstrom: vectors,
    })
}

pub(crate) fn unit_cell_to_vectors(unit_cell: [f32; 6]) -> Result<[[f32; 3]; 3]> {
    let [a, b, c, alpha_deg, beta_deg, gamma_deg] = unit_cell;
    let alpha = alpha_deg.to_radians();
    let beta = beta_deg.to_radians();
    let gamma = gamma_deg.to_radians();
    let sin_gamma = gamma.sin();
    if sin_gamma.abs() <= 1.0e-6 {
        return Err(anyhow!(
            "system.unit_cell_angstrom gamma angle cannot define a stable unit cell"
        ));
    }
    let ax = [a, 0.0, 0.0];
    let bx = b * gamma.cos();
    let by = b * sin_gamma;
    let cx = c * beta.cos();
    let cy = c * (alpha.cos() - beta.cos() * gamma.cos()) / sin_gamma;
    let cz2 = c * c - cx * cx - cy * cy;
    if cz2 <= 1.0e-6 {
        return Err(anyhow!(
            "system.unit_cell_angstrom angles cannot define a non-zero volume"
        ));
    }
    Ok([ax, [bx, by, 0.0], [cx, cy, cz2.sqrt()]])
}

pub(crate) fn vectors_to_unit_cell(vectors: [[f32; 3]; 3]) -> Result<[f32; 6]> {
    let a = vector_norm(vectors[0]);
    let b = vector_norm(vectors[1]);
    let c = vector_norm(vectors[2]);
    if a <= 0.0 || b <= 0.0 || c <= 0.0 {
        return Err(anyhow!("box vectors must have positive length"));
    }
    let alpha = angle_between(vectors[1], vectors[2], b, c);
    let beta = angle_between(vectors[0], vectors[2], a, c);
    let gamma = angle_between(vectors[0], vectors[1], a, b);
    Ok([a, b, c, alpha, beta, gamma])
}

fn angle_between(lhs: [f32; 3], rhs: [f32; 3], lhs_norm: f32, rhs_norm: f32) -> f32 {
    let cosine = (dot(lhs, rhs) / (lhs_norm * rhs_norm)).clamp(-1.0, 1.0);
    cosine.acos().to_degrees()
}

pub(crate) fn unit_cell_is_orthorhombic(unit_cell: [f32; 6]) -> bool {
    unit_cell[3..]
        .iter()
        .all(|angle| (*angle - 90.0).abs() <= 1.0e-4)
}

pub(crate) fn box_vectors_are_orthorhombic(vectors: [[f32; 3]; 3]) -> bool {
    vectors[0][1].abs() <= 1.0e-4
        && vectors[0][2].abs() <= 1.0e-4
        && vectors[1][0].abs() <= 1.0e-4
        && vectors[1][2].abs() <= 1.0e-4
        && vectors[2][0].abs() <= 1.0e-4
        && vectors[2][1].abs() <= 1.0e-4
}

pub(crate) fn vector_norm(vector: [f32; 3]) -> f32 {
    dot(vector, vector).sqrt()
}

pub(crate) fn dot(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2]
}

pub(crate) fn cross(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

pub(crate) fn pbc_axes(pbc: &str) -> [bool; 3] {
    if pbc == "none" {
        return [false, false, false];
    }
    [pbc.contains('x'), pbc.contains('y'), pbc.contains('z')]
}

pub(crate) fn has_periodic_axis(pbc: &str) -> bool {
    pbc_axes(pbc).iter().any(|enabled| *enabled)
}
