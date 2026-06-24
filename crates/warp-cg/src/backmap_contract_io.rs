use crate::backmap::BackmapAtomMetadata;
use anyhow::Result;
use std::fs;
use std::path::Path;
use traj_core::frame::Box3;

pub(super) fn write_pdb(
    path: &Path,
    coords: &[[f64; 3]],
    metadata: &[BackmapAtomMetadata],
    vectors: Option<[[f64; 3]; 3]>,
) -> Result<()> {
    let mut text = String::new();
    if let Some(vectors) = vectors {
        let alpha = vector_angle_degrees(vectors[1], vectors[2]);
        let beta = vector_angle_degrees(vectors[0], vectors[2]);
        let gamma = vector_angle_degrees(vectors[0], vectors[1]);
        text.push_str(&format!(
            "CRYST1{:9.3}{:9.3}{:9.3}{:7.2}{:7.2}{:7.2} P 1           1\n",
            norm(vectors[0]),
            norm(vectors[1]),
            norm(vectors[2]),
            alpha,
            beta,
            gamma
        ));
    }
    for (idx, (coord, atom)) in coords.iter().zip(metadata).enumerate() {
        text.push_str(&format!(
            "ATOM  {:5} {:<4} {:>3} {}{:4}    {:8.3}{:8.3}{:8.3}  1.00  0.00          {:>2}\n",
            idx + 1,
            atom.name.chars().take(4).collect::<String>(),
            atom.residue_name.chars().take(3).collect::<String>(),
            atom.chain.chars().next().unwrap_or('A'),
            atom.residue_id,
            coord[0],
            coord[1],
            coord[2],
            atom.element.chars().take(2).collect::<String>()
        ));
    }
    text.push_str("END\n");
    fs::write(path, text)?;
    Ok(())
}

pub(super) fn write_gro(
    path: &Path,
    coords: &[[f64; 3]],
    metadata: &[BackmapAtomMetadata],
    vectors: Option<[[f64; 3]; 3]>,
) -> Result<()> {
    let mut text = format!("warp-cg backmap\n{}\n", coords.len());
    for (idx, (coord, atom)) in coords.iter().zip(metadata).enumerate() {
        text.push_str(&format!(
            "{:5}{:<5}{:>5}{:5}{:8.3}{:8.3}{:8.3}\n",
            atom.residue_id.rem_euclid(100_000),
            atom.residue_name.chars().take(5).collect::<String>(),
            atom.name.chars().take(5).collect::<String>(),
            (idx + 1) % 100_000,
            coord[0] * 0.1,
            coord[1] * 0.1,
            coord[2] * 0.1
        ));
    }
    let vectors = vectors.unwrap_or([[0.0; 3]; 3]);
    let triclinic = vectors[0][1].abs() > 1.0e-12
        || vectors[0][2].abs() > 1.0e-12
        || vectors[1][0].abs() > 1.0e-12
        || vectors[1][2].abs() > 1.0e-12
        || vectors[2][0].abs() > 1.0e-12
        || vectors[2][1].abs() > 1.0e-12;
    if triclinic {
        text.push_str(&format!(
            "{:10.5}{:10.5}{:10.5}{:10.5}{:10.5}{:10.5}{:10.5}{:10.5}{:10.5}\n",
            vectors[0][0] * 0.1,
            vectors[1][1] * 0.1,
            vectors[2][2] * 0.1,
            vectors[0][1] * 0.1,
            vectors[0][2] * 0.1,
            vectors[1][0] * 0.1,
            vectors[1][2] * 0.1,
            vectors[2][0] * 0.1,
            vectors[2][1] * 0.1
        ));
    } else {
        text.push_str(&format!(
            "{:10.5}{:10.5}{:10.5}\n",
            vectors[0][0] * 0.1,
            vectors[1][1] * 0.1,
            vectors[2][2] * 0.1
        ));
    }
    fs::write(path, text)?;
    Ok(())
}

pub(super) fn to_f32(coords: &[[f64; 3]]) -> Vec<[f32; 3]> {
    coords
        .iter()
        .map(|coord| [coord[0] as f32, coord[1] as f32, coord[2] as f32])
        .collect()
}

pub(super) fn box3(vectors: Option<[[f64; 3]; 3]>) -> Box3 {
    vectors.map_or(Box3::None, |vectors| Box3::Triclinic {
        m: [
            vectors[0][0] as f32,
            vectors[0][1] as f32,
            vectors[0][2] as f32,
            vectors[1][0] as f32,
            vectors[1][1] as f32,
            vectors[1][2] as f32,
            vectors[2][0] as f32,
            vectors[2][1] as f32,
            vectors[2][2] as f32,
        ],
    })
}

pub(super) fn box_vectors_from_box3(box_: Box3) -> Option<[[f64; 3]; 3]> {
    match box_ {
        Box3::None => None,
        Box3::Orthorhombic { lx, ly, lz } => Some([
            [f64::from(lx), 0.0, 0.0],
            [0.0, f64::from(ly), 0.0],
            [0.0, 0.0, f64::from(lz)],
        ]),
        Box3::Triclinic { m } => Some([
            [f64::from(m[0]), f64::from(m[1]), f64::from(m[2])],
            [f64::from(m[3]), f64::from(m[4]), f64::from(m[5])],
            [f64::from(m[6]), f64::from(m[7]), f64::from(m[8])],
        ]),
    }
}

fn norm(vector: [f64; 3]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn vector_angle_degrees(left: [f64; 3], right: [f64; 3]) -> f64 {
    let denominator = norm(left) * norm(right);
    if denominator <= f64::EPSILON {
        return 90.0;
    }
    let cosine = ((left[0] * right[0] + left[1] * right[1] + left[2] * right[2]) / denominator)
        .clamp(-1.0, 1.0);
    cosine.acos().to_degrees()
}
