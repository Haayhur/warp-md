use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BondStats {
    pub bead_i: usize,
    pub bead_j: usize,
    pub mean: f64,
    pub std: f64,
    pub samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AngleStats {
    pub bead_i: usize,
    pub bead_j: usize,
    pub bead_k: usize,
    pub mean_deg: f64,
    pub std_deg: f64,
    pub samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DihedralStats {
    pub bead_i: usize,
    pub bead_j: usize,
    pub bead_k: usize,
    pub bead_l: usize,
    pub mean_deg: f64,
    pub std_deg: f64,
    pub samples: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BondedStats {
    pub bonds: Vec<BondStats>,
    pub angles: Vec<AngleStats>,
    pub dihedrals: Vec<DihedralStats>,
}

pub fn calculate_bond_stats(
    traj_path: &str,
    connections: &[(usize, usize)],
) -> Result<Vec<BondStats>> {
    let n_beads = connections
        .iter()
        .flat_map(|&(i, j)| [i, j])
        .max()
        .map(|idx| idx + 1)
        .unwrap_or(0);
    Ok(calculate_bonded_stats(traj_path, n_beads, connections)?.bonds)
}

pub fn calculate_bonded_stats(
    traj_path: &str,
    n_beads: usize,
    connections: &[(usize, usize)],
) -> Result<BondedStats> {
    let mapping = crate::trajectory::BeadMapping {
        bead_names: (0..n_beads).map(|idx| format!("B{}", idx)).collect(),
        atom_indices: (0..n_beads).map(|idx| vec![idx]).collect(),
    };
    let report = crate::trajectory::map_native_trajectory(
        std::path::Path::new(traj_path),
        None,
        &mapping,
        connections,
        &crate::trajectory::NativeTrajectoryOptions::default(),
    )?;

    Ok(BondedStats {
        bonds: report.bond_stats,
        angles: report.angle_stats,
        dihedrals: report.dihedral_stats,
    })
}

pub fn calculate_bond_stats_from_frames(
    frames: &[Vec<[f32; 3]>],
    connections: &[(usize, usize)],
) -> Vec<BondStats> {
    let n_beads = connections
        .iter()
        .flat_map(|&(i, j)| [i, j])
        .max()
        .map(|idx| idx + 1)
        .unwrap_or(0);
    calculate_bonded_stats_from_frames(frames, n_beads, connections).bonds
}

pub fn calculate_bonded_stats_from_frames(
    frames: &[Vec<[f32; 3]>],
    n_beads: usize,
    connections: &[(usize, usize)],
) -> BondedStats {
    let mut bond_values: HashMap<(usize, usize), Vec<f64>> = HashMap::new();
    for &(i, j) in connections {
        let key = if i < j { (i, j) } else { (j, i) };
        bond_values.insert(key, Vec::new());
    }
    let (angles, dihedrals) = bonded_term_definitions(n_beads, connections);
    let mut angle_values: HashMap<(usize, usize, usize), Vec<f64>> = angles
        .into_iter()
        .map(|angle| (angle, Vec::new()))
        .collect();
    let mut dihedral_values: HashMap<(usize, usize, usize, usize), Vec<f64>> = dihedrals
        .into_iter()
        .map(|dihedral| (dihedral, Vec::new()))
        .collect();

    for positions in frames {
        for (&(i, j), values) in bond_values.iter_mut() {
            if i < positions.len() && j < positions.len() {
                values.push(distance(positions[i], positions[j]));
            }
        }
        for (&(i, j, k), values) in angle_values.iter_mut() {
            if i < positions.len() && j < positions.len() && k < positions.len() {
                values.push(angle_deg(positions[i], positions[j], positions[k]));
            }
        }
        for (&(i, j, k, l), values) in dihedral_values.iter_mut() {
            if i < positions.len()
                && j < positions.len()
                && k < positions.len()
                && l < positions.len()
            {
                values.push(dihedral_deg(
                    positions[i],
                    positions[j],
                    positions[k],
                    positions[l],
                ));
            }
        }
    }

    bonded_stats_from_values(bond_values, angle_values, dihedral_values)
}

pub fn bonded_term_definitions(
    n_beads: usize,
    connections: &[(usize, usize)],
) -> (
    Vec<(usize, usize, usize)>,
    Vec<(usize, usize, usize, usize)>,
) {
    let mut adjacency = vec![Vec::<usize>::new(); n_beads];
    for &(i, j) in connections {
        if i < n_beads && j < n_beads {
            adjacency[i].push(j);
            adjacency[j].push(i);
        }
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    let mut angles = Vec::new();
    for (center, neighbors) in adjacency.iter().enumerate() {
        for a in 0..neighbors.len() {
            for b in a + 1..neighbors.len() {
                angles.push((neighbors[a], center, neighbors[b]));
            }
        }
    }
    angles.sort_unstable();

    let mut seen = HashSet::new();
    let mut dihedrals = Vec::new();
    for &(j, k) in connections {
        if j >= n_beads || k >= n_beads {
            continue;
        }
        for &i in &adjacency[j] {
            if i == k {
                continue;
            }
            for &l in &adjacency[k] {
                if l == j || l == i {
                    continue;
                }
                let path = (i, j, k, l);
                let rev = (l, k, j, i);
                let key = if path < rev { path } else { rev };
                if seen.insert(key) {
                    dihedrals.push(key);
                }
            }
        }
    }
    dihedrals.sort_unstable();
    (angles, dihedrals)
}

pub fn bonded_stats_from_values(
    bond_values: HashMap<(usize, usize), Vec<f64>>,
    angle_values: HashMap<(usize, usize, usize), Vec<f64>>,
    dihedral_values: HashMap<(usize, usize, usize, usize), Vec<f64>>,
) -> BondedStats {
    BondedStats {
        bonds: stats_from_values(bond_values),
        angles: angle_stats_from_values(angle_values),
        dihedrals: dihedral_stats_from_values(dihedral_values),
    }
}

pub fn stats_from_values(bond_values: HashMap<(usize, usize), Vec<f64>>) -> Vec<BondStats> {
    let mut stats = Vec::new();
    for ((i, j), values) in bond_values {
        if values.is_empty() {
            continue;
        }
        let sum: f64 = values.iter().sum();
        let mean = sum / values.len() as f64;
        let variance: f64 =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        stats.push(BondStats {
            bead_i: i,
            bead_j: j,
            mean,
            std: variance.sqrt(),
            samples: values.len(),
        });
    }
    stats.sort_by_key(|stat| (stat.bead_i, stat.bead_j));
    stats
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f64 {
    let dx = f64::from(a[0] - b[0]);
    let dy = f64::from(a[1] - b[1]);
    let dz = f64::from(a[2] - b[2]);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

pub fn angle_deg(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f64 {
    let ba = [
        f64::from(a[0] - b[0]),
        f64::from(a[1] - b[1]),
        f64::from(a[2] - b[2]),
    ];
    let bc = [
        f64::from(c[0] - b[0]),
        f64::from(c[1] - b[1]),
        f64::from(c[2] - b[2]),
    ];
    let denom = norm(ba) * norm(bc);
    if denom <= 0.0 {
        return 0.0;
    }
    let cos = (dot(ba, bc) / denom).clamp(-1.0, 1.0);
    cos.acos().to_degrees()
}

pub fn dihedral_deg(a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) -> f64 {
    let b0 = sub3(a, b);
    let b1 = sub3(c, b);
    let b2 = sub3(d, c);
    let b1n = normalize(b1);
    let v = sub3_f64(b0, scale3(b1n, dot(b0, b1n)));
    let w = sub3_f64(b2, scale3(b1n, dot(b2, b1n)));
    let x = dot(v, w);
    let y = dot(cross(b1n, v), w);
    y.atan2(x).to_degrees()
}

fn angle_stats_from_values(
    angle_values: HashMap<(usize, usize, usize), Vec<f64>>,
) -> Vec<AngleStats> {
    let mut stats = Vec::new();
    for ((i, j, k), values) in angle_values {
        if values.is_empty() {
            continue;
        }
        let (mean, std) = linear_mean_std(&values);
        stats.push(AngleStats {
            bead_i: i,
            bead_j: j,
            bead_k: k,
            mean_deg: mean,
            std_deg: std,
            samples: values.len(),
        });
    }
    stats.sort_by_key(|stat| (stat.bead_i, stat.bead_j, stat.bead_k));
    stats
}

fn dihedral_stats_from_values(
    dihedral_values: HashMap<(usize, usize, usize, usize), Vec<f64>>,
) -> Vec<DihedralStats> {
    let mut stats = Vec::new();
    for ((i, j, k, l), values) in dihedral_values {
        if values.is_empty() {
            continue;
        }
        let (mean, std) = circular_mean_std_deg(&values);
        stats.push(DihedralStats {
            bead_i: i,
            bead_j: j,
            bead_k: k,
            bead_l: l,
            mean_deg: mean,
            std_deg: std,
            samples: values.len(),
        });
    }
    stats.sort_by_key(|stat| (stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l));
    stats
}

fn linear_mean_std(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    (mean, variance.sqrt())
}

fn circular_mean_std_deg(values: &[f64]) -> (f64, f64) {
    let inv = 1.0 / values.len() as f64;
    let sin_sum = values
        .iter()
        .map(|value| value.to_radians().sin())
        .sum::<f64>()
        * inv;
    let cos_sum = values
        .iter()
        .map(|value| value.to_radians().cos())
        .sum::<f64>()
        * inv;
    let mean = sin_sum.atan2(cos_sum).to_degrees();
    let r = (sin_sum * sin_sum + cos_sum * cos_sum)
        .sqrt()
        .clamp(1.0e-12, 1.0);
    let std = (-2.0 * r.ln()).sqrt().to_degrees();
    (mean, std)
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f64; 3] {
    [
        f64::from(a[0] - b[0]),
        f64::from(a[1] - b[1]),
        f64::from(a[2] - b[2]),
    ]
}

fn sub3_f64(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale3(v: [f64; 3], scale: f64) -> [f64; 3] {
    [v[0] * scale, v[1] * scale, v[2] * scale]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = norm(v);
    if n <= 0.0 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

fn norm(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bonded_term_definitions_find_angles_and_dihedrals() {
        let (angles, dihedrals) = bonded_term_definitions(4, &[(0, 1), (1, 2), (2, 3)]);

        assert_eq!(angles, vec![(0, 1, 2), (1, 2, 3)]);
        assert_eq!(dihedrals, vec![(0, 1, 2, 3)]);
    }

    #[test]
    fn geometry_terms_compute_expected_values() {
        let angle = angle_deg([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let dihedral = dihedral_deg(
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        );

        assert!((angle - 90.0).abs() < 1.0e-6);
        assert!(dihedral.is_finite());
    }

    #[test]
    fn bonded_stats_from_frames_include_angles_and_dihedrals() {
        let frames = vec![vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ]];
        let stats = calculate_bonded_stats_from_frames(&frames, 4, &[(0, 1), (1, 2), (2, 3)]);

        assert_eq!(stats.bonds.len(), 3);
        assert_eq!(stats.angles.len(), 2);
        assert_eq!(stats.dihedrals.len(), 1);
        assert_eq!(stats.bonds[0].samples, 1);
        assert_eq!(stats.angles[0].samples, 1);
        assert_eq!(stats.dihedrals[0].samples, 1);
    }
}
