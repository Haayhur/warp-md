use traj_core::frame::Box3;

pub(super) fn distance_vec(a: [f64; 3], b: [f64; 3], box_l: Option<[f64; 3]>) -> (f64, f64, f64) {
    let mut dx = b[0] - a[0];
    let mut dy = b[1] - a[1];
    let mut dz = b[2] - a[2];
    if let Some(b) = box_l {
        if b[0] > 0.0 {
            dx -= (dx / b[0]).round() * b[0];
        }
        if b[1] > 0.0 {
            dy -= (dy / b[1]).round() * b[1];
        }
        if b[2] > 0.0 {
            dz -= (dz / b[2]).round() * b[2];
        }
    }
    (dx, dy, dz)
}

pub(super) fn hash_cluster(cluster: &[u32]) -> u64 {
    let mut hash: u64 = 1469598103934665603;
    for &v in cluster {
        hash ^= v as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash ^ (cluster.len() as u64)
}

pub(super) fn box_lengths(box_: &Box3) -> Option<[f64; 3]> {
    match box_ {
        Box3::Orthorhombic { lx, ly, lz } => Some([*lx as f64, *ly as f64, *lz as f64]),
        Box3::Triclinic { .. } => None,
        Box3::None => None,
    }
}
