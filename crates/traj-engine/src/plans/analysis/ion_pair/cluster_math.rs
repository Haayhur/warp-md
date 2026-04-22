use traj_core::pbc_math::minimum_image_delta;

pub(super) fn distance_vec(a: [f64; 3], b: [f64; 3], box_l: Option<[f64; 3]>) -> (f64, f64, f64) {
    let [dx, dy, dz] = minimum_image_delta(a, b, box_l);
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
