pub use traj_core::spatial_hash::SpatialHash;

use crate::atom_params::AtomParams;
use crate::geom::Vec3;

pub trait SpatialHashParamsExt {
    fn overlaps_params(
        &self,
        positions: &[Vec3],
        params: &[AtomParams],
        existing: &[Vec3],
        existing_params: &[AtomParams],
    ) -> bool;

    fn overlaps_short_tol_params(
        &self,
        positions: &[Vec3],
        params: &[AtomParams],
        existing: &[Vec3],
        existing_params: &[AtomParams],
    ) -> Option<f32>;

    fn overlap_penalty_params(
        &self,
        positions: &[Vec3],
        params: &[AtomParams],
        existing: &[Vec3],
        existing_params: &[AtomParams],
    ) -> f32;
}

impl SpatialHashParamsExt for SpatialHash {
    fn overlaps_params(
        &self,
        positions: &[Vec3],
        params: &[AtomParams],
        existing: &[Vec3],
        existing_params: &[AtomParams],
    ) -> bool {
        self.overlaps_with(
            positions,
            existing,
            |i| params[i].radius,
            |j| existing_params[j].radius,
        )
    }

    fn overlaps_short_tol_params(
        &self,
        positions: &[Vec3],
        params: &[AtomParams],
        existing: &[Vec3],
        existing_params: &[AtomParams],
    ) -> Option<f32> {
        let mut penalty = 0.0f32;
        for (i, p) in positions.iter().enumerate() {
            self.for_each_neighbor(*p, |idx| {
                let q = existing[idx];
                let d = p.sub(q);
                let dist2 = d.dot(d);
                let pi = params[i];
                let pj = existing_params[idx];
                let min_r = pi.radius + pj.radius;
                let min2 = min_r * min_r;
                if dist2 < min2 {
                    penalty = f32::INFINITY;
                    return;
                }
                if !(pi.use_short || pj.use_short) {
                    return;
                }
                let short_r = pi.short_radius + pj.short_radius;
                if short_r <= 0.0 {
                    return;
                }
                let short2 = short_r * short_r;
                if dist2 < short2 {
                    let diff = dist2 - short2;
                    let scale = (pi.short_scale * pj.short_scale).sqrt();
                    let weight = pi.fscale * pj.fscale * scale;
                    penalty += diff * diff * weight;
                }
            });
            if penalty.is_infinite() {
                return Some(penalty);
            }
        }
        Some(penalty)
    }

    fn overlap_penalty_params(
        &self,
        positions: &[Vec3],
        params: &[AtomParams],
        existing: &[Vec3],
        existing_params: &[AtomParams],
    ) -> f32 {
        let mut penalty = 0.0f32;
        for (i, p) in positions.iter().enumerate() {
            self.for_each_neighbor(*p, |idx| {
                let q = existing[idx];
                let d = p.sub(q);
                let dist2 = d.dot(d);
                let min_r = params[i].radius + existing_params[idx].radius;
                let min2 = min_r * min_r;
                if dist2 < min2 {
                    let dist = dist2.max(1.0e-12).sqrt();
                    let overlap = (min_r - dist).max(0.0);
                    penalty += overlap * overlap;
                }
            });
        }
        penalty
    }
}
