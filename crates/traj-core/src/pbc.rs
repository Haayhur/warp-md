use crate::error::{TrajError, TrajResult};
use crate::geom::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct PbcBox {
    pub min: Vec3,
    pub max: Vec3,
    pub length: Vec3,
}

impl PbcBox {
    pub fn from_size(size: [f32; 3]) -> TrajResult<Self> {
        if size.iter().any(|&v| v <= 0.0) {
            return Err(TrajError::Invalid("pbc box size must be positive".into()));
        }
        let min = Vec3::new(0.0, 0.0, 0.0);
        let max = Vec3::from_array(size);
        Ok(Self {
            min,
            max,
            length: max.sub(min),
        })
    }

    pub fn from_bounds(min: [f32; 3], max: [f32; 3]) -> TrajResult<Self> {
        let min_v = Vec3::from_array(min);
        let max_v = Vec3::from_array(max);
        let length = max_v.sub(min_v);
        if length.x <= 0.0 || length.y <= 0.0 || length.z <= 0.0 {
            return Err(TrajError::Invalid("pbc bounds must have max > min".into()));
        }
        Ok(Self {
            min: min_v,
            max: max_v,
            length,
        })
    }

    pub fn wrap(self, point: Vec3) -> Vec3 {
        let lx = self.length.x;
        let ly = self.length.y;
        let lz = self.length.z;
        Vec3::new(
            (point.x - self.min.x).rem_euclid(lx) + self.min.x,
            (point.y - self.min.y).rem_euclid(ly) + self.min.y,
            (point.z - self.min.z).rem_euclid(lz) + self.min.z,
        )
    }

    pub fn delta(self, a: Vec3, b: Vec3) -> Vec3 {
        let mut dx = a.x - b.x;
        let mut dy = a.y - b.y;
        let mut dz = a.z - b.z;
        let lx = self.length.x;
        let ly = self.length.y;
        let lz = self.length.z;
        if lx > 0.0 {
            dx -= (dx / lx).round() * lx;
        }
        if ly > 0.0 {
            dy -= (dy / ly).round() * ly;
        }
        if lz > 0.0 {
            dz -= (dz / lz).round() * lz;
        }
        Vec3::new(dx, dy, dz)
    }
}
