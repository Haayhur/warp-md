use crate::geom::Vec3;

#[derive(Clone, Copy, Debug)]
pub(crate) struct PlacementRecord {
    pub(crate) center: Vec3,
    pub(crate) euler: [f32; 3],
}

impl PlacementRecord {
    pub(crate) fn new(center: Vec3, euler: [f32; 3]) -> Self {
        Self { center, euler }
    }
}

impl Default for PlacementRecord {
    fn default() -> Self {
        Self {
            center: Vec3::default(),
            euler: [0.0, 0.0, 0.0],
        }
    }
}
