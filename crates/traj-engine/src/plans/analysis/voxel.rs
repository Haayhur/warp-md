use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput};

use super::water_count::WaterCountPlan;

pub struct CountInVoxelPlan {
    inner: WaterCountPlan,
}

pub struct DensityPlan {
    inner: WaterCountPlan,
    voxel_volume: f64,
}

pub struct VolmapPlan {
    inner: WaterCountPlan,
    voxel_volume: f64,
}

impl CountInVoxelPlan {
    pub fn new(
        selection: Selection,
        center_selection: Selection,
        box_unit: [f64; 3],
        region_size: [f64; 3],
    ) -> Self {
        Self {
            inner: WaterCountPlan::new(selection, center_selection, box_unit, region_size),
        }
    }

    pub fn with_shift(mut self, shift: [f64; 3]) -> Self {
        self.inner = self.inner.with_shift(shift);
        self
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.inner = self.inner.with_length_scale(scale);
        self
    }
}

impl DensityPlan {
    pub fn new(
        selection: Selection,
        center_selection: Selection,
        box_unit: [f64; 3],
        region_size: [f64; 3],
    ) -> Self {
        let voxel_volume = box_unit[0] * box_unit[1] * box_unit[2];
        Self {
            inner: WaterCountPlan::new(selection, center_selection, box_unit, region_size),
            voxel_volume,
        }
    }

    pub fn with_shift(mut self, shift: [f64; 3]) -> Self {
        self.inner = self.inner.with_shift(shift);
        self
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.inner = self.inner.with_length_scale(scale);
        self
    }
}

impl VolmapPlan {
    pub fn new(
        selection: Selection,
        center_selection: Selection,
        box_unit: [f64; 3],
        region_size: [f64; 3],
    ) -> Self {
        let voxel_volume = box_unit[0] * box_unit[1] * box_unit[2];
        Self {
            inner: WaterCountPlan::new(selection, center_selection, box_unit, region_size),
            voxel_volume,
        }
    }

    pub fn with_shift(mut self, shift: [f64; 3]) -> Self {
        self.inner = self.inner.with_shift(shift);
        self
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.inner = self.inner.with_length_scale(scale);
        self
    }
}

impl Plan for CountInVoxelPlan {
    fn name(&self) -> &'static str {
        "count_in_voxel"
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        self.inner.init(system, device)
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        self.inner.process_chunk(chunk, system, device)
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        self.inner.finalize()
    }
}

impl Plan for DensityPlan {
    fn name(&self) -> &'static str {
        "density"
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        if self.voxel_volume <= 0.0 {
            return Err(TrajError::Mismatch(
                "density requires positive voxel volume".into(),
            ));
        }
        self.inner.init(system, device)
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        self.inner.process_chunk(chunk, system, device)
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let output = self.inner.finalize()?;
        match output {
            PlanOutput::Grid(mut grid) => {
                let inv = (1.0 / self.voxel_volume) as f32;
                for v in &mut grid.mean {
                    *v *= inv;
                }
                for v in &mut grid.std {
                    *v *= inv;
                }
                Ok(PlanOutput::Grid(grid))
            }
            _ => Err(TrajError::Mismatch("density expects grid output".into())),
        }
    }
}

impl Plan for VolmapPlan {
    fn name(&self) -> &'static str {
        "volmap"
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        if self.voxel_volume <= 0.0 {
            return Err(TrajError::Mismatch(
                "volmap requires positive voxel volume".into(),
            ));
        }
        self.inner.init(system, device)
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        self.inner.process_chunk(chunk, system, device)
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let output = self.inner.finalize()?;
        match output {
            PlanOutput::Grid(mut grid) => {
                let inv = (1.0 / self.voxel_volume) as f32;
                for v in &mut grid.mean {
                    *v *= inv;
                }
                for v in &mut grid.std {
                    *v *= inv;
                }
                Ok(PlanOutput::Grid(grid))
            }
            _ => Err(TrajError::Mismatch("volmap expects grid output".into())),
        }
    }
}
