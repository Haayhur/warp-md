use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::correlators::LagMode;
use crate::executor::{CurrentOutput, Device, Plan, PlanOutput};
use crate::plans::analysis::grouping::GroupBy;
use crate::plans::analysis::msd::{DtDecimation, FrameDecimation, TimeBinning};

use super::{ConductivityPlan, DielectricPlan};

pub struct CurrentPlan {
    conductivity: ConductivityPlan,
    dielectric: DielectricPlan,
}

impl CurrentPlan {
    pub fn new(
        selection: Selection,
        group_by: GroupBy,
        charges: Vec<f64>,
        temperature: f64,
    ) -> Self {
        Self {
            conductivity: ConductivityPlan::new(
                selection.clone(),
                group_by,
                charges.clone(),
                temperature,
            ),
            dielectric: DielectricPlan::new(selection, group_by, charges)
                .with_temperature(temperature),
        }
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.conductivity = self.conductivity.with_length_scale(scale);
        self.dielectric = self.dielectric.with_length_scale(scale);
        self
    }

    pub fn with_group_types(mut self, types: Vec<usize>) -> Self {
        self.conductivity = self.conductivity.with_group_types(types.clone());
        self.dielectric = self.dielectric.with_group_types(types);
        self
    }

    pub fn with_make_whole(mut self, make_whole: bool) -> Self {
        self.dielectric = self.dielectric.with_make_whole(make_whole);
        self
    }

    pub fn with_lag_mode(mut self, mode: LagMode) -> Self {
        self.conductivity = self.conductivity.with_lag_mode(mode);
        self
    }

    pub fn with_max_lag(mut self, max_lag: usize) -> Self {
        self.conductivity = self.conductivity.with_max_lag(max_lag);
        self
    }

    pub fn with_memory_budget_bytes(mut self, budget: usize) -> Self {
        self.conductivity = self.conductivity.with_memory_budget_bytes(budget);
        self
    }

    pub fn with_multi_tau_m(mut self, m: usize) -> Self {
        self.conductivity = self.conductivity.with_multi_tau_m(m);
        self
    }

    pub fn with_multi_tau_levels(mut self, levels: usize) -> Self {
        self.conductivity = self.conductivity.with_multi_tau_levels(levels);
        self
    }

    pub fn with_frame_decimation(mut self, dec: FrameDecimation) -> Self {
        self.conductivity = self.conductivity.with_frame_decimation(dec);
        self
    }

    pub fn with_dt_decimation(mut self, dec: DtDecimation) -> Self {
        self.conductivity = self.conductivity.with_dt_decimation(dec);
        self
    }

    pub fn with_time_binning(mut self, bin: TimeBinning) -> Self {
        self.conductivity = self.conductivity.with_time_binning(bin);
        self
    }
}

impl Plan for CurrentPlan {
    fn name(&self) -> &'static str {
        "current"
    }

    fn set_frames_hint(&mut self, n_frames: Option<usize>) {
        self.conductivity.set_frames_hint(n_frames);
        self.dielectric.set_frames_hint(n_frames);
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        self.conductivity.init(system, device)?;
        self.dielectric.init(system, device)?;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        self.conductivity.process_chunk(chunk, system, device)?;
        self.dielectric.process_chunk(chunk, system, device)?;
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let (conductivity_time, conductivity, conductivity_rows, conductivity_cols) =
            match self.conductivity.finalize()? {
                PlanOutput::TimeSeries {
                    time,
                    data,
                    rows,
                    cols,
                } => (time, data, rows, cols),
                _ => {
                    return Err(TrajError::Mismatch(
                        "current expects conductivity timeseries output".into(),
                    ))
                }
            };
        let dielectric = match self.dielectric.finalize()? {
            PlanOutput::Dielectric(output) => output,
            _ => {
                return Err(TrajError::Mismatch(
                    "current expects dielectric output".into(),
                ))
            }
        };

        let conductivity_static = if conductivity_cols == 0 {
            None
        } else {
            conductivity
                .chunks(conductivity_cols)
                .rev()
                .find_map(|row| row.last().copied())
        };

        Ok(PlanOutput::Current(CurrentOutput {
            conductivity_time,
            conductivity,
            conductivity_rows,
            conductivity_cols,
            frame_time: dielectric.time,
            md_sq: dielectric.rot_sq,
            mj_sq: dielectric.trans_sq,
            md_mj: dielectric.rot_trans,
            dielectric_rot: dielectric.dielectric_rot,
            dielectric_total: dielectric.dielectric_total,
            mu_avg: dielectric.mu_avg,
            conductivity_static,
        }))
    }
}
