use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput, PlanRequirements};
use crate::plans::polymer::common::{histogram_centers, PolymerChains};

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuPolymer};

pub struct BondLengthDistributionPlan {
    selection: Selection,
    bins: usize,
    r_max: f32,
    counts: Vec<u64>,
    centers: Vec<f32>,
    chains: Option<PolymerChains>,
    selected_bond_pairs: Vec<usize>,
    bond_pairs: Vec<usize>,
    use_selected_input: bool,
    #[cfg(feature = "cuda")]
    gpu: Option<GpuPolymer>,
}

impl BondLengthDistributionPlan {
    pub fn new(selection: Selection, bins: usize, r_max: f32) -> Self {
        Self {
            selection,
            bins: bins.max(1),
            r_max,
            counts: vec![0; bins.max(1)],
            centers: histogram_centers(r_max, bins.max(1)),
            chains: None,
            selected_bond_pairs: Vec::new(),
            bond_pairs: Vec::new(),
            use_selected_input: true,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl Plan for BondLengthDistributionPlan {
    fn name(&self) -> &'static str {
        "polymer_bond_length"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(false, false)
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        self.use_selected_input = true;
        let chains = PolymerChains::from_selection(system, &self.selection)?;
        if chains.bond_pairs.is_empty() {
            return Err(TrajError::Mismatch("no bonds in selection".into()));
        }
        self.counts.fill(0);
        self.selected_bond_pairs =
            map_indices_to_local(&self.selection.indices, &chains.bond_pairs)?;
        self.bond_pairs = chains.bond_pairs.iter().map(|&idx| idx as usize).collect();
        #[cfg(feature = "cuda")]
        let mut gpu: Option<GpuPolymer> = None;
        #[cfg(feature = "cuda")]
        {
            if let Device::Cuda(ctx) = device {
                self.use_selected_input = false;
                gpu = Some(ctx.polymer_data(
                    &chains.offsets,
                    &chains.indices,
                    Some(&chains.bond_pairs),
                    None,
                )?);
            }
        }
        self.chains = Some(chains);
        #[cfg(feature = "cuda")]
        {
            self.gpu = gpu;
        }
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.use_selected_input {
            Some(&self.selection.indices)
        } else {
            None
        }
    }

    fn preferred_selection_hint(&self, _system: &System) -> Option<&[u32]> {
        Some(&self.selection.indices)
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        let _ = device;
        let bond_pairs = if self.use_selected_input && chunk.n_atoms == self.selection.indices.len()
        {
            self.selected_bond_pairs.as_slice()
        } else {
            self.bond_pairs.as_slice()
        };
        let bin_width = self.r_max / self.bins as f32;
        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            let mut counts = ctx.alloc_counts(self.bins)?;
            ctx.reset_counts(&mut counts)?;
            ctx.polymer_bond_hist(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                gpu,
                self.r_max,
                self.bins,
                &mut counts,
            )?;
            let local = ctx.read_counts(&counts)?;
            for (i, v) in local.into_iter().enumerate() {
                self.counts[i] += v;
            }
            return Ok(());
        }
        for frame in 0..chunk.n_frames {
            let frame_base = frame * chunk.n_atoms;
            let frame_coords = &chunk.coords[frame_base..frame_base + chunk.n_atoms];
            for pair in bond_pairs.chunks_exact(2) {
                let a = pair[0];
                let b = pair[1];
                let pa = frame_coords[a];
                let pb = frame_coords[b];
                let dx = pb[0] - pa[0];
                let dy = pb[1] - pa[1];
                let dz = pb[2] - pa[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                if r < self.r_max {
                    let bin = (r / bin_width) as usize;
                    if bin < self.bins {
                        self.counts[bin] += 1;
                    }
                }
            }
        }
        Ok(())
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        _source_selection: &[u32],
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        self.process_chunk(chunk, system, device)
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Histogram {
            centers: std::mem::take(&mut self.centers),
            counts: std::mem::take(&mut self.counts),
        })
    }
}

fn map_indices_to_local(selection: &[u32], global: &[u32]) -> TrajResult<Vec<usize>> {
    let mut map = std::collections::HashMap::<u32, usize>::with_capacity(selection.len());
    for (local, &idx) in selection.iter().enumerate() {
        map.insert(idx, local);
    }
    let mut local = Vec::with_capacity(global.len());
    for &idx in global {
        let mapped = *map.get(&idx).ok_or_else(|| {
            TrajError::Mismatch("polymer selection/index mapping mismatch for bond_length".into())
        })?;
        local.push(mapped);
    }
    Ok(local)
}
