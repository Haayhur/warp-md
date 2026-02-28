use traj_core::error::TrajResult;
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput, PlanRequirements};
use crate::plans::polymer::common::PolymerChains;

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuPolymer};

pub struct ChainRgPlan {
    selection: Selection,
    chains: Option<PolymerChains>,
    selected_chains: Vec<Vec<usize>>,
    results: Vec<f32>,
    n_chains: usize,
    use_selected_input: bool,
    #[cfg(feature = "cuda")]
    gpu: Option<GpuPolymer>,
}

impl ChainRgPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            chains: None,
            selected_chains: Vec::new(),
            results: Vec::new(),
            n_chains: 0,
            use_selected_input: true,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    pub fn n_chains(&self) -> usize {
        self.n_chains
    }
}

impl Plan for ChainRgPlan {
    fn name(&self) -> &'static str {
        "polymer_chain_rg"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(false, false)
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.use_selected_input = true;
        self.results.clear();
        self.selected_chains.clear();
        let chains = PolymerChains::from_selection(system, &self.selection)?;
        self.n_chains = chains.n_chains();
        self.selected_chains = build_selected_chains(&self.selection.indices, &chains.chains)?;
        self.chains = Some(chains);
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                self.use_selected_input = false;
                let chains = self.chains.as_ref().unwrap();
                let gpu = ctx.polymer_data(&chains.offsets, &chains.indices, None, None)?;
                self.gpu = Some(gpu);
            }
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
        let chains = self.chains.as_ref().unwrap();
        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            let out = ctx.polymer_chain_rg(&coords, chunk.n_atoms, chunk.n_frames, gpu)?;
            self.results.extend(out);
            return Ok(());
        }
        compute_chunk_rg(chunk, &chains.chains, &mut self.results);
        Ok(())
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        _source_selection: &[u32],
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        #[cfg(feature = "cuda")]
        if matches!(_device, Device::Cuda(_)) {
            return self.process_chunk(chunk, _system, _device);
        }
        compute_chunk_rg(chunk, &self.selected_chains, &mut self.results);
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let data = std::mem::take(&mut self.results);
        let rows = if self.n_chains > 0 {
            data.len() / self.n_chains
        } else {
            0
        };
        Ok(PlanOutput::Matrix {
            data,
            rows,
            cols: self.n_chains,
        })
    }
}

fn build_selected_chains(selection: &[u32], chains: &[Vec<usize>]) -> TrajResult<Vec<Vec<usize>>> {
    let mut global_to_local =
        std::collections::HashMap::<usize, usize>::with_capacity(selection.len());
    for (local, &global) in selection.iter().enumerate() {
        global_to_local.insert(global as usize, local);
    }
    let mut out = Vec::with_capacity(chains.len());
    for chain in chains {
        let mut local_chain = Vec::with_capacity(chain.len());
        for &global_idx in chain {
            let local_idx = *global_to_local.get(&global_idx).ok_or_else(|| {
                traj_core::error::TrajError::Mismatch(
                    "polymer selection/index mapping mismatch for chain_rg".into(),
                )
            })?;
            local_chain.push(local_idx);
        }
        out.push(local_chain);
    }
    Ok(out)
}

fn compute_chunk_rg(chunk: &FrameChunk, chains: &[Vec<usize>], out: &mut Vec<f32>) {
    if chains.is_empty() {
        return;
    }
    out.reserve(chunk.n_frames.saturating_mul(chains.len()));

    if chains.len() == 1 {
        let chain = &chains[0];
        let inv_n = 1.0f64 / chain.len() as f64;
        for frame_coords in chunk
            .coords
            .chunks_exact(chunk.n_atoms)
            .take(chunk.n_frames)
        {
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            let mut sz = 0.0f64;
            for &idx in chain {
                let p = frame_coords[idx];
                sx += p[0] as f64;
                sy += p[1] as f64;
                sz += p[2] as f64;
            }
            let mx = sx * inv_n;
            let my = sy * inv_n;
            let mz = sz * inv_n;
            let mut sum = 0.0f64;
            for &idx in chain {
                let p = frame_coords[idx];
                let dx = p[0] as f64 - mx;
                let dy = p[1] as f64 - my;
                let dz = p[2] as f64 - mz;
                sum += dx * dx + dy * dy + dz * dz;
            }
            let rg2 = (sum * inv_n).max(0.0);
            out.push(rg2.sqrt() as f32);
        }
        return;
    }

    for frame_coords in chunk
        .coords
        .chunks_exact(chunk.n_atoms)
        .take(chunk.n_frames)
    {
        for chain in chains {
            let inv_n = 1.0f64 / chain.len() as f64;
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            let mut sz = 0.0f64;
            for &idx in chain {
                let p = frame_coords[idx];
                sx += p[0] as f64;
                sy += p[1] as f64;
                sz += p[2] as f64;
            }
            let mx = sx * inv_n;
            let my = sy * inv_n;
            let mz = sz * inv_n;
            let mut sum = 0.0f64;
            for &idx in chain {
                let p = frame_coords[idx];
                let dx = p[0] as f64 - mx;
                let dy = p[1] as f64 - my;
                let dz = p[2] as f64 - mz;
                sum += dx * dx + dy * dy + dz * dz;
            }
            let rg2 = (sum * inv_n).max(0.0);
            out.push(rg2.sqrt() as f32);
        }
    }
}
