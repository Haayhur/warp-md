use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, PersistenceOutput, Plan, PlanOutput, PlanRequirements};
use crate::plans::polymer::common::PolymerChains;

pub struct PersistenceLengthPlan {
    selection: Selection,
    chains: Option<PolymerChains>,
    raw_autocorr: Vec<f32>,
    lb_sum: f64,
    lb_count: usize,
    n_chains: usize,
    chain_len: usize,
    frames: usize,
}

impl PersistenceLengthPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            chains: None,
            raw_autocorr: Vec::new(),
            lb_sum: 0.0,
            lb_count: 0,
            n_chains: 0,
            chain_len: 0,
            frames: 0,
        }
    }
}

impl Plan for PersistenceLengthPlan {
    fn name(&self) -> &'static str {
        "polymer_persistence_length"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(false, false)
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        let chains = PolymerChains::from_selection(system, &self.selection)?;
        let chain_len = chains.ensure_equal_length()?;
        self.n_chains = chains.n_chains();
        self.chain_len = chain_len;
        self.raw_autocorr = vec![0.0; chain_len - 1];
        self.lb_sum = 0.0;
        self.lb_count = 0;
        self.frames = 0;
        self.chains = Some(chains);
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let chains = self.chains.as_ref().unwrap();
        for frame in 0..chunk.n_frames {
            for chain in &chains.chains {
                let mut vecs = Vec::with_capacity(chain.len() - 1);
                for pair in chain.windows(2) {
                    let a = chunk.coords[frame * chunk.n_atoms + pair[0]];
                    let b = chunk.coords[frame * chunk.n_atoms + pair[1]];
                    let dx = b[0] - a[0];
                    let dy = b[1] - a[1];
                    let dz = b[2] - a[2];
                    let len = (dx * dx + dy * dy + dz * dz).sqrt();
                    if len > 0.0 {
                        vecs.push([dx / len, dy / len, dz / len]);
                        self.lb_sum += len as f64;
                        self.lb_count += 1;
                    } else {
                        vecs.push([0.0, 0.0, 0.0]);
                    }
                }
                let n_bonds = vecs.len();
                for i in 0..n_bonds {
                    let vi = vecs[i];
                    for j in i..n_bonds {
                        let vj = vecs[j];
                        let dot = vi[0] * vj[0] + vi[1] * vj[1] + vi[2] * vj[2];
                        let offset = j - i;
                        self.raw_autocorr[offset] += dot;
                    }
                }
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let n_bonds = self.chain_len - 1;
        if n_bonds == 0 {
            return Err(TrajError::Mismatch("chain length too short".into()));
        }
        let mut norm = Vec::with_capacity(n_bonds);
        for i in 0..n_bonds {
            norm.push((n_bonds - i) as f32 * self.n_chains as f32 * self.frames as f32);
        }
        let mut autocorr = Vec::with_capacity(n_bonds);
        for i in 0..n_bonds {
            let n = norm[i];
            autocorr.push(if n > 0.0 {
                self.raw_autocorr[i] / n
            } else {
                0.0
            });
        }
        let lb = if self.lb_count > 0 {
            (self.lb_sum / self.lb_count as f64) as f32
        } else {
            0.0
        };
        let mut x = Vec::with_capacity(n_bonds);
        for i in 0..n_bonds {
            x.push(lb * i as f32);
        }
        let lp = fit_exponential_decay(&x, &autocorr).unwrap_or(0.0);
        let mut fit = Vec::with_capacity(n_bonds);
        for i in 0..n_bonds {
            if lp > 0.0 {
                fit.push((-x[i] / lp).exp());
            } else {
                fit.push(0.0);
            }
        }
        let kuhn = if lp > 0.0 { 2.0 * lp } else { 0.0 };
        Ok(PlanOutput::Persistence(PersistenceOutput {
            bond_autocorrelation: autocorr,
            lb,
            lp,
            fit,
            kuhn_length: kuhn,
        }))
    }
}

fn fit_exponential_decay(x: &[f32], y: &[f32]) -> Option<f32> {
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xx = 0.0f64;
    let mut sum_xy = 0.0f64;
    let mut n = 0.0f64;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        if yi <= 0.0 {
            continue;
        }
        let ly = (yi as f64).ln();
        let xf = xi as f64;
        sum_x += xf;
        sum_y += ly;
        sum_xx += xf * xf;
        sum_xy += xf * ly;
        n += 1.0;
    }
    if n < 2.0 {
        return None;
    }
    let denom = n * sum_xx - sum_x * sum_x;
    if denom == 0.0 {
        return None;
    }
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    if slope >= 0.0 {
        return None;
    }
    let lp = (-1.0 / slope) as f32;
    Some(lp)
}
