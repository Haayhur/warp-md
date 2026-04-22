use super::*;

pub(super) const COULOMB_CONST: f64 = 138.935456;

#[derive(Clone, Copy)]
pub(super) struct PairOverride {
    pub(super) qprod: f64,
    pub(super) sigma: f64,
    pub(super) epsilon: f64,
}

#[derive(Clone, Copy)]
pub(super) enum GistPbc {
    None,
    Orthorhombic {
        lx: f64,
        ly: f64,
        lz: f64,
    },
    Triclinic {
        cell: [[f64; 3]; 3],
        inv: [[f64; 3]; 3],
    },
}

pub struct GistDirectPlan {
    pub(super) oxygen_indices: Vec<u32>,
    pub(super) hydrogen1_indices: Vec<u32>,
    pub(super) hydrogen2_indices: Vec<u32>,
    pub(super) orientation_valid: Vec<u8>,
    pub(super) water_offsets: Vec<usize>,
    pub(super) water_atoms: Vec<u32>,
    pub(super) solute_indices: Vec<u32>,
    pub(super) charges: Vec<f64>,
    pub(super) sigmas: Vec<f64>,
    pub(super) epsilons: Vec<f64>,
    pub(super) exceptions: HashMap<u64, PairOverride>,
    pub(super) periodic: bool,
    pub(super) cutoff: f64,
    pub(super) origin: [f64; 3],
    pub(super) dims: [usize; 3],
    pub(super) spacing: f64,
    pub(super) padding: f64,
    pub(super) orientation_bins: usize,
    pub(super) length_scale: f64,
    pub(super) auto_grid: bool,
    pub(super) frame_filter: Option<Vec<usize>>,
    pub(super) frame_filter_pos: usize,
    pub(super) max_frames: Option<usize>,
    pub(super) counts: Vec<u32>,
    pub(super) orient_counts: Vec<u32>,
    pub(super) energy_sw: Vec<f64>,
    pub(super) energy_ww: Vec<f64>,
    pub(super) direct_sw_total: f64,
    pub(super) direct_ww_total: f64,
    pub(super) record_frame_energies: bool,
    pub(super) record_pme_frame_totals: bool,
    pub(super) frame_direct_sw: Vec<f64>,
    pub(super) frame_direct_ww: Vec<f64>,
    pub(super) frame_pme_sw: Vec<f64>,
    pub(super) frame_pme_ww: Vec<f64>,
    pub(super) frame_offsets: Vec<usize>,
    pub(super) frame_cells: Vec<u32>,
    pub(super) frame_sw: Vec<f64>,
    pub(super) frame_ww: Vec<f64>,
    #[cfg(feature = "cuda")]
    pub(super) gpu: Option<GistDirectGpuState>,
    pub(super) n_frames: usize,
    pub(super) global_frame: usize,
}

#[cfg(feature = "cuda")]
pub(super) struct GistDirectGpuState {
    pub(super) ctx: GpuContext,
    pub(super) oxygen_idx: GpuBufferU32,
    pub(super) h1_idx: GpuBufferU32,
    pub(super) h2_idx: GpuBufferU32,
    pub(super) orient_valid: GpuBufferU32,
    pub(super) counts: Option<GpuBufferU32>,
    pub(super) orient_counts: Option<GpuBufferU32>,
    pub(super) n_cells: usize,
    pub(super) water_offsets: GpuBufferU32,
    pub(super) water_atoms: GpuBufferU32,
    pub(super) solute_atoms: GpuBufferU32,
    pub(super) charges: GpuBufferF32,
    pub(super) sigmas: GpuBufferF32,
    pub(super) epsilons: GpuBufferF32,
    pub(super) ex_i: GpuBufferU32,
    pub(super) ex_j: GpuBufferU32,
    pub(super) ex_qprod: GpuBufferF32,
    pub(super) ex_sigma: GpuBufferF32,
    pub(super) ex_epsilon: GpuBufferF32,
}

impl GistDirectPlan {
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    pub fn origin(&self) -> [f64; 3] {
        self.origin
    }

    pub fn orientation_bins(&self) -> usize {
        self.orientation_bins
    }

    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    pub fn energy_sw(&self) -> &[f64] {
        &self.energy_sw
    }

    pub fn energy_ww(&self) -> &[f64] {
        &self.energy_ww
    }

    pub fn direct_sw_total(&self) -> f64 {
        self.direct_sw_total
    }

    pub fn direct_ww_total(&self) -> f64 {
        self.direct_ww_total
    }

    pub fn frame_direct_sw(&self) -> &[f64] {
        &self.frame_direct_sw
    }

    pub fn frame_direct_ww(&self) -> &[f64] {
        &self.frame_direct_ww
    }

    pub fn frame_pme_sw(&self) -> &[f64] {
        &self.frame_pme_sw
    }

    pub fn frame_pme_ww(&self) -> &[f64] {
        &self.frame_pme_ww
    }

    pub fn frame_offsets(&self) -> &[usize] {
        &self.frame_offsets
    }

    pub fn frame_cells(&self) -> &[u32] {
        &self.frame_cells
    }

    pub fn frame_sw(&self) -> &[f64] {
        &self.frame_sw
    }

    pub fn frame_ww(&self) -> &[f64] {
        &self.frame_ww
    }
}
