#[derive(Clone, Copy)]
struct PairOverride {
    qprod: f64,
    sigma: f64,
    epsilon: f64,
}

#[derive(Clone, Copy)]
enum GistPbc {
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
    oxygen_indices: Vec<u32>,
    hydrogen1_indices: Vec<u32>,
    hydrogen2_indices: Vec<u32>,
    orientation_valid: Vec<u8>,
    water_offsets: Vec<usize>,
    water_atoms: Vec<u32>,
    solute_indices: Vec<u32>,
    charges: Vec<f64>,
    sigmas: Vec<f64>,
    epsilons: Vec<f64>,
    exceptions: HashMap<u64, PairOverride>,
    periodic: bool,
    cutoff: f64,
    origin: [f64; 3],
    dims: [usize; 3],
    spacing: f64,
    padding: f64,
    orientation_bins: usize,
    length_scale: f64,
    auto_grid: bool,
    frame_filter: Option<Vec<usize>>,
    frame_filter_pos: usize,
    max_frames: Option<usize>,
    counts: Vec<u32>,
    orient_counts: Vec<u32>,
    energy_sw: Vec<f64>,
    energy_ww: Vec<f64>,
    direct_sw_total: f64,
    direct_ww_total: f64,
    record_frame_energies: bool,
    record_pme_frame_totals: bool,
    frame_direct_sw: Vec<f64>,
    frame_direct_ww: Vec<f64>,
    frame_pme_sw: Vec<f64>,
    frame_pme_ww: Vec<f64>,
    frame_offsets: Vec<usize>,
    frame_cells: Vec<u32>,
    frame_sw: Vec<f64>,
    frame_ww: Vec<f64>,
    #[cfg(feature = "cuda")]
    gpu: Option<GistDirectGpuState>,
    n_frames: usize,
    global_frame: usize,
}

#[cfg(feature = "cuda")]
struct GistDirectGpuState {
    ctx: GpuContext,
    oxygen_idx: GpuBufferU32,
    h1_idx: GpuBufferU32,
    h2_idx: GpuBufferU32,
    orient_valid: GpuBufferU32,
    counts: Option<GpuBufferU32>,
    orient_counts: Option<GpuBufferU32>,
    n_cells: usize,
    water_offsets: GpuBufferU32,
    water_atoms: GpuBufferU32,
    solute_atoms: GpuBufferU32,
    charges: GpuBufferF32,
    sigmas: GpuBufferF32,
    epsilons: GpuBufferF32,
    ex_i: GpuBufferU32,
    ex_j: GpuBufferU32,
    ex_qprod: GpuBufferF32,
    ex_sigma: GpuBufferF32,
    ex_epsilon: GpuBufferF32,
}

impl GistDirectPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new_auto(
        oxygen_indices: Vec<u32>,
        hydrogen1_indices: Vec<u32>,
        hydrogen2_indices: Vec<u32>,
        orientation_valid: Vec<u8>,
        water_offsets: Vec<u32>,
        water_atoms: Vec<u32>,
        solute_indices: Vec<u32>,
        charges: Vec<f64>,
        sigmas: Vec<f64>,
        epsilons: Vec<f64>,
        exceptions: Vec<(u32, u32, f64, f64, f64)>,
        spacing: f64,
        padding: f64,
        orientation_bins: usize,
        cutoff: f64,
        periodic: bool,
    ) -> TrajResult<Self> {
        if spacing <= 0.0 {
            return Err(TrajError::Parse("gist spacing must be > 0".into()));
        }
        if padding < 0.0 {
            return Err(TrajError::Parse("gist padding must be non-negative".into()));
        }
        if cutoff <= 0.0 {
            return Err(TrajError::Parse("gist cutoff must be > 0".into()));
        }
        validate_water_vectors(
            &oxygen_indices,
            &hydrogen1_indices,
            &hydrogen2_indices,
            &orientation_valid,
        )?;
        if charges.is_empty() || sigmas.len() != charges.len() || epsilons.len() != charges.len() {
            return Err(TrajError::Mismatch(
                "gist nonbonded vectors must be non-empty and match in length".into(),
            ));
        }
        let n_waters = oxygen_indices.len();
        if water_offsets.len() != n_waters + 1 {
            return Err(TrajError::Mismatch(
                "gist water_offsets must be n_waters + 1".into(),
            ));
        }
        let mut offsets = Vec::with_capacity(water_offsets.len());
        let mut prev = 0usize;
        for (idx, off) in water_offsets.into_iter().enumerate() {
            let off = off as usize;
            if idx == 0 && off != 0 {
                return Err(TrajError::Mismatch(
                    "gist water_offsets[0] must be 0".into(),
                ));
            }
            if off < prev || off > water_atoms.len() {
                return Err(TrajError::Mismatch(
                    "gist water_offsets must be non-decreasing within bounds".into(),
                ));
            }
            offsets.push(off);
            prev = off;
        }
        if prev != water_atoms.len() {
            return Err(TrajError::Mismatch(
                "gist water_offsets end must equal water_atoms length".into(),
            ));
        }

        let n_params = charges.len();
        for &idx in oxygen_indices.iter() {
            if idx as usize >= n_params {
                return Err(TrajError::Mismatch(
                    "gist oxygen index exceeds nonbonded parameter length".into(),
                ));
            }
        }
        for &idx in hydrogen1_indices.iter() {
            if idx as usize >= n_params {
                return Err(TrajError::Mismatch(
                    "gist hydrogen1 index exceeds nonbonded parameter length".into(),
                ));
            }
        }
        for &idx in hydrogen2_indices.iter() {
            if idx as usize >= n_params {
                return Err(TrajError::Mismatch(
                    "gist hydrogen2 index exceeds nonbonded parameter length".into(),
                ));
            }
        }
        for &idx in water_atoms.iter() {
            if idx as usize >= n_params {
                return Err(TrajError::Mismatch(
                    "gist water atom index exceeds nonbonded parameter length".into(),
                ));
            }
        }
        for &idx in solute_indices.iter() {
            if idx as usize >= n_params {
                return Err(TrajError::Mismatch(
                    "gist solute atom index exceeds nonbonded parameter length".into(),
                ));
            }
        }

        let mut exception_map = HashMap::with_capacity(exceptions.len());
        for (a, b, qprod, sigma, epsilon) in exceptions.into_iter() {
            let ia = a as usize;
            let ib = b as usize;
            if ia >= n_params || ib >= n_params {
                return Err(TrajError::Mismatch(
                    "gist exception index exceeds nonbonded parameter length".into(),
                ));
            }
            exception_map.insert(
                pair_key(ia, ib),
                PairOverride {
                    qprod,
                    sigma,
                    epsilon,
                },
            );
        }

        let orientation_bins = orientation_bins.max(1);
        Ok(Self {
            oxygen_indices,
            hydrogen1_indices,
            hydrogen2_indices,
            orientation_valid,
            water_offsets: offsets,
            water_atoms,
            solute_indices,
            charges,
            sigmas,
            epsilons,
            exceptions: exception_map,
            periodic,
            cutoff,
            origin: [0.0, 0.0, 0.0],
            dims: [0, 0, 0],
            spacing,
            padding,
            orientation_bins,
            length_scale: 1.0,
            auto_grid: true,
            frame_filter: None,
            frame_filter_pos: 0,
            max_frames: None,
            counts: Vec::new(),
            orient_counts: Vec::new(),
            energy_sw: Vec::new(),
            energy_ww: Vec::new(),
            direct_sw_total: 0.0,
            direct_ww_total: 0.0,
            record_frame_energies: false,
            record_pme_frame_totals: false,
            frame_direct_sw: Vec::new(),
            frame_direct_ww: Vec::new(),
            frame_pme_sw: Vec::new(),
            frame_pme_ww: Vec::new(),
            frame_offsets: Vec::new(),
            frame_cells: Vec::new(),
            frame_sw: Vec::new(),
            frame_ww: Vec::new(),
            #[cfg(feature = "cuda")]
            gpu: None,
            n_frames: 0,
            global_frame: 0,
        })
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }

    pub fn with_max_frames(mut self, max_frames: Option<usize>) -> Self {
        self.max_frames = max_frames;
        self
    }

    pub fn with_frame_indices(mut self, frame_indices: Option<Vec<usize>>) -> Self {
        self.frame_filter = frame_indices.map(sorted_unique_indices);
        self
    }

    pub fn with_record_frame_energies(mut self, enabled: bool) -> Self {
        self.record_frame_energies = enabled;
        self
    }

    pub fn with_record_pme_frame_totals(mut self, enabled: bool) -> Self {
        self.record_pme_frame_totals = enabled;
        self
    }

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

    fn keep_frame(&mut self, abs_frame: usize) -> bool {
        keep_frame_internal(
            self.max_frames,
            self.n_frames,
            self.frame_filter.as_ref(),
            &mut self.frame_filter_pos,
            abs_frame,
        )
    }

    fn ensure_grid_for_frame(&mut self, chunk: &FrameChunk, frame: usize) -> TrajResult<()> {
        if !self.auto_grid || !self.counts.is_empty() {
            return Ok(());
        }
        let n_atoms = chunk.n_atoms;
        if n_atoms == 0 {
            return Err(TrajError::Mismatch(
                "gist grid requires at least one atom".into(),
            ));
        }
        let base = frame * n_atoms;
        let use_all = self.solute_indices.is_empty();
        let mut min_xyz = [f64::INFINITY; 3];
        let mut max_xyz = [f64::NEG_INFINITY; 3];
        if use_all {
            for atom_idx in 0..n_atoms {
                let p = chunk.coords[base + atom_idx];
                let x = p[0] as f64 * self.length_scale;
                let y = p[1] as f64 * self.length_scale;
                let z = p[2] as f64 * self.length_scale;
                min_xyz[0] = min_xyz[0].min(x);
                min_xyz[1] = min_xyz[1].min(y);
                min_xyz[2] = min_xyz[2].min(z);
                max_xyz[0] = max_xyz[0].max(x);
                max_xyz[1] = max_xyz[1].max(y);
                max_xyz[2] = max_xyz[2].max(z);
            }
        } else {
            for &idx in self.solute_indices.iter() {
                let atom_idx = idx as usize;
                if atom_idx >= n_atoms {
                    return Err(TrajError::Mismatch(
                        "gist solute selection index out of bounds".into(),
                    ));
                }
                let p = chunk.coords[base + atom_idx];
                let x = p[0] as f64 * self.length_scale;
                let y = p[1] as f64 * self.length_scale;
                let z = p[2] as f64 * self.length_scale;
                min_xyz[0] = min_xyz[0].min(x);
                min_xyz[1] = min_xyz[1].min(y);
                min_xyz[2] = min_xyz[2].min(z);
                max_xyz[0] = max_xyz[0].max(x);
                max_xyz[1] = max_xyz[1].max(y);
                max_xyz[2] = max_xyz[2].max(z);
            }
        }
        self.origin = [
            min_xyz[0] - self.padding,
            min_xyz[1] - self.padding,
            min_xyz[2] - self.padding,
        ];
        self.dims = dims_from_bounds(min_xyz, max_xyz, self.padding, self.spacing);
        let n_cells = self.dims[0] * self.dims[1] * self.dims[2];
        self.counts = vec![0u32; n_cells];
        self.orient_counts = vec![0u32; n_cells * self.orientation_bins];
        self.energy_sw = vec![0.0f64; n_cells];
        self.energy_ww = vec![0.0f64; n_cells];
        Ok(())
    }

    fn water_atom_slice(&self, water_idx: usize) -> &[u32] {
        let start = self.water_offsets[water_idx];
        let end = self.water_offsets[water_idx + 1];
        &self.water_atoms[start..end]
    }

    fn pbc_geometry(&self, chunk: &FrameChunk, frame: usize) -> TrajResult<GistPbc> {
        if !self.periodic {
            return Ok(GistPbc::None);
        }
        match chunk.box_[frame] {
            Box3::Orthorhombic { lx, ly, lz } => Ok(GistPbc::Orthorhombic {
                lx: lx as f64 * self.length_scale,
                ly: ly as f64 * self.length_scale,
                lz: lz as f64 * self.length_scale,
            }),
            Box3::Triclinic { m } => {
                let mut scaled = [0.0f32; 9];
                for (i, value) in m.into_iter().enumerate() {
                    scaled[i] = value * self.length_scale as f32;
                }
                let (cell, inv) = pbc_utils::cell_and_inv_from_box(Box3::Triclinic { m: scaled })?;
                Ok(GistPbc::Triclinic { cell, inv })
            }
            Box3::None => Err(TrajError::Mismatch(
                "gist direct with periodic=true requires box vectors".into(),
            )),
        }
    }

    fn pair_energy(
        &self,
        chunk: &FrameChunk,
        frame_base: usize,
        atom_i: usize,
        atom_j: usize,
        pbc: GistPbc,
    ) -> TrajResult<f64> {
        if atom_i == atom_j {
            return Ok(0.0);
        }
        if atom_i >= chunk.n_atoms || atom_j >= chunk.n_atoms {
            return Err(TrajError::Mismatch(
                "gist atom index out of bounds for frame".into(),
            ));
        }
        if atom_i >= self.charges.len() || atom_j >= self.charges.len() {
            return Err(TrajError::Mismatch(
                "gist atom index out of nonbonded parameter bounds".into(),
            ));
        }
        let pi = chunk.coords[frame_base + atom_i];
        let pj = chunk.coords[frame_base + atom_j];
        let mut dx = (pi[0] as f64 - pj[0] as f64) * self.length_scale;
        let mut dy = (pi[1] as f64 - pj[1] as f64) * self.length_scale;
        let mut dz = (pi[2] as f64 - pj[2] as f64) * self.length_scale;
        match pbc {
            GistPbc::None => {}
            GistPbc::Orthorhombic { lx, ly, lz } => {
                pbc_utils::apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
            }
            GistPbc::Triclinic { cell, inv } => {
                pbc_utils::apply_pbc_triclinic(&mut dx, &mut dy, &mut dz, &cell, &inv);
            }
        }
        let r2 = dx * dx + dy * dy + dz * dz;
        if r2 <= 0.0 {
            return Ok(0.0);
        }
        let r = r2.sqrt();
        if r > self.cutoff {
            return Ok(0.0);
        }

        let key = pair_key(atom_i, atom_j);
        let (qprod, sigma, epsilon) = if let Some(pair) = self.exceptions.get(&key) {
            (pair.qprod, pair.sigma, pair.epsilon)
        } else {
            let qprod = self.charges[atom_i] * self.charges[atom_j];
            let sigma = 0.5 * (self.sigmas[atom_i] + self.sigmas[atom_j]);
            let epsilon = (self.epsilons[atom_i] * self.epsilons[atom_j]).sqrt();
            (qprod, sigma, epsilon)
        };

        let mut e = 0.0;
        if epsilon != 0.0 && sigma != 0.0 {
            let sr = sigma / r;
            let sr2 = sr * sr;
            let sr6 = sr2 * sr2 * sr2;
            e += 4.0 * epsilon * (sr6 * sr6 - sr6);
        }
        if qprod != 0.0 {
            e += COULOMB_CONST * qprod / r;
        }
        Ok(e)
    }

    fn group_energy(
        &self,
        chunk: &FrameChunk,
        frame: usize,
        group_a: &[u32],
        group_b: &[u32],
        pbc: GistPbc,
    ) -> TrajResult<f64> {
        let frame_base = frame * chunk.n_atoms;
        let mut e_total = 0.0;
        for &ai in group_a.iter() {
            let ai = ai as usize;
            for &aj in group_b.iter() {
                let aj = aj as usize;
                if ai == aj {
                    continue;
                }
                e_total += self.pair_energy(chunk, frame_base, ai, aj, pbc)?;
            }
        }
        Ok(e_total)
    }
}
