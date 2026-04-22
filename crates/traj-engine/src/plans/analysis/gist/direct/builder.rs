use super::*;

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
}
