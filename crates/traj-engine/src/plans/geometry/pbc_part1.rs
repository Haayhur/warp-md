#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuSelection};



pub struct FixImageBondsPlan {
    inner: ImagePlan,
}


impl FixImageBondsPlan {

    pub fn new(selection: Selection) -> Self {
        Self {
            inner: ImagePlan::new(selection),
        }
    }
}


impl Plan for FixImageBondsPlan {
    fn name(&self) -> &'static str {
        "fiximagedbonds"
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



pub struct RandomizeIonsPlan {
    selection: Selection,
    around: Option<Selection>,
    by: f64,
    overlap: f64,
    noimage: bool,
    max_attempts: usize,
    seed: u64,
    state: u64,
    mask: Vec<bool>,
    results: Vec<f32>,
    ion_residues: Vec<ResidueGroup>,
    solvent_residues: Vec<ResidueGroup>,
    solute_indices: Vec<u32>,

    #[cfg(feature = "cuda")]
    gpu: Option<RandomizeIonsGpuState>,
}

#[derive(Clone)]
struct ResidueGroup {
    atoms: Vec<usize>,
    anchor: usize,
}



#[cfg(feature = "cuda")]
struct RandomizeIonsGpuState {
    solute: Option<GpuSelection>,
}


impl RandomizeIonsPlan {

    pub fn new(selection: Selection, seed: u64) -> Self {
        Self {
            selection,
            around: None,
            by: 0.0,
            overlap: 0.0,
            noimage: false,
            max_attempts: 1000,
            seed,
            state: seed,
            mask: Vec::new(),
            results: Vec::new(),
            ion_residues: Vec::new(),
            solvent_residues: Vec::new(),
            solute_indices: Vec::new(),

            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }


    pub fn with_around(
        mut self,
        around: Option<Selection>,
        by: f64,
        overlap: f64,
        noimage: bool,
    ) -> Self {
        self.around = around;
        self.by = by.max(0.0);
        self.overlap = overlap.max(0.0);
        self.noimage = noimage;
        self
    }


    pub fn with_max_attempts(mut self, max_attempts: usize) -> Self {
        self.max_attempts = max_attempts.max(1);
        self
    }
}

fn min_distance_to_indices(
    point: [f64; 3],
    coords: &[[f64; 3]],
    indices: &[u32],
    cell_inv: Option<&([[f64; 3]; 3], [[f64; 3]; 3])>,
) -> f64 {
    if indices.is_empty() {
        return f64::INFINITY;
    }
    let mut min_val = f64::INFINITY;
    for &idx in indices.iter() {
        let p = coords[idx as usize];
        let mut dx = point[0] - p[0];
        let mut dy = point[1] - p[1];
        let mut dz = point[2] - p[2];
        if let Some((cell, inv)) = cell_inv {
            apply_pbc_triclinic(&mut dx, &mut dy, &mut dz, cell, inv);
        }
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if dist < min_val {
            min_val = dist;
        }
    }
    min_val
}


impl Plan for RandomizeIonsPlan {
    fn name(&self) -> &'static str {
        "randomize_ions"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.state = self.seed;
        self.mask = vec![false; system.n_atoms()];
        for &idx in self.selection.indices.iter() {
            if let Some(slot) = self.mask.get_mut(idx as usize) {
                *slot = true;
            }
        }
        let mut residue_atoms: HashMap<i32, Vec<usize>> = HashMap::new();
        for (idx, resid) in system.atoms.resid.iter().enumerate() {
            residue_atoms.entry(*resid).or_default().push(idx);
        }
        let mut ion_anchor: HashMap<i32, usize> = HashMap::new();
        for &idx in self.selection.indices.iter() {
            let resid = system.atoms.resid[idx as usize];
            ion_anchor.entry(resid).or_insert(idx as usize);
        }
        let mut ion_resids: Vec<i32> = ion_anchor.keys().copied().collect();
        ion_resids.sort_unstable();
        let mut ion_set: HashSet<i32> = HashSet::with_capacity(ion_resids.len());
        self.ion_residues.clear();
        for resid in ion_resids.iter().copied() {
            if let Some(atoms) = residue_atoms.get(&resid) {
                let anchor = ion_anchor.get(&resid).copied().unwrap_or(atoms[0]);
                self.ion_residues.push(ResidueGroup {
                    atoms: atoms.clone(),
                    anchor,
                });
                ion_set.insert(resid);
            }
        }
        self.solute_indices.clear();
        let mut solute_resids: HashSet<i32> = HashSet::new();
        if let Some(around) = &self.around {
            for &idx in around.indices.iter() {
                if !self.mask.get(idx as usize).copied().unwrap_or(false) {
                    self.solute_indices.push(idx);
                    let resid = system.atoms.resid[idx as usize];
                    solute_resids.insert(resid);
                }
            }
        }
        self.solvent_residues.clear();
        let mut solvent_resids: Vec<i32> = residue_atoms.keys().copied().collect();
        solvent_resids.sort_unstable();
        for resid in solvent_resids {
            if ion_set.contains(&resid) || solute_resids.contains(&resid) {
                continue;
            }
            if let Some(atoms) = residue_atoms.get(&resid) {
                self.solvent_residues.push(ResidueGroup {
                    anchor: atoms[0],
                    atoms: atoms.clone(),
                });
            }
        }

        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let solute = if !self.solute_indices.is_empty() {
                    Some(ctx.selection(&self.solute_indices, None)?)
                } else {
                    None
                };
                self.gpu = Some(RandomizeIonsGpuState { solute });
            }
        }
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let n_atoms = chunk.n_atoms;
        if n_atoms == 0 {
            return Ok(());
        }
        if self.ion_residues.is_empty() {
            for frame in 0..chunk.n_frames {
                for atom in 0..n_atoms {
                    let p = chunk.coords[frame * n_atoms + atom];
                    self.results.extend_from_slice(&[p[0], p[1], p[2]]);
                }
            }
            return Ok(());
        }


        #[cfg(feature = "cuda")]
        let mut gpu_min: Option<Vec<f32>> = None;

        #[cfg(not(feature = "cuda"))]
        let gpu_min: Option<Vec<f32>> = None;

        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
            if self.by > 0.0
                && !self.solute_indices.is_empty()
                && !self.solvent_residues.is_empty()
            {
                if let Some(solute) = &gpu.solute {
                    let coords = convert_coords(&chunk.coords);
                    let n_solvent = self.solvent_residues.len();
                    let mut points = Vec::with_capacity(chunk.n_frames * n_solvent);
                    for frame in 0..chunk.n_frames {
                        for res in self.solvent_residues.iter() {
                            let p = chunk.coords[frame * n_atoms + res.anchor];
                            points.push(traj_gpu::Float4 {
                                x: p[0],
                                y: p[1],
                                z: p[2],
                                w: 0.0,
                            });
                        }
                    }
                    let (cell, inv) = if self.noimage {
                        (
                            vec![traj_gpu::Float4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }; chunk.n_frames * 3],
                            vec![traj_gpu::Float4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 }; chunk.n_frames * 3],
                        )
                    } else {
                        chunk_cell_mats(chunk)?
                    };
                    let mins = ctx.min_distance_points(
                        &coords,
                        n_atoms,
                        chunk.n_frames,
                        solute,
                        &points,
                        &cell,
                        &inv,
                        !self.noimage,
                    )?;
                    gpu_min = Some(mins);
                }
            }
        }

        let by = self.by;
        let overlap = self.overlap;
        let use_image = !self.noimage;

        for frame in 0..chunk.n_frames {
            let mut coords_frame = vec![[0.0f64; 3]; n_atoms];
            for atom in 0..n_atoms {
                let p = chunk.coords[frame * n_atoms + atom];
                coords_frame[atom] = [p[0] as f64, p[1] as f64, p[2] as f64];
            }

            let cell_inv = if use_image {
                Some(cell_and_inv_from_box(chunk.box_[frame])?)
            } else {
                None
            };

            let mut eligible: Vec<usize> = Vec::new();
            let n_solvent = self.solvent_residues.len();
            if n_solvent == 0 {
                return Err(TrajError::Mismatch(
                    "randomize_ions requires solvent residues".into(),
                ));
            }
            for (i, res) in self.solvent_residues.iter().enumerate() {
                if by > 0.0 && !self.solute_indices.is_empty() {
                    let min_dist = if let Some(mins) = &gpu_min {
                        mins[frame * n_solvent + i] as f64
                    } else {
                        let point = coords_frame[res.anchor];
                        min_distance_to_indices(point, &coords_frame, &self.solute_indices, cell_inv.as_ref())
                    };
                    if min_dist >= by {
                        eligible.push(i);
                    }
                } else {
                    eligible.push(i);
                }
            }
            if eligible.is_empty() {
                return Err(TrajError::Mismatch(
                    "randomize_ions: no solvent residues satisfy around/by constraints".into(),
                ));
            }

            let mut placed: Vec<[f64; 3]> = Vec::new();
            for ion in self.ion_residues.iter() {
                let ion_pos = coords_frame[ion.anchor];
                let mut attempts = 0usize;
                let mut placed_ok = false;
                while attempts < self.max_attempts {
                    if eligible.is_empty() {
                        break;
                    }
                    let pick = ((next_f64(&mut self.state) * eligible.len() as f64) as usize)
                        .min(eligible.len() - 1);
                    let solvent_idx = eligible[pick];
                    let solvent = &self.solvent_residues[solvent_idx];
                    let sol_pos = coords_frame[solvent.anchor];
                    let mut ok = true;
                    if overlap > 0.0 {
                        for other in placed.iter() {
                            let mut dx = sol_pos[0] - other[0];
                            let mut dy = sol_pos[1] - other[1];
                            let mut dz = sol_pos[2] - other[2];
                            if let Some((cell, inv)) = cell_inv.as_ref() {
                                apply_pbc_triclinic(&mut dx, &mut dy, &mut dz, cell, inv);
                            }
                            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                            if dist < overlap {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if ok {
                        let dx = sol_pos[0] - ion_pos[0];
                        let dy = sol_pos[1] - ion_pos[1];
                        let dz = sol_pos[2] - ion_pos[2];
                        for &atom in ion.atoms.iter() {
                            coords_frame[atom][0] += dx;
                            coords_frame[atom][1] += dy;
                            coords_frame[atom][2] += dz;
                        }
                        for &atom in solvent.atoms.iter() {
                            coords_frame[atom][0] -= dx;
                            coords_frame[atom][1] -= dy;
                            coords_frame[atom][2] -= dz;
                        }
                        eligible.swap_remove(pick);
                        placed.push(sol_pos);
                        placed_ok = true;
                        break;
                    }
                    attempts += 1;
                }
                if !placed_ok {
                    return Err(TrajError::Mismatch(
                        "randomize_ions: failed to place ion within max_attempts".into(),
                    ));
                }
            }

            for atom in 0..n_atoms {
                let p = coords_frame[atom];
                self.results.push(p[0] as f32);
                self.results.push(p[1] as f32);
                self.results.push(p[2] as f32);
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}



pub struct ReplicateCellPlan {
    selection: Selection,
    repeats: [usize; 3],
    results: Vec<f32>,
    frames: usize,

    #[cfg(feature = "cuda")]
    gpu: Option<GpuSelection>,
}


impl ReplicateCellPlan {

    pub fn new(selection: Selection, repeats: [usize; 3]) -> Self {
        Self {
            selection,
            repeats,
            results: Vec::new(),
            frames: 0,

            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}


impl Plan for ReplicateCellPlan {
    fn name(&self) -> &'static str {
        "replicate_cell"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.frames = 0;

        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let selection = ctx.selection(&self.selection.indices, None)?;
                self.gpu = Some(selection);
            }
        }
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        if self.repeats[0] == 0 || self.repeats[1] == 0 || self.repeats[2] == 0 {
            return Ok(());
        }

        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            let (cell, _inv) = chunk_cell_mats(chunk)?;
            let out = ctx.replicate_cell(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                gpu,
                &cell,
                self.repeats,
            )?;
            self.results.extend(out);
            self.frames += chunk.n_frames;
            return Ok(());
        }
        let n_atoms = chunk.n_atoms;
        let sel = &self.selection.indices;
        let reps = self.repeats;
        for frame in 0..chunk.n_frames {
            let (ax, ay, az, bx, by, bz, cx, cy, cz) = match chunk.box_[frame] {
                Box3::Orthorhombic { lx, ly, lz } => (lx, 0.0, 0.0, 0.0, ly, 0.0, 0.0, 0.0, lz),
                Box3::Triclinic { m } => (m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]),
                Box3::None => {
                    return Err(TrajError::Mismatch(
                        "replicate_cell requires box vectors".into(),
                    ))
                }
            };
            for ix in 0..reps[0] {
                for iy in 0..reps[1] {
                    for iz in 0..reps[2] {
                        let sx = ix as f64 * ax as f64
                            + iy as f64 * bx as f64
                            + iz as f64 * cx as f64;
                        let sy = ix as f64 * ay as f64
                            + iy as f64 * by as f64
                            + iz as f64 * cy as f64;
                        let sz = ix as f64 * az as f64
                            + iy as f64 * bz as f64
                            + iz as f64 * cz as f64;
                        for &idx in sel.iter() {
                            let p = chunk.coords[frame * n_atoms + idx as usize];
                            self.results.push((p[0] as f64 + sx) as f32);
                            self.results.push((p[1] as f64 + sy) as f32);
                            self.results.push((p[2] as f64 + sz) as f32);
                        }
                    }
                }
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}
