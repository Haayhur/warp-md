#[cfg(feature = "cuda")]
struct RmsdGpuState {
    selection: GpuSelection,
    reference: GpuReference,
}

impl RmsdPlan {
    pub fn new(selection: Selection, reference_mode: ReferenceMode, align: bool) -> Self {
        Self {
            selection,
            selection_usize: Vec::new(),
            dense_selection_usize: Vec::new(),
            use_selected_input: false,
            align,
            reference_mode,
            reference: None,
            results: Vec::new(),

            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl SymmRmsdPlan {
    pub fn new(selection: Selection, reference_mode: ReferenceMode, align: bool) -> Self {
        Self {
            inner: RmsdPlan::new(selection, reference_mode, align),
        }
    }
}

impl PairwiseRmsdPlan {
    pub fn new(selection: Selection, metric: PairwiseMetric, pbc: PbcMode) -> Self {
        Self {
            selection,
            metric,
            pbc,
            use_selected_input: false,
            frames: Vec::new(),
            boxes: Vec::new(),
        }
    }
}

impl DistanceRmsdPlan {
    pub fn new(selection: Selection, reference_mode: ReferenceMode, pbc: PbcMode) -> Self {
        Self {
            selection,
            reference_mode,
            pbc,
            reference_dists: None,
            results: Vec::new(),
        }
    }
}

fn process_cpu_chunk_out(
    chunk: &FrameChunk,
    n_atoms: usize,
    reference: &[[f32; 4]],
    selection: &[usize],
    align: bool,
    out: &mut Vec<f32>,
) {
    for frame in 0..chunk.n_frames {
        let frame_coords = &chunk.coords[frame * n_atoms..(frame + 1) * n_atoms];
        let rmsd = if align {
            rmsd_aligned_selected(frame_coords, reference, selection)
        } else {
            rmsd_raw_selected(frame_coords, reference, selection)
        };
        out.push(rmsd);
    }
}

impl Plan for RmsdPlan {
    fn name(&self) -> &'static str {
        "rmsd"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(false, false)
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.selection_usize.clear();
        self.dense_selection_usize.clear();
        self.use_selected_input = true;
        self.selection_usize
            .extend(self.selection.indices.iter().map(|&idx| idx as usize));
        self.dense_selection_usize
            .extend(0..self.selection_usize.len());
        let _ = device;

        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
        }
        match self.reference_mode {
            ReferenceMode::Topology => {
                let positions0 = system
                    .positions0
                    .as_ref()
                    .ok_or_else(|| TrajError::Mismatch("no topology reference coords".into()))?;
                let mut reference = Vec::with_capacity(self.selection_usize.len());
                for &idx in self.selection_usize.iter() {
                    reference.push(positions0[idx]);
                }
                self.reference = Some(reference);
            }
            ReferenceMode::Frame0 => {
                self.reference = None;
            }
        }

        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(reference)) = (device, self.reference.as_ref()) {
            let dense_selection: Vec<u32> =
                (0..self.selection.indices.len()).map(|i| i as u32).collect();
            let selection = if self.use_selected_input {
                ctx.selection(&dense_selection, None)?
            } else {
                ctx.selection(&self.selection.indices, None)?
            };
            let reference_gpu = ctx.reference(&convert_coords(reference))?;
            self.gpu = Some(RmsdGpuState {
                selection,
                reference: reference_gpu,
            });
        }
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.use_selected_input {
            Some(self.selection.indices.as_slice())
        } else {
            None
        }
    }

    fn preferred_selection_hint(&self, _system: &System) -> Option<&[u32]> {
        Some(self.selection.indices.as_slice())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        let _ = device;
        let n_atoms = chunk.n_atoms;
        if self.reference.is_none() && matches!(self.reference_mode, ReferenceMode::Frame0) {
            if chunk.n_frames == 0 {
                return Ok(());
            }
            let mut reference = Vec::with_capacity(self.selection_usize.len());
            for &idx in self.selection_usize.iter() {
                reference.push(chunk.coords[idx]);
            }
            self.reference = Some(reference);

            #[cfg(feature = "cuda")]
            if let Device::Cuda(ctx) = device {
                let dense_selection: Vec<u32> =
                    (0..self.selection.indices.len()).map(|i| i as u32).collect();
                let selection = if self.use_selected_input {
                    ctx.selection(&dense_selection, None)?
                } else {
                    ctx.selection(&self.selection.indices, None)?
                };
                let reference_gpu =
                    ctx.reference(&convert_coords(self.reference.as_ref().unwrap()))?;
                self.gpu = Some(RmsdGpuState {
                    selection,
                    reference: reference_gpu,
                });
            }
        }

        let reference = self
            .reference
            .as_ref()
            .ok_or_else(|| TrajError::Mismatch("reference not initialized".into()))?;

        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            if self.align {
                let cov = ctx.rmsd_covariance(
                    &coords,
                    n_atoms,
                    chunk.n_frames,
                    &gpu.selection,
                    &gpu.reference,
                )?;
                let n_sel = gpu.selection.n_sel();
                for frame in 0..chunk.n_frames {
                    let rmsd = rmsd_from_cov(
                        &cov.cov[frame],
                        cov.sum_x2[frame] as f64,
                        cov.sum_y2[frame] as f64,
                        n_sel,
                    );
                    self.results.push(rmsd);
                }
            } else {
                let results = ctx.rmsd_raw(
                    &coords,
                    n_atoms,
                    chunk.n_frames,
                    &gpu.selection,
                    &gpu.reference,
                )?;
                self.results.extend(results);
            }
            return Ok(());
        }

        process_cpu_chunk_out(
            chunk,
            n_atoms,
            reference,
            &self.selection_usize,
            self.align,
            &mut self.results,
        );
        Ok(())
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        source_selection: &[u32],
        _system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        let _ = device;
        if source_selection != self.selection.indices.as_slice() {
            return Err(TrajError::Mismatch(
                "rmsd selected chunk does not match expected IO selection".into(),
            ));
        }
        if self.reference.is_none() && matches!(self.reference_mode, ReferenceMode::Frame0) {
            if chunk.n_frames == 0 {
                return Ok(());
            }
            let end = chunk.n_atoms.min(chunk.coords.len());
            self.reference = Some(chunk.coords[..end].to_vec());

            #[cfg(feature = "cuda")]
            if let Device::Cuda(ctx) = device {
                let dense_selection: Vec<u32> =
                    (0..chunk.n_atoms).map(|i| i as u32).collect();
                let selection = ctx.selection(&dense_selection, None)?;
                let reference_gpu =
                    ctx.reference(&convert_coords(self.reference.as_ref().unwrap()))?;
                self.gpu = Some(RmsdGpuState {
                    selection,
                    reference: reference_gpu,
                });
            }
        }

        let reference = self
            .reference
            .as_ref()
            .ok_or_else(|| TrajError::Mismatch("reference not initialized".into()))?;

        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            if self.align {
                let cov = ctx.rmsd_covariance(
                    &coords,
                    chunk.n_atoms,
                    chunk.n_frames,
                    &gpu.selection,
                    &gpu.reference,
                )?;
                let n_sel = gpu.selection.n_sel();
                for frame in 0..chunk.n_frames {
                    let rmsd = rmsd_from_cov(
                        &cov.cov[frame],
                        cov.sum_x2[frame] as f64,
                        cov.sum_y2[frame] as f64,
                        n_sel,
                    );
                    self.results.push(rmsd);
                }
            } else {
                let results = ctx.rmsd_raw(
                    &coords,
                    chunk.n_atoms,
                    chunk.n_frames,
                    &gpu.selection,
                    &gpu.reference,
                )?;
                self.results.extend(results);
            }
            return Ok(());
        }

        process_cpu_chunk_out(
            chunk,
            chunk.n_atoms,
            reference,
            &self.dense_selection_usize,
            self.align,
            &mut self.results,
        );
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

impl Plan for SymmRmsdPlan {
    fn name(&self) -> &'static str {
        "symmrmsd"
    }

    fn requirements(&self) -> PlanRequirements {
        self.inner.requirements()
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

impl Plan for PairwiseRmsdPlan {
    fn name(&self) -> &'static str {
        "pairwise_rmsd"
    }

    fn requirements(&self) -> PlanRequirements {
        let needs_box =
            matches!(self.metric, PairwiseMetric::Dme) && matches!(self.pbc, PbcMode::Orthorhombic);
        PlanRequirements::new(needs_box, false)
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.use_selected_input = matches!(_device, Device::Cpu);
        self.frames.clear();
        self.boxes.clear();
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.use_selected_input {
            Some(self.selection.indices.as_slice())
        } else {
            None
        }
    }

    fn preferred_selection_hint(&self, _system: &System) -> Option<&[u32]> {
        Some(self.selection.indices.as_slice())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let n_atoms = chunk.n_atoms;
        let sel = &self.selection.indices;
        for frame in 0..chunk.n_frames {
            let mut frame_sel = Vec::with_capacity(sel.len());
            for &idx in sel.iter() {
                frame_sel.push(chunk.coords[frame * n_atoms + idx as usize]);
            }
            self.frames.push(frame_sel);
            if matches!(self.metric, PairwiseMetric::Dme)
                && matches!(self.pbc, PbcMode::Orthorhombic)
            {
                self.boxes.push(Some(box_lengths(chunk, frame)?));
            } else {
                self.boxes.push(None);
            }
        }
        Ok(())
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        source_selection: &[u32],
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        if !self.use_selected_input {
            return Err(TrajError::Mismatch(
                "pairwise_rmsd selected chunk received while selected IO is disabled".into(),
            ));
        }
        if source_selection != self.selection.indices.as_slice() {
            return Err(TrajError::Mismatch(
                "pairwise_rmsd selected chunk does not match expected IO selection".into(),
            ));
        }

        let n_atoms = chunk.n_atoms;
        for frame in 0..chunk.n_frames {
            let start = frame * n_atoms;
            let end = start + n_atoms;
            self.frames.push(chunk.coords[start..end].to_vec());
            if matches!(self.metric, PairwiseMetric::Dme)
                && matches!(self.pbc, PbcMode::Orthorhombic)
            {
                self.boxes.push(Some(box_lengths(chunk, frame)?));
            } else {
                self.boxes.push(None);
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let n_frames = self.frames.len();
        if n_frames == 0 {
            return Ok(PlanOutput::Matrix {
                data: Vec::new(),
                rows: 0,
                cols: 0,
            });
        }
        let mut data = vec![0.0f32; n_frames * n_frames];
        let n_sel = self.selection.indices.len();
        if n_sel < 2 {
            return Ok(PlanOutput::Matrix {
                data,
                rows: n_frames,
                cols: n_frames,
            });
        }

        if matches!(self.metric, PairwiseMetric::Dme) {
            let mut dists = Vec::with_capacity(n_frames);
            for (idx, frame) in self.frames.iter().enumerate() {
                let box_ = if matches!(self.pbc, PbcMode::Orthorhombic) {
                    self.boxes[idx].ok_or_else(|| {
                        TrajError::Mismatch("pairwise_rmsd requires orthorhombic box".into())
                    })?
                } else {
                    (0.0, 0.0, 0.0)
                };
                let dist = pair_distances_compact(
                    frame,
                    self.pbc,
                    if matches!(self.pbc, PbcMode::Orthorhombic) {
                        Some(box_)
                    } else {
                        None
                    },
                );
                dists.push(dist);
            }
            let n_pairs = dists[0].len();
            for i in 0..n_frames {
                for j in (i + 1)..n_frames {
                    let mut sum = 0.0f64;
                    for k in 0..n_pairs {
                        let diff = dists[i][k] - dists[j][k];
                        sum += diff * diff;
                    }
                    let rmsd = if n_pairs == 0 {
                        0.0
                    } else {
                        (sum / n_pairs as f64).sqrt() as f32
                    };
                    data[i * n_frames + j] = rmsd;
                    data[j * n_frames + i] = rmsd;
                }
            }
            return Ok(PlanOutput::Matrix {
                data,
                rows: n_frames,
                cols: n_frames,
            });
        }

        for i in 0..n_frames {
            for j in (i + 1)..n_frames {
                let rmsd = match self.metric {
                    PairwiseMetric::Rms => rmsd_aligned(&self.frames[i], &self.frames[j]),
                    PairwiseMetric::Nofit => rmsd_raw(&self.frames[i], &self.frames[j]),
                    PairwiseMetric::Dme => 0.0,
                };
                data[i * n_frames + j] = rmsd;
                data[j * n_frames + i] = rmsd;
            }
        }

        Ok(PlanOutput::Matrix {
            data,
            rows: n_frames,
            cols: n_frames,
        })
    }
}
impl Plan for DistanceRmsdPlan {
    fn name(&self) -> &'static str {
        "distance_rmsd"
    }

    fn requirements(&self) -> PlanRequirements {
        let needs_box = matches!(self.pbc, PbcMode::Orthorhombic);
        PlanRequirements::new(needs_box, false)
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.reference_dists = None;
        match self.reference_mode {
            ReferenceMode::Topology => {
                if matches!(self.pbc, PbcMode::Orthorhombic) {
                    return Err(TrajError::Mismatch(
                        "distance_rmsd with topology reference does not support PBC".into(),
                    ));
                }
                let positions0 = system
                    .positions0
                    .as_ref()
                    .ok_or_else(|| TrajError::Mismatch("no topology reference coords".into()))?;
                let ref_dists =
                    pair_distances(positions0, &self.selection.indices, PbcMode::None, None);
                self.reference_dists = Some(ref_dists);
            }
            ReferenceMode::Frame0 => {}
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
        if self.reference_dists.is_none() && matches!(self.reference_mode, ReferenceMode::Frame0) {
            if chunk.n_frames == 0 {
                return Ok(());
            }
            let box_ = if matches!(self.pbc, PbcMode::Orthorhombic) {
                Some(box_lengths(chunk, 0)?)
            } else {
                None
            };
            let coords = &chunk.coords[0..n_atoms];
            let ref_dists = pair_distances(coords, &self.selection.indices, self.pbc, box_);
            self.reference_dists = Some(ref_dists);
        }

        let reference = self
            .reference_dists
            .as_ref()
            .ok_or_else(|| TrajError::Mismatch("reference not initialized".into()))?;
        let n_pairs = reference.len();
        if n_pairs == 0 {
            self.results
                .extend(std::iter::repeat(0.0).take(chunk.n_frames));
            return Ok(());
        }

        for frame in 0..chunk.n_frames {
            let box_ = if matches!(self.pbc, PbcMode::Orthorhombic) {
                Some(box_lengths(chunk, frame)?)
            } else {
                None
            };
            let mut sum = 0.0f64;
            let mut idx = 0usize;
            let sel = &self.selection.indices;
            for i in 0..sel.len() {
                let a_idx = sel[i] as usize;
                let pa = chunk.coords[frame * n_atoms + a_idx];
                for j in (i + 1)..sel.len() {
                    let b_idx = sel[j] as usize;
                    let pb = chunk.coords[frame * n_atoms + b_idx];
                    let mut dx = (pb[0] - pa[0]) as f64;
                    let mut dy = (pb[1] - pa[1]) as f64;
                    let mut dz = (pb[2] - pa[2]) as f64;
                    if let Some((lx, ly, lz)) = box_ {
                        apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
                    }
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    let diff = dist - reference[idx];
                    sum += diff * diff;
                    idx += 1;
                }
            }
            let rmsd = (sum / n_pairs as f64).sqrt() as f32;
            self.results.push(rmsd);
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

fn rmsd_raw(frame: &[[f32; 4]], reference: &[[f32; 4]]) -> f32 {
    let mut sum = 0.0f64;
    let n = frame.len().min(reference.len());
    for i in 0..n {
        let dx = frame[i][0] as f64 - reference[i][0] as f64;
        let dy = frame[i][1] as f64 - reference[i][1] as f64;
        let dz = frame[i][2] as f64 - reference[i][2] as f64;
        sum += dx * dx + dy * dy + dz * dz;
    }
    ((sum / n as f64).sqrt()) as f32
}

fn rmsd_raw_selected(frame: &[[f32; 4]], reference: &[[f32; 4]], selection: &[usize]) -> f32 {
    let n = selection.len().min(reference.len());
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for i in 0..n {
        let idx = selection[i];
        if idx >= frame.len() {
            break;
        }
        let dx = frame[idx][0] as f64 - reference[i][0] as f64;
        let dy = frame[idx][1] as f64 - reference[i][1] as f64;
        let dz = frame[idx][2] as f64 - reference[i][2] as f64;
        sum += dx * dx + dy * dy + dz * dz;
    }
    ((sum / n as f64).sqrt()) as f32
}

fn rmsd_aligned(frame: &[[f32; 4]], reference: &[[f32; 4]]) -> f32 {
    let n = frame.len().min(reference.len());
    if n == 0 {
        return 0.0;
    }
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        x.push(Vector3::new(
            frame[i][0] as f64,
            frame[i][1] as f64,
            frame[i][2] as f64,
        ));
        y.push(Vector3::new(
            reference[i][0] as f64,
            reference[i][1] as f64,
            reference[i][2] as f64,
        ));
    }
    let cx = centroid(&x);
    let cy = centroid(&y);
    let mut h: Matrix3<f64> = Matrix3::zeros();
    for i in 0..n {
        let xr = x[i] - cx;
        let yr = y[i] - cy;
        h += xr * yr.transpose();
    }
    let svd = h.svd(true, true);
    let (u, v_t) = match (svd.u, svd.v_t) {
        (Some(u), Some(v_t)) => (u, v_t),
        _ => return rmsd_raw(frame, reference),
    };
    let mut r: Matrix3<f64> = v_t.transpose() * u.transpose();
    if r.determinant() < 0.0 {
        let mut v_t_adj = v_t;
        v_t_adj.row_mut(2).neg_mut();
        r = v_t_adj.transpose() * u.transpose();
    }
    let mut sum = 0.0f64;
    for i in 0..n {
        let xr = x[i] - cx;
        let yr = y[i] - cy;
        let aligned = r * xr;
        let diff = aligned - yr;
        sum += diff.dot(&diff);
    }
    ((sum / n as f64).sqrt()) as f32
}

fn rmsd_aligned_selected(frame: &[[f32; 4]], reference: &[[f32; 4]], selection: &[usize]) -> f32 {
    let n = selection.len().min(reference.len());
    if n == 0 {
        return 0.0;
    }

    let mut cx = [0.0f64; 3];
    let mut cy = [0.0f64; 3];
    for i in 0..n {
        let idx = selection[i];
        if idx >= frame.len() {
            return 0.0;
        }
        let p = frame[idx];
        let q = reference[i];
        cx[0] += p[0] as f64;
        cx[1] += p[1] as f64;
        cx[2] += p[2] as f64;
        cy[0] += q[0] as f64;
        cy[1] += q[1] as f64;
        cy[2] += q[2] as f64;
    }
    let inv_n = 1.0 / n as f64;
    cx[0] *= inv_n;
    cx[1] *= inv_n;
    cx[2] *= inv_n;
    cy[0] *= inv_n;
    cy[1] *= inv_n;
    cy[2] *= inv_n;

    let mut h: Matrix3<f64> = Matrix3::zeros();
    for i in 0..n {
        let idx = selection[i];
        let p = frame[idx];
        let q = reference[i];
        let px = p[0] as f64 - cx[0];
        let py = p[1] as f64 - cx[1];
        let pz = p[2] as f64 - cx[2];
        let qx = q[0] as f64 - cy[0];
        let qy = q[1] as f64 - cy[1];
        let qz = q[2] as f64 - cy[2];

        h[(0, 0)] += px * qx;
        h[(0, 1)] += px * qy;
        h[(0, 2)] += px * qz;
        h[(1, 0)] += py * qx;
        h[(1, 1)] += py * qy;
        h[(1, 2)] += py * qz;
        h[(2, 0)] += pz * qx;
        h[(2, 1)] += pz * qy;
        h[(2, 2)] += pz * qz;
    }

    let svd = h.svd(true, true);
    let (u, v_t) = match (svd.u, svd.v_t) {
        (Some(u), Some(v_t)) => (u, v_t),
        _ => return rmsd_raw_selected(frame, reference, selection),
    };
    let mut r: Matrix3<f64> = v_t.transpose() * u.transpose();
    if r.determinant() < 0.0 {
        let mut v_t_adj = v_t;
        v_t_adj.row_mut(2).neg_mut();
        r = v_t_adj.transpose() * u.transpose();
    }

    let mut sum = 0.0f64;
    for i in 0..n {
        let idx = selection[i];
        let p = frame[idx];
        let q = reference[i];
        let px = p[0] as f64 - cx[0];
        let py = p[1] as f64 - cx[1];
        let pz = p[2] as f64 - cx[2];
        let qx = q[0] as f64 - cy[0];
        let qy = q[1] as f64 - cy[1];
        let qz = q[2] as f64 - cy[2];

        let ax = r[(0, 0)] * px + r[(0, 1)] * py + r[(0, 2)] * pz;
        let ay = r[(1, 0)] * px + r[(1, 1)] * py + r[(1, 2)] * pz;
        let az = r[(2, 0)] * px + r[(2, 1)] * py + r[(2, 2)] * pz;

        let dx = ax - qx;
        let dy = ay - qy;
        let dz = az - qz;
        sum += dx * dx + dy * dy + dz * dz;
    }
    ((sum * inv_n).sqrt()) as f32
}

fn pair_distances(
    coords: &[[f32; 4]],
    sel: &[u32],
    pbc: PbcMode,
    box_: Option<(f64, f64, f64)>,
) -> Vec<f64> {
    let n_pairs = sel.len().saturating_sub(1) * sel.len() / 2;
    let mut out = Vec::with_capacity(n_pairs);
    for i in 0..sel.len() {
        let a_idx = sel[i] as usize;
        let pa = coords[a_idx];
        for j in (i + 1)..sel.len() {
            let b_idx = sel[j] as usize;
            let pb = coords[b_idx];
            let mut dx = (pb[0] - pa[0]) as f64;
            let mut dy = (pb[1] - pa[1]) as f64;
            let mut dz = (pb[2] - pa[2]) as f64;
            if matches!(pbc, PbcMode::Orthorhombic) {
                if let Some((lx, ly, lz)) = box_ {
                    apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
                }
            }
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            out.push(dist);
        }
    }
    out
}

fn pair_distances_compact(
    coords: &[[f32; 4]],
    pbc: PbcMode,
    box_: Option<(f64, f64, f64)>,
) -> Vec<f64> {
    let n_pairs = coords.len().saturating_sub(1) * coords.len() / 2;
    let mut out = Vec::with_capacity(n_pairs);
    let (lx, ly, lz) = box_.unwrap_or((0.0, 0.0, 0.0));
    for i in 0..coords.len() {
        let pa = coords[i];
        for j in (i + 1)..coords.len() {
            let pb = coords[j];
            let mut dx = (pb[0] - pa[0]) as f64;
            let mut dy = (pb[1] - pa[1]) as f64;
            let mut dz = (pb[2] - pa[2]) as f64;
            if matches!(pbc, PbcMode::Orthorhombic) {
                apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
            }
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            out.push(dist);
        }
    }
    out
}

fn box_lengths(chunk: &FrameChunk, frame: usize) -> TrajResult<(f64, f64, f64)> {
    match chunk.box_[frame] {
        Box3::Orthorhombic { lx, ly, lz } => Ok((lx as f64, ly as f64, lz as f64)),
        _ => Err(TrajError::Mismatch(
            "orthorhombic box required for PBC".into(),
        )),
    }
}

fn apply_pbc(dx: &mut f64, dy: &mut f64, dz: &mut f64, lx: f64, ly: f64, lz: f64) {
    if lx > 0.0 {
        *dx -= (*dx / lx).round() * lx;
    }
    if ly > 0.0 {
        *dy -= (*dy / ly).round() * ly;
    }
    if lz > 0.0 {
        *dz -= (*dz / lz).round() * lz;
    }
}
