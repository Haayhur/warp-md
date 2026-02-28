use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk, FrameChunkBuilder};
use traj_core::system::System;
use traj_io::TrajReader;

#[cfg(feature = "cuda")]
use traj_gpu::GpuContext;

#[derive(Clone)]
pub enum Device {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(GpuContext),
}

impl Device {
    pub fn cpu() -> Self {
        Device::Cpu
    }

    pub fn from_spec(spec: &str) -> TrajResult<Self> {
        let spec = spec.trim();
        if spec.eq_ignore_ascii_case("cpu") {
            return Ok(Device::Cpu);
        }
        if spec.eq_ignore_ascii_case("auto") {
            #[cfg(feature = "cuda")]
            {
                if let Ok(ctx) = GpuContext::new(0) {
                    return Ok(Device::Cuda(ctx));
                }
            }
            return Ok(Device::Cpu);
        }
        if spec.to_ascii_lowercase().starts_with("cuda") {
            #[cfg(feature = "cuda")]
            {
                let idx = parse_cuda_index(spec)?;
                return Ok(Device::Cuda(GpuContext::new(idx)?));
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(TrajError::Unsupported(
                    "cuda feature disabled; rebuild with --features cuda".into(),
                ));
            }
        }
        Err(TrajError::Unsupported(format!(
            "unknown device spec '{spec}'"
        )))
    }
}

#[cfg(feature = "cuda")]
fn parse_cuda_index(spec: &str) -> TrajResult<usize> {
    if let Some((head, tail)) = spec.split_once(':') {
        if !head.eq_ignore_ascii_case("cuda") || tail.contains(':') {
            return Err(TrajError::Parse(format!(
                "invalid cuda device spec '{spec}'"
            )));
        }
        return tail
            .parse()
            .map_err(|_| TrajError::Parse(format!("invalid cuda device spec '{spec}'")));
    }
    if spec.eq_ignore_ascii_case("cuda") {
        return Ok(0);
    }
    Err(TrajError::Parse(format!(
        "invalid cuda device spec '{spec}'"
    )))
}

pub struct Executor {
    system: System,
    chunk_frames: usize,
    device: Device,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanRequirements {
    pub needs_box: bool,
    pub needs_time: bool,
}

impl PlanRequirements {
    pub const fn new(needs_box: bool, needs_time: bool) -> Self {
        Self {
            needs_box,
            needs_time,
        }
    }
}

impl Default for PlanRequirements {
    fn default() -> Self {
        Self::new(true, true)
    }
}

#[derive(Clone)]
pub struct SelectedFrame {
    pub coords: Vec<[f32; 4]>,
    pub box_: Box3,
    pub time_ps: Option<f32>,
}

pub struct SelectedFramesReader {
    n_atoms: usize,
    frames: Vec<SelectedFrame>,
    cursor: usize,
}

impl SelectedFramesReader {
    pub fn new(n_atoms: usize, frames: Vec<SelectedFrame>) -> Self {
        Self {
            n_atoms,
            frames,
            cursor: 0,
        }
    }
}

impl TrajReader for SelectedFramesReader {
    fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    fn n_frames_hint(&self) -> Option<usize> {
        Some(self.frames.len())
    }

    fn read_chunk(&mut self, max_frames: usize, out: &mut FrameChunkBuilder) -> TrajResult<usize> {
        let max_frames = max_frames.max(1);
        if self.cursor >= self.frames.len() {
            return Ok(0);
        }
        out.reset(self.n_atoms, max_frames);
        let mut written = 0usize;
        while written < max_frames && self.cursor < self.frames.len() {
            let frame = &self.frames[self.cursor];
            let dst = out.start_frame(frame.box_, frame.time_ps);
            dst.copy_from_slice(&frame.coords);
            self.cursor += 1;
            written += 1;
        }
        Ok(written)
    }
}

struct FrameLimitReader<'a, R: TrajReader> {
    inner: &'a mut R,
    remaining: usize,
}

impl<'a, R: TrajReader> FrameLimitReader<'a, R> {
    fn new(inner: &'a mut R, remaining: usize) -> Self {
        Self { inner, remaining }
    }
}

impl<R: TrajReader> TrajReader for FrameLimitReader<'_, R> {
    fn n_atoms(&self) -> usize {
        self.inner.n_atoms()
    }

    fn n_frames_hint(&self) -> Option<usize> {
        match self.inner.n_frames_hint() {
            Some(hint) => Some(hint.min(self.remaining)),
            None => Some(self.remaining),
        }
    }

    fn read_chunk(&mut self, max_frames: usize, out: &mut FrameChunkBuilder) -> TrajResult<usize> {
        if self.remaining == 0 {
            return Ok(0);
        }
        let capped = max_frames.max(1).min(self.remaining);
        let read = self.inner.read_chunk(capped, out)?;
        self.remaining = self.remaining.saturating_sub(read);
        Ok(read)
    }

    fn read_chunk_selected(
        &mut self,
        max_frames: usize,
        selection: &[u32],
        out: &mut FrameChunkBuilder,
    ) -> TrajResult<usize> {
        if self.remaining == 0 {
            return Ok(0);
        }
        let capped = max_frames.max(1).min(self.remaining);
        let read = self.inner.read_chunk_selected(capped, selection, out)?;
        self.remaining = self.remaining.saturating_sub(read);
        Ok(read)
    }
}

pub fn normalize_frame_indices(frame_indices: Vec<i64>, n_frames: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(frame_indices.len());
    for raw in frame_indices.into_iter() {
        let mut idx = raw;
        if idx < 0 {
            idx += n_frames as i64;
        }
        if idx >= 0 && (idx as usize) < n_frames {
            out.push(idx as usize);
        }
    }
    out
}

fn validate_selection(selection: &[u32], n_atoms: usize) -> TrajResult<()> {
    for &idx in selection {
        if (idx as usize) >= n_atoms {
            return Err(TrajError::Mismatch(format!(
                "selection index {idx} out of bounds for trajectory with {n_atoms} atoms"
            )));
        }
    }
    Ok(())
}

fn extract_chunk_frame(chunk: &FrameChunk, frame: usize) -> SelectedFrame {
    let start = frame * chunk.n_atoms;
    let end = start + chunk.n_atoms;
    SelectedFrame {
        coords: chunk.coords[start..end].to_vec(),
        box_: chunk.box_.get(frame).copied().unwrap_or(Box3::None),
        time_ps: chunk.time_ps.as_ref().and_then(|t| t.get(frame).copied()),
    }
}

pub fn count_frames<R: TrajReader>(reader: &mut R, chunk_frames: usize) -> TrajResult<usize> {
    let chunk_frames = chunk_frames.max(1);
    let mut builder = FrameChunkBuilder::new(reader.n_atoms(), chunk_frames);
    let mut total = 0usize;
    loop {
        let read = reader.read_chunk(chunk_frames, &mut builder)?;
        if read == 0 {
            break;
        }
        total += read;
    }
    Ok(total)
}

pub fn collect_selected_frames<R: TrajReader>(
    reader: &mut R,
    source_indices: &[usize],
    chunk_frames: usize,
) -> TrajResult<Vec<SelectedFrame>> {
    collect_selected_frames_with_requirements(
        reader,
        source_indices,
        chunk_frames,
        PlanRequirements::default(),
    )
}

pub fn collect_selected_frames_with_requirements<R: TrajReader>(
    reader: &mut R,
    source_indices: &[usize],
    chunk_frames: usize,
    requirements: PlanRequirements,
) -> TrajResult<Vec<SelectedFrame>> {
    if source_indices.is_empty() {
        return Ok(Vec::new());
    }

    let chunk_frames = chunk_frames.max(1);
    let mut builder = FrameChunkBuilder::new(reader.n_atoms(), chunk_frames);
    builder.set_requirements(requirements.needs_box, requirements.needs_time);

    let mut targets: Vec<(usize, usize)> = source_indices
        .iter()
        .copied()
        .enumerate()
        .map(|(pos, frame_idx)| (frame_idx, pos))
        .collect();
    targets.sort_unstable_by_key(|(frame_idx, _)| *frame_idx);

    let mut ordered: Vec<Option<SelectedFrame>> = vec![None; source_indices.len()];
    let mut target_cursor = 0usize;
    let target_len = targets.len();
    let mut global = 0usize;
    loop {
        if target_cursor >= target_len {
            break;
        }
        let read = reader.read_chunk(chunk_frames, &mut builder)?;
        if read == 0 {
            break;
        }
        let chunk = builder.finish_take()?;
        let mut done = false;
        for frame in 0..chunk.n_frames {
            if target_cursor >= target_len {
                done = true;
                break;
            }
            let frame_idx = global;
            global += 1;

            // Fast forward in case requested frame indices include stale values
            // (for example duplicated/unsorted sources already passed).
            while target_cursor < target_len && targets[target_cursor].0 < frame_idx {
                target_cursor += 1;
            }
            if target_cursor >= target_len {
                done = true;
                break;
            }
            if targets[target_cursor].0 != frame_idx {
                continue;
            }

            let selected = extract_chunk_frame(&chunk, frame);
            let mut end = target_cursor + 1;
            while end < target_len && targets[end].0 == frame_idx {
                end += 1;
            }

            if end == target_cursor + 1 {
                let out_pos = targets[target_cursor].1;
                ordered[out_pos] = Some(selected);
            } else {
                for &(.., out_pos) in &targets[target_cursor..end - 1] {
                    ordered[out_pos] = Some(selected.clone());
                }
                let out_pos = targets[end - 1].1;
                ordered[out_pos] = Some(selected);
            }
            target_cursor = end;
        }
        builder.reclaim(chunk);
        if done {
            break;
        }
    }

    Ok(ordered.into_iter().flatten().collect())
}

fn is_contiguous_prefix(source_indices: &[usize]) -> bool {
    source_indices
        .iter()
        .enumerate()
        .all(|(expected, &actual)| actual == expected)
}

impl Executor {
    pub fn new(system: System) -> Self {
        Self {
            system,
            chunk_frames: 128,
            device: Device::Cpu,
        }
    }

    pub fn with_chunk_frames(mut self, chunk_frames: usize) -> Self {
        self.chunk_frames = chunk_frames.max(1);
        self
    }

    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn with_device_spec(mut self, spec: &str) -> TrajResult<Self> {
        self.device = Device::from_spec(spec)?;
        Ok(self)
    }

    pub fn system(&self) -> &System {
        &self.system
    }

    pub fn system_mut(&mut self) -> &mut System {
        &mut self.system
    }

    pub fn run_plan<P: Plan>(
        &mut self,
        plan: &mut P,
        traj: &mut dyn TrajReader,
    ) -> TrajResult<PlanOutput> {
        if traj.n_atoms() != self.system.n_atoms() {
            return Err(TrajError::Mismatch(
                "trajectory atom count does not match system".into(),
            ));
        }
        plan.set_frames_hint(traj.n_frames_hint());
        plan.init(&self.system, &self.device)?;
        let requirements = plan.requirements();
        if let Some(selection) = plan.preferred_selection().map(|sel| sel.to_vec()) {
            validate_selection(&selection, self.system.n_atoms())?;
            let mut builder = FrameChunkBuilder::new(selection.len(), self.chunk_frames);
            builder.set_requirements(requirements.needs_box, requirements.needs_time);
            loop {
                let frames =
                    traj.read_chunk_selected(self.chunk_frames, &selection, &mut builder)?;
                if frames == 0 {
                    break;
                }
                let chunk = builder.finish_take()?;
                plan.process_chunk_selected(&chunk, &selection, &self.system, &self.device)?;
                builder.reclaim(chunk);
            }
        } else {
            let mut builder = FrameChunkBuilder::new(self.system.n_atoms(), self.chunk_frames);
            builder.set_requirements(requirements.needs_box, requirements.needs_time);
            loop {
                let frames = traj.read_chunk(self.chunk_frames, &mut builder)?;
                if frames == 0 {
                    break;
                }
                let chunk = builder.finish_take()?;
                plan.process_chunk(&chunk, &self.system, &self.device)?;
                builder.reclaim(chunk);
            }
        }
        plan.finalize()
    }

    pub fn run_plan_on_selected_frames<P: Plan, R: TrajReader>(
        &mut self,
        plan: &mut P,
        traj: &mut R,
        source_indices: &[usize],
    ) -> TrajResult<PlanOutput> {
        // Fast path for common capped-prefix subsets: avoid collecting and replaying
        // selected frames through an intermediate in-memory trajectory.
        if is_contiguous_prefix(source_indices) {
            let mut limited = FrameLimitReader::new(traj, source_indices.len());
            return self.run_plan(plan, &mut limited);
        }
        let selected = collect_selected_frames_with_requirements(
            traj,
            source_indices,
            self.chunk_frames,
            plan.requirements(),
        )?;
        let mut selected_reader = SelectedFramesReader::new(traj.n_atoms(), selected);
        self.run_plan(plan, &mut selected_reader)
    }
}

pub trait Plan {
    fn name(&self) -> &'static str;
    fn set_frames_hint(&mut self, _n_frames: Option<usize>) {}
    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::default()
    }
    fn preferred_n_atoms_hint(&self, _system: &System) -> Option<usize> {
        self.preferred_selection().map(|sel| sel.len())
    }
    fn preferred_selection_hint(&self, _system: &System) -> Option<&[u32]> {
        None
    }
    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()>;
    fn preferred_selection(&self) -> Option<&[u32]> {
        None
    }
    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        device: &Device,
    ) -> TrajResult<()>;
    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        _source_selection: &[u32],
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        self.process_chunk(chunk, system, device)
    }
    fn finalize(&mut self) -> TrajResult<PlanOutput>;
}

pub enum PlanOutput {
    Series(Vec<f32>),
    Matrix {
        data: Vec<f32>,
        rows: usize,
        cols: usize,
    },
    Histogram {
        centers: Vec<f32>,
        counts: Vec<u64>,
    },
    Rdf(RdfOutput),
    Persistence(PersistenceOutput),
    TimeSeries {
        time: Vec<f32>,
        data: Vec<f32>,
        rows: usize,
        cols: usize,
    },
    Dielectric(DielectricOutput),
    StructureFactor(StructureFactorOutput),
    Grid(GridOutput),
    Pca(PcaOutput),
    Clustering(ClusteringOutput),
}

pub struct RdfOutput {
    pub r: Vec<f32>,
    pub g_r: Vec<f32>,
    pub counts: Vec<u64>,
}

pub struct PersistenceOutput {
    pub bond_autocorrelation: Vec<f32>,
    pub lb: f32,
    pub lp: f32,
    pub fit: Vec<f32>,
    pub kuhn_length: f32,
}

pub struct DielectricOutput {
    pub time: Vec<f32>,
    pub rot_sq: Vec<f32>,
    pub trans_sq: Vec<f32>,
    pub rot_trans: Vec<f32>,
    pub dielectric_rot: f32,
    pub dielectric_total: f32,
    pub mu_avg: f32,
}

pub struct StructureFactorOutput {
    pub r: Vec<f32>,
    pub g_r: Vec<f32>,
    pub q: Vec<f32>,
    pub s_q: Vec<f32>,
}

pub struct GridOutput {
    pub dims: [usize; 3],
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub first: Vec<u32>,
    pub last: Vec<u32>,
    pub min: Vec<u32>,
    pub max: Vec<u32>,
}

pub struct PcaOutput {
    pub eigenvalues: Vec<f32>,
    pub eigenvectors: Vec<f32>,
    pub n_components: usize,
    pub n_features: usize,
}

pub struct ClusteringOutput {
    pub labels: Vec<i32>,
    pub centroids: Vec<u32>,
    pub sizes: Vec<u32>,
    pub method: String,
    pub n_frames: usize,
}

#[cfg(test)]
mod tests {
    use super::{
        collect_selected_frames, normalize_frame_indices, Device, Executor, Plan, PlanOutput,
        PlanRequirements,
    };
    use traj_core::frame::{Box3, FrameChunkBuilder};
    use traj_core::interner::StringInterner;
    use traj_core::system::{AtomTable, System};
    use traj_io::TrajReader;

    struct MockReader {
        n_atoms: usize,
        coords: Vec<Vec<[f32; 4]>>,
        cursor: usize,
    }

    impl MockReader {
        fn new(coords: Vec<Vec<[f32; 4]>>) -> Self {
            let n_atoms = coords.first().map(|f| f.len()).unwrap_or(0);
            Self {
                n_atoms,
                coords,
                cursor: 0,
            }
        }
    }

    impl TrajReader for MockReader {
        fn n_atoms(&self) -> usize {
            self.n_atoms
        }

        fn n_frames_hint(&self) -> Option<usize> {
            Some(self.coords.len())
        }

        fn read_chunk(
            &mut self,
            max_frames: usize,
            out: &mut FrameChunkBuilder,
        ) -> traj_core::error::TrajResult<usize> {
            if self.cursor >= self.coords.len() {
                return Ok(0);
            }
            let max_frames = max_frames.max(1);
            out.reset(self.n_atoms, max_frames);
            let mut written = 0usize;
            while written < max_frames && self.cursor < self.coords.len() {
                let dst = out.start_frame(Box3::None, Some(self.cursor as f32));
                dst.copy_from_slice(&self.coords[self.cursor]);
                self.cursor += 1;
                written += 1;
            }
            Ok(written)
        }
    }

    #[test]
    fn normalize_frame_indices_resolves_negative_and_drops_oob() {
        let got = normalize_frame_indices(vec![0, -1, 4, -10], 5);
        assert_eq!(got, vec![0, 4, 4]);
    }

    #[test]
    fn collect_selected_frames_keeps_input_order_and_duplicates() {
        let frames = vec![
            vec![[0.0, 0.0, 0.0, 0.0]],
            vec![[1.0, 0.0, 0.0, 0.0]],
            vec![[2.0, 0.0, 0.0, 0.0]],
        ];
        let mut reader = MockReader::new(frames);
        let selected = collect_selected_frames(&mut reader, &[2, 0, 2], 2).unwrap();
        assert_eq!(selected.len(), 3);
        assert_eq!(selected[0].coords[0][0], 2.0);
        assert_eq!(selected[1].coords[0][0], 0.0);
        assert_eq!(selected[2].coords[0][0], 2.0);
    }

    #[test]
    fn collect_selected_frames_stops_after_last_requested_index() {
        struct CountingReader {
            n_atoms: usize,
            n_frames: usize,
            cursor: usize,
            read_calls: usize,
        }

        impl CountingReader {
            fn new(n_atoms: usize, n_frames: usize) -> Self {
                Self {
                    n_atoms,
                    n_frames,
                    cursor: 0,
                    read_calls: 0,
                }
            }
        }

        impl TrajReader for CountingReader {
            fn n_atoms(&self) -> usize {
                self.n_atoms
            }

            fn n_frames_hint(&self) -> Option<usize> {
                Some(self.n_frames)
            }

            fn read_chunk(
                &mut self,
                max_frames: usize,
                out: &mut FrameChunkBuilder,
            ) -> traj_core::error::TrajResult<usize> {
                self.read_calls += 1;
                if self.cursor >= self.n_frames {
                    return Ok(0);
                }
                out.reset(self.n_atoms, max_frames.max(1));
                let mut written = 0usize;
                while written < max_frames && self.cursor < self.n_frames {
                    let dst = out.start_frame(Box3::None, Some(self.cursor as f32));
                    for atom in dst.iter_mut() {
                        *atom = [self.cursor as f32, 0.0, 0.0, 0.0];
                    }
                    self.cursor += 1;
                    written += 1;
                }
                Ok(written)
            }
        }

        let mut reader = CountingReader::new(1, 100);
        let selected = collect_selected_frames(&mut reader, &[0], 10).unwrap();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].coords[0][0], 0.0);
        assert_eq!(reader.read_calls, 1);
    }

    struct NoMetaPlan {
        frames: usize,
    }

    impl NoMetaPlan {
        fn new() -> Self {
            Self { frames: 0 }
        }
    }

    impl Plan for NoMetaPlan {
        fn name(&self) -> &'static str {
            "no_meta_test"
        }

        fn requirements(&self) -> PlanRequirements {
            PlanRequirements::new(false, false)
        }

        fn init(&mut self, _system: &System, _device: &Device) -> traj_core::error::TrajResult<()> {
            self.frames = 0;
            Ok(())
        }

        fn process_chunk(
            &mut self,
            chunk: &traj_core::frame::FrameChunk,
            _system: &System,
            _device: &Device,
        ) -> traj_core::error::TrajResult<()> {
            assert!(chunk.box_.is_empty());
            assert!(chunk.time_ps.is_none());
            self.frames += chunk.n_frames;
            Ok(())
        }

        fn finalize(&mut self) -> traj_core::error::TrajResult<PlanOutput> {
            Ok(PlanOutput::Series(vec![self.frames as f32]))
        }
    }

    #[test]
    fn executor_applies_plan_requirements_to_reader_buffers() {
        let mut atoms = AtomTable::default();
        atoms.name_id.push(0);
        atoms.resname_id.push(0);
        atoms.resid.push(1);
        atoms.chain_id.push(0);
        atoms.element_id.push(0);
        atoms.mass.push(12.0);
        let system = System::with_atoms(atoms, StringInterner::default(), None);

        let frames = vec![
            vec![[0.0, 0.0, 0.0, 1.0]],
            vec![[1.0, 0.0, 0.0, 1.0]],
            vec![[2.0, 0.0, 0.0, 1.0]],
        ];
        let mut reader = MockReader::new(frames);
        let mut exec = Executor::new(system).with_chunk_frames(2);
        let mut plan = NoMetaPlan::new();
        let out = exec.run_plan(&mut plan, &mut reader).unwrap();
        match out {
            PlanOutput::Series(values) => assert_eq!(values, vec![3.0]),
            _ => panic!("unexpected output"),
        }
    }

    fn build_system(n_atoms: usize) -> System {
        let mut atoms = AtomTable::default();
        for _ in 0..n_atoms {
            atoms.name_id.push(0);
            atoms.resname_id.push(0);
            atoms.resid.push(1);
            atoms.chain_id.push(0);
            atoms.element_id.push(0);
            atoms.mass.push(12.0);
        }
        System::with_atoms(atoms, StringInterner::default(), None)
    }

    struct SelectedCountingReader {
        n_atoms: usize,
        n_frames: usize,
        cursor: usize,
        read_chunk_calls: usize,
        read_chunk_selected_calls: usize,
    }

    impl SelectedCountingReader {
        fn new(n_atoms: usize, n_frames: usize) -> Self {
            Self {
                n_atoms,
                n_frames,
                cursor: 0,
                read_chunk_calls: 0,
                read_chunk_selected_calls: 0,
            }
        }
    }

    impl TrajReader for SelectedCountingReader {
        fn n_atoms(&self) -> usize {
            self.n_atoms
        }

        fn n_frames_hint(&self) -> Option<usize> {
            Some(self.n_frames)
        }

        fn read_chunk(
            &mut self,
            max_frames: usize,
            out: &mut FrameChunkBuilder,
        ) -> traj_core::error::TrajResult<usize> {
            self.read_chunk_calls += 1;
            if self.cursor >= self.n_frames {
                return Ok(0);
            }
            out.reset(self.n_atoms, max_frames.max(1));
            let mut written = 0usize;
            while written < max_frames && self.cursor < self.n_frames {
                let dst = out.start_frame(Box3::None, Some(self.cursor as f32));
                for atom in dst.iter_mut() {
                    *atom = [self.cursor as f32, 0.0, 0.0, 1.0];
                }
                self.cursor += 1;
                written += 1;
            }
            Ok(written)
        }

        fn read_chunk_selected(
            &mut self,
            max_frames: usize,
            selection: &[u32],
            out: &mut FrameChunkBuilder,
        ) -> traj_core::error::TrajResult<usize> {
            self.read_chunk_selected_calls += 1;
            out.reset(selection.len(), max_frames.max(1));
            if self.cursor >= self.n_frames {
                return Ok(0);
            }
            let mut written = 0usize;
            while written < max_frames && self.cursor < self.n_frames {
                let dst = out.start_frame(Box3::None, Some(self.cursor as f32));
                for atom in dst.iter_mut() {
                    *atom = [self.cursor as f32, 1.0, 0.0, 1.0];
                }
                self.cursor += 1;
                written += 1;
            }
            Ok(written)
        }
    }

    struct PreferredSelectionCountPlan {
        frames: usize,
        selection: Vec<u32>,
    }

    impl PreferredSelectionCountPlan {
        fn new(selection: Vec<u32>) -> Self {
            Self {
                frames: 0,
                selection,
            }
        }
    }

    impl Plan for PreferredSelectionCountPlan {
        fn name(&self) -> &'static str {
            "preferred_selection_count"
        }

        fn requirements(&self) -> PlanRequirements {
            PlanRequirements::new(false, false)
        }

        fn init(&mut self, _system: &System, _device: &Device) -> traj_core::error::TrajResult<()> {
            self.frames = 0;
            Ok(())
        }

        fn preferred_selection(&self) -> Option<&[u32]> {
            Some(self.selection.as_slice())
        }

        fn process_chunk(
            &mut self,
            _chunk: &traj_core::frame::FrameChunk,
            _system: &System,
            _device: &Device,
        ) -> traj_core::error::TrajResult<()> {
            panic!("process_chunk should not be used when preferred_selection is set");
        }

        fn process_chunk_selected(
            &mut self,
            chunk: &traj_core::frame::FrameChunk,
            _source_selection: &[u32],
            _system: &System,
            _device: &Device,
        ) -> traj_core::error::TrajResult<()> {
            assert_eq!(chunk.n_atoms, self.selection.len());
            self.frames += chunk.n_frames;
            Ok(())
        }

        fn finalize(&mut self) -> traj_core::error::TrajResult<PlanOutput> {
            Ok(PlanOutput::Series(vec![self.frames as f32]))
        }
    }

    #[test]
    fn run_plan_on_selected_frames_contiguous_prefix_streams_directly() {
        let system = build_system(1);
        let mut reader = SelectedCountingReader::new(1, 16);
        let mut exec = Executor::new(system).with_chunk_frames(4);
        let mut plan = PreferredSelectionCountPlan::new(vec![0]);
        let out = exec
            .run_plan_on_selected_frames(&mut plan, &mut reader, &[0, 1, 2, 3, 4])
            .unwrap();
        match out {
            PlanOutput::Series(values) => assert_eq!(values, vec![5.0]),
            _ => panic!("unexpected output"),
        }
        assert_eq!(reader.read_chunk_calls, 0);
        assert_eq!(reader.read_chunk_selected_calls, 2);
    }

    #[test]
    fn run_plan_on_selected_frames_non_prefix_uses_collection_fallback() {
        let system = build_system(1);
        let mut reader = SelectedCountingReader::new(1, 16);
        let mut exec = Executor::new(system).with_chunk_frames(4);
        let mut plan = PreferredSelectionCountPlan::new(vec![0]);
        let out = exec
            .run_plan_on_selected_frames(&mut plan, &mut reader, &[1, 3, 5])
            .unwrap();
        match out {
            PlanOutput::Series(values) => assert_eq!(values, vec![3.0]),
            _ => panic!("unexpected output"),
        }
        assert!(reader.read_chunk_calls > 0);
        assert_eq!(reader.read_chunk_selected_calls, 0);
    }
}
