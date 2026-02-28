use std::collections::VecDeque;

use nalgebra::{Matrix3, Vector3};

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{ClusteringOutput, Device, Plan, PlanOutput, PlanRequirements};

#[cfg(feature = "cuda")]
use traj_gpu::convert_coords;

#[derive(Debug, Clone, Copy)]
pub enum ClusterMethod {
    Dbscan {
        eps: f32,
        min_samples: usize,
    },
    Kmeans {
        n_clusters: usize,
        max_iter: usize,
        tol: f32,
        seed: u64,
    },
}

pub struct TrajectoryClusterPlan {
    selection: Selection,
    method: ClusterMethod,
    memory_budget_bytes: Option<usize>,
    use_selected_input: bool,
    frames: Vec<Vec<[f32; 4]>>,
    #[cfg(feature = "cuda")]
    gpu: Option<traj_gpu::GpuContext>,
}

impl TrajectoryClusterPlan {
    pub fn new(selection: Selection, method: ClusterMethod) -> Self {
        Self {
            selection,
            method,
            memory_budget_bytes: None,
            use_selected_input: false,
            frames: Vec::new(),
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    pub fn with_memory_budget_bytes(mut self, bytes: Option<usize>) -> Self {
        self.memory_budget_bytes = bytes;
        self
    }
}

impl Plan for TrajectoryClusterPlan {
    fn name(&self) -> &'static str {
        "trajectory_cluster"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(false, false)
    }

    fn init(&mut self, _system: &System, device: &Device) -> TrajResult<()> {
        if self.selection.indices.is_empty() {
            return Err(TrajError::Mismatch(
                "trajectory_cluster requires a non-empty selection".into(),
            ));
        }
        self.frames.clear();
        self.use_selected_input = matches!(device, Device::Cpu);
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = device {
                self.use_selected_input = false;
                self.gpu = Some(ctx.clone());
            }
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
        _device: &Device,
    ) -> TrajResult<()> {
        let n_atoms = chunk.n_atoms;
        for frame in 0..chunk.n_frames {
            let mut selected = Vec::with_capacity(self.selection.indices.len());
            let base = frame * n_atoms;
            for &idx in self.selection.indices.iter() {
                let atom = idx as usize;
                if atom >= n_atoms {
                    return Err(TrajError::Mismatch(
                        "trajectory_cluster selection index out of bounds".into(),
                    ));
                }
                selected.push(chunk.coords[base + atom]);
            }
            self.frames.push(selected);
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
                "trajectory_cluster selected chunk received while selected IO is disabled".into(),
            ));
        }
        if source_selection != self.selection.indices.as_slice() {
            return Err(TrajError::Mismatch(
                "trajectory_cluster selected chunk does not match expected IO selection".into(),
            ));
        }
        let n_atoms = chunk.n_atoms;
        for frame in 0..chunk.n_frames {
            let start = frame * n_atoms;
            let end = start + n_atoms;
            self.frames.push(chunk.coords[start..end].to_vec());
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let n_frames = self.frames.len();
        if n_frames == 0 {
            return Ok(PlanOutput::Clustering(ClusteringOutput {
                labels: Vec::new(),
                centroids: Vec::new(),
                sizes: Vec::new(),
                method: method_name(&self.method).to_string(),
                n_frames: 0,
            }));
        }

        let output = match self.method {
            ClusterMethod::Dbscan { eps, min_samples } => {
                if eps <= 0.0 {
                    return Err(TrajError::Parse(
                        "trajectory_cluster DBSCAN eps must be > 0".into(),
                    ));
                }
                let min_samples = min_samples.max(1);
                let pairs = n_frames.saturating_mul(n_frames.saturating_sub(1)) / 2;
                let bytes = pairs.saturating_mul(std::mem::size_of::<f32>());
                if let Some(limit) = self.memory_budget_bytes {
                    if bytes > limit {
                        return Err(TrajError::Unsupported(format!(
                            "trajectory_cluster distance matrix requires {bytes} bytes but memory_budget_bytes={limit}"
                        )));
                    }
                }
                #[allow(unused_mut)]
                let mut distances = pairwise_rmsd_triangular_cpu(&self.frames);
                #[cfg(feature = "cuda")]
                if let Some(ctx) = &self.gpu {
                    distances = pairwise_rmsd_triangular_gpu(ctx, &self.frames)?;
                }
                let (labels, centroids, sizes) =
                    dbscan_cluster(&distances, n_frames, eps as f64, min_samples);
                ClusteringOutput {
                    labels,
                    centroids,
                    sizes,
                    method: "dbscan".into(),
                    n_frames,
                }
            }
            ClusterMethod::Kmeans {
                n_clusters,
                max_iter,
                tol,
                seed,
            } => {
                let n_clusters = n_clusters.clamp(1, n_frames);
                let max_iter = max_iter.max(1);
                let tol = tol.max(0.0) as f64;
                let features = build_aligned_features(&self.frames)?;
                let (labels, centroids, sizes) =
                    kmeans_cluster(&features, n_clusters, max_iter, tol, seed);
                ClusteringOutput {
                    labels,
                    centroids,
                    sizes,
                    method: "kmeans".into(),
                    n_frames,
                }
            }
        };
        Ok(PlanOutput::Clustering(output))
    }
}

fn method_name(method: &ClusterMethod) -> &'static str {
    match method {
        ClusterMethod::Dbscan { .. } => "dbscan",
        ClusterMethod::Kmeans { .. } => "kmeans",
    }
}

fn tri_pairs_len(n: usize) -> usize {
    n.saturating_mul(n.saturating_sub(1)) / 2
}

fn tri_index(i: usize, j: usize, n: usize) -> usize {
    let (a, b) = if i < j { (i, j) } else { (j, i) };
    let row_offset = a * (2 * n - a - 1) / 2;
    row_offset + (b - a - 1)
}

fn tri_get(tri: &[f32], i: usize, j: usize, n: usize) -> f64 {
    if i == j {
        0.0
    } else {
        tri[tri_index(i, j, n)] as f64
    }
}

fn pairwise_rmsd_triangular_cpu(frames: &[Vec<[f32; 4]>]) -> Vec<f32> {
    let n = frames.len();
    let mut tri = vec![0.0f32; tri_pairs_len(n)];
    for i in 0..n {
        for j in (i + 1)..n {
            let rmsd = kabsch_rmsd(&frames[i], &frames[j]) as f32;
            tri[tri_index(i, j, n)] = rmsd;
        }
    }
    tri
}

#[cfg(feature = "cuda")]
fn pairwise_rmsd_triangular_gpu(
    ctx: &traj_gpu::GpuContext,
    frames: &[Vec<[f32; 4]>],
) -> TrajResult<Vec<f32>> {
    let n_frames = frames.len();
    let n_sel = frames[0].len();
    let mut coords = Vec::with_capacity(n_frames * n_sel);
    for frame in frames.iter() {
        coords.extend_from_slice(frame.as_slice());
    }
    let coords_gpu = convert_coords(&coords);
    let local_selection: Vec<u32> = (0..n_sel as u32).collect();
    let selection_gpu = ctx.selection(&local_selection, None)?;
    let mut tri = vec![0.0f32; tri_pairs_len(n_frames)];
    for i in 0..n_frames {
        let start = i * n_sel;
        let end = start + n_sel;
        let reference_gpu = ctx.reference(&convert_coords(&coords[start..end]))?;
        let cov =
            ctx.rmsd_covariance(&coords_gpu, n_sel, n_frames, &selection_gpu, &reference_gpu)?;
        for j in (i + 1)..n_frames {
            tri[tri_index(i, j, n_frames)] = rmsd_from_cov_local(
                &cov.cov[j],
                cov.sum_x2[j] as f64,
                cov.sum_y2[j] as f64,
                n_sel,
            ) as f32;
        }
    }
    Ok(tri)
}

fn dbscan_cluster(
    tri: &[f32],
    n_frames: usize,
    eps: f64,
    min_samples: usize,
) -> (Vec<i32>, Vec<u32>, Vec<u32>) {
    const UNCLASSIFIED: i32 = -2;
    const NOISE: i32 = -1;

    let mut labels = vec![UNCLASSIFIED; n_frames];
    let mut visited = vec![false; n_frames];
    let mut cluster_id = 0i32;
    let mut neigh = Vec::new();

    for i in 0..n_frames {
        if visited[i] {
            continue;
        }
        visited[i] = true;
        neighbors_within(tri, n_frames, i, eps, &mut neigh);
        if neigh.len() < min_samples {
            labels[i] = NOISE;
            continue;
        }
        labels[i] = cluster_id;
        let mut queued = vec![false; n_frames];
        let mut queue = VecDeque::new();
        for &p in neigh.iter() {
            if p != i && !queued[p] {
                queue.push_back(p);
                queued[p] = true;
            }
        }
        while let Some(p) = queue.pop_front() {
            if !visited[p] {
                visited[p] = true;
                neighbors_within(tri, n_frames, p, eps, &mut neigh);
                if neigh.len() >= min_samples {
                    for &q in neigh.iter() {
                        if !queued[q] {
                            queue.push_back(q);
                            queued[q] = true;
                        }
                    }
                }
            }
            if labels[p] == UNCLASSIFIED || labels[p] == NOISE {
                labels[p] = cluster_id;
            }
        }
        cluster_id += 1;
    }

    let n_clusters = cluster_id.max(0) as usize;
    let mut centroids = Vec::with_capacity(n_clusters);
    let mut sizes = Vec::with_capacity(n_clusters);
    for cid in 0..n_clusters {
        let mut members = Vec::new();
        for (idx, &label) in labels.iter().enumerate() {
            if label == cid as i32 {
                members.push(idx);
            }
        }
        sizes.push(members.len() as u32);
        if members.is_empty() {
            centroids.push(0);
            continue;
        }
        let mut best = members[0];
        let mut best_sum = f64::INFINITY;
        for &candidate in members.iter() {
            let mut sum = 0.0f64;
            for &other in members.iter() {
                sum += tri_get(tri, candidate, other, n_frames);
            }
            if sum < best_sum {
                best_sum = sum;
                best = candidate;
            }
        }
        centroids.push(best as u32);
    }
    (labels, centroids, sizes)
}

fn neighbors_within(tri: &[f32], n_frames: usize, point: usize, eps: f64, out: &mut Vec<usize>) {
    out.clear();
    for j in 0..n_frames {
        if tri_get(tri, point, j, n_frames) <= eps {
            out.push(j);
        }
    }
}

fn build_aligned_features(frames: &[Vec<[f32; 4]>]) -> TrajResult<Vec<Vec<f64>>> {
    let reference = &frames[0];
    let mut out = Vec::with_capacity(frames.len());
    for frame in frames.iter() {
        let aligned = align_to_reference(frame, reference)?;
        out.push(aligned);
    }
    Ok(out)
}

fn align_to_reference(frame: &[[f32; 4]], reference: &[[f32; 4]]) -> TrajResult<Vec<f64>> {
    if frame.len() != reference.len() {
        return Err(TrajError::Mismatch(
            "trajectory_cluster frame/reference size mismatch".into(),
        ));
    }
    if frame.is_empty() {
        return Ok(Vec::new());
    }
    let mut x = Vec::with_capacity(frame.len());
    let mut y = Vec::with_capacity(reference.len());
    for p in frame.iter() {
        x.push(Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64));
    }
    for p in reference.iter() {
        y.push(Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64));
    }
    let cx = centroid(&x);
    let cy = centroid(&y);
    let mut h = Matrix3::<f64>::zeros();
    for (px, py) in x.iter().zip(y.iter()) {
        let x0 = px - cx;
        let y0 = py - cy;
        h += x0 * y0.transpose();
    }
    let svd = h.svd(true, true);
    let (Some(u), Some(v_t)) = (svd.u, svd.v_t) else {
        return Err(TrajError::Unsupported(
            "trajectory_cluster Kabsch SVD failed".into(),
        ));
    };
    let mut r = v_t.transpose() * u.transpose();
    if r.determinant() < 0.0 {
        let mut v_t_fix = v_t;
        v_t_fix[(2, 0)] *= -1.0;
        v_t_fix[(2, 1)] *= -1.0;
        v_t_fix[(2, 2)] *= -1.0;
        r = v_t_fix.transpose() * u.transpose();
    }

    let mut out = Vec::with_capacity(frame.len() * 3);
    for px in x.iter() {
        let aligned = r * (px - cx) + cy;
        out.push(aligned[0]);
        out.push(aligned[1]);
        out.push(aligned[2]);
    }
    Ok(out)
}

fn kmeans_cluster(
    features: &[Vec<f64>],
    n_clusters: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> (Vec<i32>, Vec<u32>, Vec<u32>) {
    let n_samples = features.len();
    if n_samples == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let n_clusters = n_clusters.clamp(1, n_samples);
    let n_features = features[0].len();
    let mut rng_state = if seed == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        seed
    };

    let mut order: Vec<usize> = (0..n_samples).collect();
    fisher_yates_shuffle(&mut order, &mut rng_state);
    let mut centers: Vec<Vec<f64>> = order
        .iter()
        .take(n_clusters)
        .map(|&i| features[i].clone())
        .collect();
    let mut labels = vec![0i32; n_samples];
    let mut counts = vec![0usize; n_clusters];

    for _ in 0..max_iter {
        counts.fill(0);
        let mut sums = vec![vec![0.0f64; n_features]; n_clusters];
        for (i, sample) in features.iter().enumerate() {
            let mut best = 0usize;
            let mut best_dist = f64::INFINITY;
            for (c, center) in centers.iter().enumerate() {
                let d = sq_dist(sample, center);
                if d < best_dist {
                    best_dist = d;
                    best = c;
                }
            }
            labels[i] = best as i32;
            counts[best] += 1;
            let sum = &mut sums[best];
            for k in 0..n_features {
                sum[k] += sample[k];
            }
        }

        let mut max_shift = 0.0f64;
        for c in 0..n_clusters {
            if counts[c] == 0 {
                let idx = (xorshift64(&mut rng_state) as usize) % n_samples;
                let new_center = features[idx].clone();
                max_shift = max_shift.max(sq_dist(&centers[c], &new_center).sqrt());
                centers[c] = new_center;
                continue;
            }
            let inv = 1.0 / counts[c] as f64;
            let mut new_center = vec![0.0f64; n_features];
            for k in 0..n_features {
                new_center[k] = sums[c][k] * inv;
            }
            max_shift = max_shift.max(sq_dist(&centers[c], &new_center).sqrt());
            centers[c] = new_center;
        }
        if max_shift <= tol {
            break;
        }
    }

    counts.fill(0);
    for &label in labels.iter() {
        counts[label as usize] += 1;
    }

    let mut centroids = vec![0u32; n_clusters];
    for c in 0..n_clusters {
        let mut best_idx = 0usize;
        let mut best_dist = f64::INFINITY;
        for (i, sample) in features.iter().enumerate() {
            if labels[i] != c as i32 {
                continue;
            }
            let d = sq_dist(sample, &centers[c]);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        centroids[c] = best_idx as u32;
    }

    (
        labels,
        centroids,
        counts.into_iter().map(|v| v as u32).collect(),
    )
}

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

fn fisher_yates_shuffle(values: &mut [usize], state: &mut u64) {
    if values.len() < 2 {
        return;
    }
    for i in (1..values.len()).rev() {
        let j = (xorshift64(state) as usize) % (i + 1);
        values.swap(i, j);
    }
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    if x == 0 {
        x = 0x9E37_79B9_7F4A_7C15;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn kabsch_rmsd(a: &[[f32; 4]], b: &[[f32; 4]]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut x = Vec::with_capacity(a.len());
    let mut y = Vec::with_capacity(b.len());
    for p in a.iter() {
        x.push(Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64));
    }
    for p in b.iter() {
        y.push(Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64));
    }
    let cx = centroid(&x);
    let cy = centroid(&y);

    let mut h = Matrix3::<f64>::zeros();
    let mut sum_x2 = 0.0f64;
    let mut sum_y2 = 0.0f64;
    for (px, py) in x.iter().zip(y.iter()) {
        let x0 = px - cx;
        let y0 = py - cy;
        h += x0 * y0.transpose();
        sum_x2 += x0.dot(&x0);
        sum_y2 += y0.dot(&y0);
    }
    let svd = h.svd(true, true);
    let mut sigma_sum = svd.singular_values[0] + svd.singular_values[1] + svd.singular_values[2];
    if let (Some(u), Some(v_t)) = (svd.u, svd.v_t) {
        let det = (v_t.transpose() * u.transpose()).determinant();
        if det < 0.0 {
            sigma_sum -= 2.0 * svd.singular_values[2];
        }
    }
    let n = a.len() as f64;
    let rmsd2 = (sum_x2 + sum_y2 - 2.0 * sigma_sum) / n;
    if rmsd2 <= 0.0 {
        0.0
    } else {
        rmsd2.sqrt()
    }
}

fn centroid(points: &[Vector3<f64>]) -> Vector3<f64> {
    let mut c = Vector3::new(0.0, 0.0, 0.0);
    for p in points {
        c += p;
    }
    c / (points.len() as f64)
}

fn rmsd_from_cov_local(cov: &[f32; 9], sum_x2: f64, sum_y2: f64, n_sel: usize) -> f64 {
    if n_sel == 0 {
        return 0.0;
    }
    let cov_f64 = [
        cov[0] as f64,
        cov[1] as f64,
        cov[2] as f64,
        cov[3] as f64,
        cov[4] as f64,
        cov[5] as f64,
        cov[6] as f64,
        cov[7] as f64,
        cov[8] as f64,
    ];
    let m = Matrix3::from_row_slice(&cov_f64);
    let svd = m.svd(true, true);
    let mut sigma_sum = svd.singular_values[0] + svd.singular_values[1] + svd.singular_values[2];
    if let (Some(u), Some(v_t)) = (svd.u, svd.v_t) {
        let det = (v_t.transpose() * u.transpose()).determinant();
        if det < 0.0 {
            sigma_sum -= 2.0 * svd.singular_values[2];
        }
    }
    let n = n_sel as f64;
    let rmsd2 = (sum_x2 + sum_y2 - 2.0 * sigma_sum) / n;
    if rmsd2 <= 0.0 {
        0.0
    } else {
        rmsd2.sqrt()
    }
}
