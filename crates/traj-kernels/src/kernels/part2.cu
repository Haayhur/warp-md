    }
    extern __shared__ float smin[];
    smin[tid] = min_val;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float other = smin[tid + offset];
            if (other < smin[tid]) smin[tid] = other;
        }
        __syncthreads();
    }
    if (tid == 0) {
        float val = smin[0];
        out[frame] = (val >= 1.0e29f) ? 0.0f : val;
    }
}

__global__ void closest_atom_point(const Float4* coords,
                                   const unsigned int* sel,
                                   int n_sel,
                                   int n_atoms,
                                   int n_frames,
                                   Float4 point,
                                   const Float4* boxes,
                                   float* out) {
    int frame = blockIdx.x;
    int tid = threadIdx.x;
    if (frame >= n_frames) return;
    float min_val = 1.0e30f;
    unsigned int min_idx = 0;
    for (int i = tid; i < n_sel; i += blockDim.x) {
        unsigned int atom = sel[i];
        Float4 p = coords[frame * n_atoms + atom];
        Float4 box = boxes[frame];
        float dx = p.x - point.x;
        float dy = p.y - point.y;
        float dz = p.z - point.z;
        dx = wrap_pbc(dx, box.x);
        dy = wrap_pbc(dy, box.y);
        dz = wrap_pbc(dz, box.z);
        float dist = dx * dx + dy * dy + dz * dz;
        if (dist < min_val) {
            min_val = dist;
            min_idx = atom;
        }
    }
    extern __shared__ unsigned char smem[];
    float* sdist = (float*)smem;
    unsigned int* sidx = (unsigned int*)(sdist + blockDim.x);
    sdist[tid] = min_val;
    sidx[tid] = min_idx;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float other = sdist[tid + offset];
            unsigned int other_idx = sidx[tid + offset];
            if (other < sdist[tid]) {
                sdist[tid] = other;
                sidx[tid] = other_idx;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        if (sdist[0] >= 1.0e29f) {
            out[frame] = -1.0f;
        } else {
            out[frame] = (float)sidx[0];
        }
    }
}

__global__ void search_neighbors_count(const Float4* coords,
                                       const unsigned int* sel_a,
                                       const unsigned int* sel_b,
                                       int n_sel_a,
                                       int n_sel_b,
                                       int n_atoms,
                                       int n_frames,
                                       const Float4* boxes,
                                       float cutoff,
                                       unsigned int* counts) {
    int frame = blockIdx.y;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || b >= n_sel_b) return;
    unsigned int atom_b = sel_b[b];
    Float4 pb = coords[frame * n_atoms + atom_b];
    Float4 box = boxes[frame];
    float min_val = 1.0e30f;
    for (int a = 0; a < n_sel_a; a++) {
        unsigned int atom_a = sel_a[a];
        Float4 pa = coords[frame * n_atoms + atom_a];
        float dx = pb.x - pa.x;
        float dy = pb.y - pa.y;
        float dz = pb.z - pa.z;
        dx = wrap_pbc(dx, box.x);
        dy = wrap_pbc(dy, box.y);
        dz = wrap_pbc(dz, box.z);
        float dist = dx * dx + dy * dy + dz * dz;
        if (dist < min_val) min_val = dist;
    }
    if (min_val <= cutoff * cutoff) {
        atomicAdd(&counts[frame], 1);
    }
}

__global__ void min_dist_a(const Float4* coords,
                           const unsigned int* sel_a,
                           const unsigned int* sel_b,
                           int n_sel_a,
                           int n_sel_b,
                           int n_atoms,
                           int n_frames,
                           const Float4* boxes,
                           float* out) {
    int frame = blockIdx.y;
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || a >= n_sel_a) return;
    unsigned int atom_a = sel_a[a];
    Float4 pa = coords[frame * n_atoms + atom_a];
    Float4 box = boxes[frame];
    float min_val = 1.0e30f;
    for (int b = 0; b < n_sel_b; b++) {
        unsigned int atom_b = sel_b[b];
        Float4 pb = coords[frame * n_atoms + atom_b];
        float dx = pb.x - pa.x;
        float dy = pb.y - pa.y;
        float dz = pb.z - pa.z;
        dx = wrap_pbc(dx, box.x);
        dy = wrap_pbc(dy, box.y);
        dz = wrap_pbc(dz, box.z);
        float dist = sqrtf(dx * dx + dy * dy + dz * dz);
        if (dist < min_val) min_val = dist;
    }
    out[frame * n_sel_a + a] = min_val;
}

__global__ void min_dist_b(const Float4* coords,
                           const unsigned int* sel_a,
                           const unsigned int* sel_b,
                           int n_sel_a,
                           int n_sel_b,
                           int n_atoms,
                           int n_frames,
                           const Float4* boxes,
                           float* out) {
    int frame = blockIdx.y;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || b >= n_sel_b) return;
    unsigned int atom_b = sel_b[b];
    Float4 pb = coords[frame * n_atoms + atom_b];
    Float4 box = boxes[frame];
    float min_val = 1.0e30f;
    for (int a = 0; a < n_sel_a; a++) {
        unsigned int atom_a = sel_a[a];
        Float4 pa = coords[frame * n_atoms + atom_a];
        float dx = pb.x - pa.x;
        float dy = pb.y - pa.y;
        float dz = pb.z - pa.z;
        dx = wrap_pbc(dx, box.x);
        dy = wrap_pbc(dy, box.y);
        dz = wrap_pbc(dz, box.z);
        float dist = sqrtf(dx * dx + dy * dy + dz * dz);
        if (dist < min_val) min_val = dist;
    }
    out[frame * n_sel_b + b] = min_val;
}

__global__ void min_dist_a_triclinic(const Float4* coords,
                                     const unsigned int* sel_a,
                                     const unsigned int* sel_b,
                                     int n_sel_a,
                                     int n_sel_b,
                                     int n_atoms,
                                     int n_frames,
                                     const Float4* cell,
                                     const Float4* inv,
                                     float* out) {
    int frame = blockIdx.y;
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || a >= n_sel_a) return;
    unsigned int atom_a = sel_a[a];
    Float4 pa = coords[frame * n_atoms + atom_a];
    Float4 c0 = cell[frame * 3];
    Float4 c1 = cell[frame * 3 + 1];
    Float4 c2 = cell[frame * 3 + 2];
    Float4 r0 = inv[frame * 3];
    Float4 r1 = inv[frame * 3 + 1];
    Float4 r2 = inv[frame * 3 + 2];
    float min_val = 1.0e30f;
    for (int b = 0; b < n_sel_b; b++) {
        unsigned int atom_b = sel_b[b];
        Float4 pb = coords[frame * n_atoms + atom_b];
        float dx = pb.x - pa.x;
        float dy = pb.y - pa.y;
        float dz = pb.z - pa.z;
        wrap_triclinic(&dx, &dy, &dz, c0, c1, c2, r0, r1, r2);
        float dist = sqrtf(dx * dx + dy * dy + dz * dz);
        if (dist < min_val) min_val = dist;
    }
    out[frame * n_sel_a + a] = min_val;
}

__global__ void min_dist_b_triclinic(const Float4* coords,
                                     const unsigned int* sel_a,
                                     const unsigned int* sel_b,
                                     int n_sel_a,
                                     int n_sel_b,
                                     int n_atoms,
                                     int n_frames,
                                     const Float4* cell,
                                     const Float4* inv,
                                     float* out) {
    int frame = blockIdx.y;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || b >= n_sel_b) return;
    unsigned int atom_b = sel_b[b];
    Float4 pb = coords[frame * n_atoms + atom_b];
    Float4 c0 = cell[frame * 3];
    Float4 c1 = cell[frame * 3 + 1];
    Float4 c2 = cell[frame * 3 + 2];
    Float4 r0 = inv[frame * 3];
    Float4 r1 = inv[frame * 3 + 1];
    Float4 r2 = inv[frame * 3 + 2];
    float min_val = 1.0e30f;
    for (int a = 0; a < n_sel_a; a++) {
        unsigned int atom_a = sel_a[a];
        Float4 pa = coords[frame * n_atoms + atom_a];
        float dx = pb.x - pa.x;
        float dy = pb.y - pa.y;
        float dz = pb.z - pa.z;
        wrap_triclinic(&dx, &dy, &dz, c0, c1, c2, r0, r1, r2);
        float dist = sqrtf(dx * dx + dy * dy + dz * dz);
        if (dist < min_val) min_val = dist;
    }
    out[frame * n_sel_b + b] = min_val;
}

__global__ void min_dist_points(const Float4* coords,
                                const unsigned int* sel,
                                int n_sel,
                                const Float4* points,
                                int n_points,
                                int n_atoms,
                                int n_frames,
                                const Float4* cell,
                                const Float4* inv,
                                int image,
                                float* out) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_points) return;
    Float4 p = points[frame * n_points + idx];
    float min_val = 1.0e30f;
    Float4 c0, c1, c2, r0, r1, r2;
    if (image) {
        c0 = cell[frame * 3];
        c1 = cell[frame * 3 + 1];
        c2 = cell[frame * 3 + 2];
        r0 = inv[frame * 3];
        r1 = inv[frame * 3 + 1];
        r2 = inv[frame * 3 + 2];
    }
    for (int i = 0; i < n_sel; i++) {
        unsigned int atom = sel[i];
        Float4 pa = coords[frame * n_atoms + atom];
        float dx = p.x - pa.x;
        float dy = p.y - pa.y;
        float dz = p.z - pa.z;
        if (image) {
            wrap_triclinic(&dx, &dy, &dz, c0, c1, c2, r0, r1, r2);
        }
        float dist = sqrtf(dx * dx + dy * dy + dz * dz);
        if (dist < min_val) min_val = dist;
    }
    out[frame * n_points + idx] = min_val;
}

__global__ void max_dist_points(const Float4* coords,
                                const unsigned int* sel,
                                int n_sel,
                                const Float4* points,
                                int n_atoms,
                                int n_frames,
                                float* out) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_sel) return;
    unsigned int atom = sel[idx];
    Float4 p = coords[frame * n_atoms + atom];
    Float4 c = points[frame];
    float dx = p.x - c.x;
    float dy = p.y - c.y;
    float dz = p.z - c.z;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    atomicMaxFloat(&out[frame], dist);
}

__global__ void multipucker_histogram(const Float4* coords,
                                      const unsigned int* sel,
                                      int n_sel,
                                      const Float4* centers,
                                      int n_atoms,
                                      int n_frames,
                                      int bins,
                                      float range_max,
                                      float* out) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_sel || bins <= 0 || range_max <= 0.0f) return;
    unsigned int atom = sel[idx];
    Float4 p = coords[frame * n_atoms + atom];
    Float4 c = centers[frame];
    float dx = p.x - c.x;
    float dy = p.y - c.y;
    float dz = p.z - c.z;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    if (dist > range_max) return;
    float scale = (float)bins / range_max;
    int bin = (int)floorf(dist * scale);
    if (bin >= bins) bin = bins - 1;
    atomicAdd(&out[frame * bins + bin], 1.0f);
}

__global__ void multipucker_distances(const Float4* coords,
                                      const unsigned int* sel,
                                      int n_sel,
                                      const Float4* centers,
                                      int n_atoms,
                                      int n_frames,
                                      float* distances,
                                      float* frame_max) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_sel) return;
    unsigned int atom = sel[idx];
    Float4 p = coords[frame * n_atoms + atom];
    Float4 c = centers[frame];
    float dx = p.x - c.x;
    float dy = p.y - c.y;
    float dz = p.z - c.z;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    distances[frame * n_sel + idx] = dist;
    atomicMaxFloat(&frame_max[frame], dist);
}

__global__ void multipucker_histogram_from_distances(const float* distances,
                                                     int n_sel,
                                                     int n_frames,
                                                     int bins,
                                                     float range_max,
                                                     float* out) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_sel || bins <= 0 || range_max <= 0.0f) return;
    float dist = distances[frame * n_sel + idx];
    if (dist > range_max) return;
    float scale = (float)bins / range_max;
    int bin = (int)floorf(dist * scale);
    if (bin >= bins) bin = bins - 1;
    atomicAdd(&out[frame * bins + bin], 1.0f);
}

__global__ void multipucker_normalize_rows(float* hist, int n_frames, int bins) {
    int frame = blockIdx.x;
    if (frame >= n_frames || bins <= 0) return;
    if (threadIdx.x != 0) return;
    int row = frame * bins;
    float sum = 0.0f;
    for (int i = 0; i < bins; i++) {
        sum += hist[row + i];
    }
    if (sum <= 0.0f) return;
    float inv = 1.0f / sum;
    for (int i = 0; i < bins; i++) {
        hist[row + i] *= inv;
    }
}

__global__ void hausdorff_reduce(const float* min_a,
                                 const float* min_b,
                                 int n_sel_a,
                                 int n_sel_b,
                                 int n_frames,
                                 float* out) {
    int frame = blockIdx.x;
    int tid = threadIdx.x;
    if (frame >= n_frames) return;
    float max_val = 0.0f;
    for (int i = tid; i < n_sel_a; i += blockDim.x) {
        float v = min_a[frame * n_sel_a + i];
        if (v > max_val) max_val = v;
    }
    for (int j = tid; j < n_sel_b; j += blockDim.x) {
        float v = min_b[frame * n_sel_b + j];
        if (v > max_val) max_val = v;
    }
    extern __shared__ float smax[];
    smax[tid] = max_val;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float other = smax[tid + offset];
            if (other > smax[tid]) smax[tid] = other;
        }
        __syncthreads();
    }
    if (tid == 0) out[frame] = smax[0];
}

__global__ void atom_map_pairs(const Float4* coords,
                               const unsigned int* sel_a,
                               const unsigned int* sel_b,
                               int n_sel_a,
                               int n_sel_b,
                               int n_atoms,
                               int n_frames,
                               const Float4* boxes,
                               float* out) {
    int frame = blockIdx.y;
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || a >= n_sel_a) return;
    unsigned int atom_a = sel_a[a];
    Float4 pa = coords[frame * n_atoms + atom_a];
    Float4 box = boxes[frame];
    float min_val = 1.0e30f;
    unsigned int min_idx = 0;
    for (int b = 0; b < n_sel_b; b++) {
        unsigned int atom_b = sel_b[b];
        Float4 pb = coords[frame * n_atoms + atom_b];
        float dx = pb.x - pa.x;
        float dy = pb.y - pa.y;
        float dz = pb.z - pa.z;
        dx = wrap_pbc(dx, box.x);
        dy = wrap_pbc(dy, box.y);
        dz = wrap_pbc(dz, box.z);
        float dist = sqrtf(dx * dx + dy * dy + dz * dz);
        if (dist < min_val) {
            min_val = dist;
            min_idx = atom_b;
        }
    }
    out[frame * n_sel_a + a] = (n_sel_b == 0) ? -1.0f : (float)min_idx;
}

__global__ void closest_min_dist(const Float4* coords,
                                 const unsigned int* sel_a,
                                 const unsigned int* sel_b,
                                 int n_sel_a,
                                 int n_sel_b,
                                 int n_atoms,
                                 int n_frames,
                                 const Float4* boxes,
                                 float* out) {
    int frame = blockIdx.y;
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || b >= n_sel_b) return;
    unsigned int atom_b = sel_b[b];
    Float4 pb = coords[frame * n_atoms + atom_b];
    Float4 box = boxes[frame];
    float min_val = 1.0e30f;
    for (int a = 0; a < n_sel_a; a++) {
        unsigned int atom_a = sel_a[a];
        Float4 pa = coords[frame * n_atoms + atom_a];
        float dx = pb.x - pa.x;
        float dy = pb.y - pa.y;
        float dz = pb.z - pa.z;
        dx = wrap_pbc(dx, box.x);
        dy = wrap_pbc(dy, box.y);
        dz = wrap_pbc(dz, box.z);
        float dist2 = dx * dx + dy * dy + dz * dz;
        if (dist2 < min_val) min_val = dist2;
    }
    out[frame * n_sel_b + b] = min_val;
}

__global__ void closest_topk(const float* min_dists,
                             const unsigned int* sel_b,
                             int n_sel_b,
                             int n_frames,
                             int k,
                             float* out) {
    int frame = blockIdx.x;
    if (frame >= n_frames) return;
    if (threadIdx.x != 0) return;
    int base = frame * n_sel_b;
    int out_base = frame * k;
    for (int kk = 0; kk < k; kk++) {
        float best = 1.0e30f;
        int best_idx = -1;
        for (int b = 0; b < n_sel_b; b++) {
            float d = min_dists[base + b];
            // skip already selected
            bool used = false;
            for (int prev = 0; prev < kk; prev++) {
                if ((int)out[out_base + prev] == (int)sel_b[b]) {
                    used = true;
                    break;
                }
            }
            if (used) continue;
            if (d < best) {
                best = d;
                best_idx = (int)sel_b[b];
            }
        }
        out[out_base + kk] = (best_idx < 0) ? -1.0f : (float)best_idx;
    }
}

__global__ void rotate_dihedral_apply(const Float4* coords,
                                      const unsigned int* mask,
                                      int n_atoms,
                                      int n_frames,
                                      const Float4* pivots,
                                      const Float4* axes,
                                      const float* angles,
                                      float* out) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_atoms) return;
    int base = (frame * n_atoms + idx) * 3;
    Float4 p = coords[frame * n_atoms + idx];
    if (mask[idx] == 0) {
        out[base] = p.x;
        out[base + 1] = p.y;
        out[base + 2] = p.z;
        return;
    }
    Float4 pivot = pivots[frame];
    Float4 axis = axes[frame];
    float ax = axis.x;
    float ay = axis.y;
    float az = axis.z;
    float norm = sqrtf(ax * ax + ay * ay + az * az);
    if (norm == 0.0f) {
        out[base] = p.x;
        out[base + 1] = p.y;
        out[base + 2] = p.z;
        return;
    }
    ax /= norm;
    ay /= norm;
    az /= norm;
    float vx = p.x - pivot.x;
    float vy = p.y - pivot.y;
    float vz = p.z - pivot.z;
    float cosv = cosf(angles[frame]);
    float sinv = sinf(angles[frame]);
    float dot = ax * vx + ay * vy + az * vz;
    float cx = ay * vz - az * vy;
    float cy = az * vx - ax * vz;
    float cz = ax * vy - ay * vx;
    float rx = vx * cosv + cx * sinv + ax * dot * (1.0f - cosv);
    float ry = vy * cosv + cy * sinv + ay * dot * (1.0f - cosv);
    float rz = vz * cosv + cz * sinv + az * dot * (1.0f - cosv);
    out[base] = pivot.x + rx;
    out[base + 1] = pivot.y + ry;
    out[base + 2] = pivot.z + rz;
}

__global__ void image_coords(const Float4* coords,
                             int n_atoms,
                             int n_frames,
                             const Float4* cell,
                             const Float4* inv,
                             const unsigned int* mask,
                             float* out) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_atoms) return;
    Float4 p = coords[frame * n_atoms + idx];
    int base = (frame * n_atoms + idx) * 3;
    if (mask[idx] == 0) {
        out[base] = p.x;
        out[base + 1] = p.y;
        out[base + 2] = p.z;
        return;
    }
    Float4 c0 = cell[frame * 3];
    Float4 c1 = cell[frame * 3 + 1];
