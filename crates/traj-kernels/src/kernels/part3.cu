    Float4 c2 = cell[frame * 3 + 2];
    Float4 r0 = inv[frame * 3];
    Float4 r1 = inv[frame * 3 + 1];
    Float4 r2 = inv[frame * 3 + 2];
    float x = p.x;
    float y = p.y;
    float z = p.z;
    float f0 = r0.x * x + r1.x * y + r2.x * z;
    float f1 = r0.y * x + r1.y * y + r2.y * z;
    float f2 = r0.z * x + r1.z * y + r2.z * z;
    f0 -= floorf(f0);
    f1 -= floorf(f1);
    f2 -= floorf(f2);
    float nx = f0 * c0.x + f1 * c1.x + f2 * c2.x;
    float ny = f0 * c0.y + f1 * c1.y + f2 * c2.y;
    float nz = f0 * c0.z + f1 * c1.z + f2 * c2.z;
    out[base] = nx;
    out[base + 1] = ny;
    out[base + 2] = nz;
}

__global__ void chirality_volume(const Float4* coms,
                                 int n_frames,
                                 int n_centers,
                                 float* out) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_centers) return;
    int base = frame * (n_centers * 4) + idx * 4;
    Float4 a = coms[base];
    Float4 b = coms[base + 1];
    Float4 c = coms[base + 2];
    Float4 d = coms[base + 3];
    float abx = b.x - a.x;
    float aby = b.y - a.y;
    float abz = b.z - a.z;
    float acx = c.x - a.x;
    float acy = c.y - a.y;
    float acz = c.z - a.z;
    float adx = d.x - a.x;
    float ady = d.y - a.y;
    float adz = d.z - a.z;
    float cx = aby * acz - abz * acy;
    float cy = abz * acx - abx * acz;
    float cz = abx * acy - aby * acx;
    float vol = cx * adx + cy * ady + cz * adz;
    out[frame * n_centers + idx] = vol;
}

__global__ void torsion_diffusion_counts(const Float4* coords,
                                         const unsigned int* torsions,
                                         int n_torsions,
                                         int n_atoms,
                                         int n_frames,
                                         unsigned int* counts) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_torsions) return;
    int base = idx * 4;
    unsigned int a_idx = torsions[base];
    unsigned int b_idx = torsions[base + 1];
    unsigned int c_idx = torsions[base + 2];
    unsigned int d_idx = torsions[base + 3];
    Float4 a = coords[frame * n_atoms + a_idx];
    Float4 b = coords[frame * n_atoms + b_idx];
    Float4 c = coords[frame * n_atoms + c_idx];
    Float4 d = coords[frame * n_atoms + d_idx];
    float b0x = a.x - b.x;
    float b0y = a.y - b.y;
    float b0z = a.z - b.z;
    float b1x = c.x - b.x;
    float b1y = c.y - b.y;
    float b1z = c.z - b.z;
    float b2x = d.x - c.x;
    float b2y = d.y - c.y;
    float b2z = d.z - c.z;
    float deg = dihedral_from_vectors(b0x, b0y, b0z,
                                      b1x, b1y, b1z,
                                      b2x, b2y, b2z,
                                      1, 0);
    int cat = 0;
    if (deg > 150.0f || deg < -150.0f) {
        cat = 0;
    } else if (fabsf(deg) < 30.0f) {
        cat = 1;
    } else if (deg > 0.0f) {
        cat = 2;
    } else {
        cat = 3;
    }
    atomicAdd(&counts[frame * 4 + cat], 1);
}

__global__ void bbox_minmax(const Float4* coords,
                            const unsigned int* sel,
                            int n_sel,
                            int n_atoms,
                            int n_frames,
                            float* min_x,
                            float* min_y,
                            float* min_z,
                            float* max_x,
                            float* max_y,
                            float* max_z) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_sel) return;
    unsigned int atom = sel[idx];
    Float4 p = coords[frame * n_atoms + atom];
    atomicMinFloat(&min_x[frame], p.x);
    atomicMinFloat(&min_y[frame], p.y);
    atomicMinFloat(&min_z[frame], p.z);
    atomicMaxFloat(&max_x[frame], p.x);
    atomicMaxFloat(&max_y[frame], p.y);
    atomicMaxFloat(&max_z[frame], p.z);
}

__global__ void bbox_area(const float* min_x,
                          const float* min_y,
                          const float* min_z,
                          const float* max_x,
                          const float* max_y,
                          const float* max_z,
                          int n_frames,
                          float* out) {
    int frame = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames) return;
    float dx = max_x[frame] - min_x[frame];
    float dy = max_y[frame] - min_y[frame];
    float dz = max_z[frame] - min_z[frame];
    if (dx < 0.0f) dx = 0.0f;
    if (dy < 0.0f) dy = 0.0f;
    if (dz < 0.0f) dz = 0.0f;
    out[frame] = 2.0f * (dx * dy + dy * dz + dx * dz);
}

__global__ void sasa_approx(const Float4* coords,
                            const unsigned int* sel,
                            const float* radii,
                            const Float4* sphere_points,
                            int n_sel,
                            int n_points,
                            int n_atoms,
                            int n_frames,
                            float* out) {
    long long tid = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    long long stride = (long long)blockDim.x * (long long)gridDim.x;
    long long total = (long long)n_frames * (long long)n_sel * (long long)n_points;
    const float four_pi = 12.566370614359172f;
    for (long long t = tid; t < total; t += stride) {
        int point_idx = (int)(t % n_points);
        long long tmp = t / n_points;
        int atom_idx = (int)(tmp % n_sel);
        int frame = (int)(tmp / n_sel);
        float r_i = radii[atom_idx];
        if (r_i <= 0.0f) continue;
        unsigned int atom_i = sel[atom_idx];
        Float4 center_i = coords[frame * n_atoms + atom_i];
        Float4 dir = sphere_points[point_idx];
        float px = center_i.x + dir.x * r_i;
        float py = center_i.y + dir.y * r_i;
        float pz = center_i.z + dir.z * r_i;
        bool occluded = false;
        for (int j = 0; j < n_sel; j++) {
            if (j == atom_idx) continue;
            float r_j = radii[j];
            if (r_j <= 0.0f) continue;
            unsigned int atom_j = sel[j];
            Float4 center_j = coords[frame * n_atoms + atom_j];
            float dx = px - center_j.x;
            float dy = py - center_j.y;
            float dz = pz - center_j.z;
            if (dx * dx + dy * dy + dz * dz < r_j * r_j) {
                occluded = true;
                break;
            }
        }
        if (!occluded) {
            float area = (four_pi * r_i * r_i) / (float)n_points;
            atomicAdd(&out[frame], area);
        }
    }
}

__global__ void volume_orthorhombic(const Float4* boxes,
                                    int n_frames,
                                    float* out) {
    int frame = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames) return;
    Float4 box = boxes[frame];
    out[frame] = box.x * box.y * box.z;
}

__global__ void volume_cell(const Float4* cell,
                            int n_frames,
                            float* out) {
    int frame = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames) return;
    Float4 c0 = cell[frame * 3];
    Float4 c1 = cell[frame * 3 + 1];
    Float4 c2 = cell[frame * 3 + 2];
    float det = c0.x * (c1.y * c2.z - c1.z * c2.y)
        - c0.y * (c1.x * c2.z - c1.z * c2.x)
        + c0.z * (c1.x * c2.y - c1.y * c2.x);
    out[frame] = fabsf(det);
}

__global__ void shift_coords(const Float4* coords,
                             int n_atoms,
                             int n_frames,
                             const Float4* shifts,
                             float* out) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_atoms) return;
    Float4 p = coords[frame * n_atoms + idx];
    Float4 s = shifts[frame];
    int base = (frame * n_atoms + idx) * 3;
    out[base] = p.x + s.x;
    out[base + 1] = p.y + s.y;
    out[base + 2] = p.z + s.z;
}

__global__ void translate_coords(const Float4* coords,
                                 int n_atoms,
                                 int n_frames,
                                 float dx,
                                 float dy,
                                 float dz,
                                 float* out) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_atoms) return;
    Float4 p = coords[frame * n_atoms + idx];
    int base = (frame * n_atoms + idx) * 3;
    out[base] = p.x + dx;
    out[base + 1] = p.y + dy;
    out[base + 2] = p.z + dz;
}

__global__ void scale_coords(const Float4* coords,
                             int n_atoms,
                             int n_frames,
                             float scale,
                             float* out) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_atoms) return;
    Float4 p = coords[frame * n_atoms + idx];
    int base = (frame * n_atoms + idx) * 3;
    out[base] = p.x * scale;
    out[base + 1] = p.y * scale;
    out[base + 2] = p.z * scale;
}

__global__ void transform_coords(const Float4* coords,
                                 int n_atoms,
                                 int n_frames,
                                 const float* rot,
                                 const float* trans,
                                 float* out) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_atoms) return;
    Float4 p = coords[frame * n_atoms + idx];
    float x = p.x;
    float y = p.y;
    float z = p.z;
    float rx = rot[0] * x + rot[1] * y + rot[2] * z + trans[0];
    float ry = rot[3] * x + rot[4] * y + rot[5] * z + trans[1];
    float rz = rot[6] * x + rot[7] * y + rot[8] * z + trans[2];
    int base = (frame * n_atoms + idx) * 3;
    out[base] = rx;
    out[base + 1] = ry;
    out[base + 2] = rz;
}

__global__ void randomize_ions_apply(const Float4* coords,
                                     const unsigned int* mask,
                                     const float* rand_vals,
                                     int n_atoms,
                                     int n_frames,
                                     const Float4* cell,
                                     float* out) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_atoms) return;
    Float4 p = coords[frame * n_atoms + idx];
    Float4 a = cell[frame * 3];
    Float4 b = cell[frame * 3 + 1];
    Float4 c = cell[frame * 3 + 2];
    int base = (frame * n_atoms + idx) * 3;
    if (mask[idx] == 0) {
        out[base] = p.x;
        out[base + 1] = p.y;
        out[base + 2] = p.z;
    } else {
        float rx = rand_vals[base];
        float ry = rand_vals[base + 1];
        float rz = rand_vals[base + 2];
        out[base] = rx * a.x + ry * b.x + rz * c.x;
        out[base + 1] = rx * a.y + ry * b.y + rz * c.y;
        out[base + 2] = rx * a.z + ry * b.z + rz * c.z;
    }
}

__global__ void native_contacts_count(const Float4* coords,
                                      const unsigned int* pairs,
                                      int n_pairs,
                                      int n_atoms,
                                      int n_frames,
                                      const Float4* boxes,
                                      float cutoff,
                                      unsigned int* counts) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_pairs) return;
    unsigned int atom_a = pairs[idx * 2];
    unsigned int atom_b = pairs[idx * 2 + 1];
    Float4 pa = coords[frame * n_atoms + atom_a];
    Float4 pb = coords[frame * n_atoms + atom_b];
    Float4 box = boxes[frame];
    float dx = pb.x - pa.x;
    float dy = pb.y - pa.y;
    float dz = pb.z - pa.z;
    dx = wrap_pbc(dx, box.x);
    dy = wrap_pbc(dy, box.y);
    dz = wrap_pbc(dz, box.z);
    float dist2 = dx * dx + dy * dy + dz * dz;
    if (dist2 <= cutoff * cutoff) {
        atomicAdd(&counts[frame], 1);
    }
}

__global__ void gather_selection(const Float4* coords,
                                 const unsigned int* sel,
                                 int n_sel,
                                 int n_atoms,
                                 int n_frames,
                                 float* out) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_sel) return;
    unsigned int atom = sel[idx];
    Float4 p = coords[frame * n_atoms + atom];
    int base = (frame * n_sel + idx) * 3;
    out[base] = p.x;
    out[base + 1] = p.y;
    out[base + 2] = p.z;
}

__global__ void replicate_cell(const Float4* coords,
                               const unsigned int* sel,
                               int n_sel,
                               int n_atoms,
                               int n_frames,
                               const Float4* cell,
                               int rx,
                               int ry,
                               int rz,
                               float* out) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames) return;
    int reps = rx * ry * rz;
    int total = n_sel * reps;
    if (idx >= total) return;
    int rep = idx / n_sel;
    int sel_idx = idx - rep * n_sel;
    int iz = rep % rz;
    int iy = (rep / rz) % ry;
    int ix = rep / (ry * rz);
    Float4 a = cell[frame * 3];
    Float4 b = cell[frame * 3 + 1];
    Float4 c = cell[frame * 3 + 2];
    float sx = (float)ix * a.x + (float)iy * b.x + (float)iz * c.x;
    float sy = (float)ix * a.y + (float)iy * b.y + (float)iz * c.y;
    float sz = (float)ix * a.z + (float)iy * b.z + (float)iz * c.z;
    unsigned int atom = sel[sel_idx];
    Float4 p = coords[frame * n_atoms + atom];
    int out_idx = (frame * total + idx) * 3;
    out[out_idx] = p.x + sx;
    out[out_idx + 1] = p.y + sy;
    out[out_idx + 2] = p.z + sz;
}

__global__ void rmsf_accum(const Float4* coords,
                           const unsigned int* sel,
                           int n_sel,
                           int n_atoms,
                           int n_frames,
                           float* sum_x,
                           float* sum_y,
                           float* sum_z,
                           float* sum_sq) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_sel) return;
    unsigned int atom = sel[idx];
    Float4 p = coords[frame * n_atoms + atom];
    atomicAdd(&sum_x[idx], p.x);
    atomicAdd(&sum_y[idx], p.y);
    atomicAdd(&sum_z[idx], p.z);
    atomicAdd(&sum_sq[idx], p.x * p.x + p.y * p.y + p.z * p.z);
}

__global__ void mean_structure_accum(const Float4* coords,
                                     const unsigned int* sel,
                                     int n_sel,
                                     int n_atoms,
                                     int n_frames,
                                     float* sum_x,
                                     float* sum_y,
                                     float* sum_z) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_sel) return;
    unsigned int atom = sel[idx];
    Float4 p = coords[frame * n_atoms + atom];
    atomicAdd(&sum_x[idx], p.x);
    atomicAdd(&sum_y[idx], p.y);
    atomicAdd(&sum_z[idx], p.z);
}

__global__ void align_centroid(const Float4* coords,
                               const Float4* ref,
                               const unsigned int* sel,
                               const float* weights,
                               int n_sel,
                               int n_atoms,
                               int n_frames,
                               float* sum_x,
                               float* sum_y,
                               float* sum_w) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_sel) return;
    unsigned int atom = sel[idx];
    float w = weights[idx];
    Float4 p = coords[frame * n_atoms + atom];
    Float4 r = ref[idx];
    atomicAdd(&sum_x[frame * 3 + 0], p.x * w);
    atomicAdd(&sum_x[frame * 3 + 1], p.y * w);
    atomicAdd(&sum_x[frame * 3 + 2], p.z * w);
    atomicAdd(&sum_y[frame * 3 + 0], r.x * w);
    atomicAdd(&sum_y[frame * 3 + 1], r.y * w);
    atomicAdd(&sum_y[frame * 3 + 2], r.z * w);
    atomicAdd(&sum_w[frame], w);
}

__global__ void align_cov(const Float4* coords,
                          const Float4* ref,
                          const unsigned int* sel,
                          const float* weights,
                          int n_sel,
                          int n_atoms,
                          int n_frames,
                          const float* sum_x,
                          const float* sum_y,
                          const float* sum_w,
                          float* cov) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_sel) return;
    float wsum = sum_w[frame];
    if (wsum == 0.0f) return;
    float cx = sum_x[frame * 3 + 0] / wsum;
    float cy = sum_x[frame * 3 + 1] / wsum;
    float cz = sum_x[frame * 3 + 2] / wsum;
    float rx = sum_y[frame * 3 + 0] / wsum;
    float ry = sum_y[frame * 3 + 1] / wsum;
    float rz = sum_y[frame * 3 + 2] / wsum;
    unsigned int atom = sel[idx];
    float w = weights[idx];
    Float4 p = coords[frame * n_atoms + atom];
    Float4 r = ref[idx];
    float x0 = p.x - cx;
    float x1 = p.y - cy;
    float x2 = p.z - cz;
    float y0 = r.x - rx;
    float y1 = r.y - ry;
    float y2 = r.z - rz;
    atomicAdd(&cov[frame * 9 + 0], w * x0 * y0);
    atomicAdd(&cov[frame * 9 + 1], w * x0 * y1);
    atomicAdd(&cov[frame * 9 + 2], w * x0 * y2);
    atomicAdd(&cov[frame * 9 + 3], w * x1 * y0);
    atomicAdd(&cov[frame * 9 + 4], w * x1 * y1);
    atomicAdd(&cov[frame * 9 + 5], w * x1 * y2);
    atomicAdd(&cov[frame * 9 + 6], w * x2 * y0);
    atomicAdd(&cov[frame * 9 + 7], w * x2 * y1);
    atomicAdd(&cov[frame * 9 + 8], w * x2 * y2);
}

__global__ void inertia_tensor(const Float4* coords,
                               const unsigned int* sel,
                               const float* weights,
                               int n_sel,
                               int n_atoms,
                               int n_frames,
                               const Float4* centers,
                               float* out) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_sel) return;
    unsigned int atom = sel[idx];
    float w = weights[idx];
    Float4 p = coords[frame * n_atoms + atom];
    Float4 c = centers[frame];
    float x = p.x - c.x;
    float y = p.y - c.y;
    float z = p.z - c.z;
    int base = frame * 6;
    atomicAdd(&out[base + 0], w * (y * y + z * z));
    atomicAdd(&out[base + 1], w * (x * x + z * z));
    atomicAdd(&out[base + 2], w * (x * x + y * y));
    atomicAdd(&out[base + 3], -w * x * y);
    atomicAdd(&out[base + 4], -w * x * z);
    atomicAdd(&out[base + 5], -w * y * z);
}

__global__ void rmsd_per_res_accum(const Float4* coords,
                                   const Float4* ref,
                                   const unsigned int* group_offsets,
                                   const unsigned int* group_indices,
                                   int n_groups,
                                   int n_atoms,
                                   int n_frames,
                                   int max_len,
                                   const float* rotations,
                                   const float* cx,
                                   const float* cy,
                                   float* out) {
    int atom_in_group = blockIdx.x * blockDim.x + threadIdx.x;
    int frame = blockIdx.y;
    int group = blockIdx.z;
    if (frame >= n_frames || group >= n_groups) return;
    unsigned int start = group_offsets[group];
    unsigned int end = group_offsets[group + 1];
    unsigned int len = end - start;
    if (atom_in_group >= len) return;
    unsigned int atom = group_indices[start + atom_in_group];
    Float4 p = coords[frame * n_atoms + atom];
    Float4 r = ref[atom];
    int rbase = frame * 9;
    float r00 = rotations[rbase + 0];
    float r01 = rotations[rbase + 1];
    float r02 = rotations[rbase + 2];
    float r10 = rotations[rbase + 3];
