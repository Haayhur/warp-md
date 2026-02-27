    float r11 = rotations[rbase + 4];
    float r12 = rotations[rbase + 5];
    float r20 = rotations[rbase + 6];
    float r21 = rotations[rbase + 7];
    float r22 = rotations[rbase + 8];
    int cbase = frame * 3;
    float cx0 = cx[cbase + 0];
    float cx1 = cx[cbase + 1];
    float cx2 = cx[cbase + 2];
    float cy0 = cy[cbase + 0];
    float cy1 = cy[cbase + 1];
    float cy2 = cy[cbase + 2];
    float x0 = p.x - cx0;
    float x1 = p.y - cx1;
    float x2 = p.z - cx2;
    float ax = r00 * x0 + r01 * x1 + r02 * x2;
    float ay = r10 * x0 + r11 * x1 + r12 * x2;
    float az = r20 * x0 + r21 * x1 + r22 * x2;
    float y0 = r.x - cy0;
    float y1 = r.y - cy1;
    float y2 = r.z - cy2;
    float dx = ax - y0;
    float dy = ay - y1;
    float dz = az - y2;
    int out_idx = frame * n_groups + group;
    atomicAdd(&out[out_idx], dx * dx + dy * dy + dz * dz);
}

__global__ void rdf_hist(const Float4* coords,
                         const unsigned int* sel_a,
                         const unsigned int* sel_b,
                         int n_a,
                         int n_b,
                         int n_atoms,
                         int n_frames,
                         float r_max,
                         float bin_width,
                         int pbc,
                         const float* box_l,
                         unsigned long long* counts,
                         int same_sel) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames) return;
    long total = (long)n_a * (long)n_b;
    if (idx >= total) return;
    int ia = idx / n_b;
    int ib = idx - ia * n_b;
    if (same_sel && ia == ib) return;
    unsigned int atom_a = sel_a[ia];
    unsigned int atom_b = sel_b[ib];
    Float4 pa = coords[frame * n_atoms + atom_a];
    Float4 pb = coords[frame * n_atoms + atom_b];
    float dx = pb.x - pa.x;
    float dy = pb.y - pa.y;
    float dz = pb.z - pa.z;
    if (pbc) {
        float lx = box_l[frame * 3 + 0];
        float ly = box_l[frame * 3 + 1];
        float lz = box_l[frame * 3 + 2];
        dx -= roundf(dx / lx) * lx;
        dy -= roundf(dy / ly) * ly;
        dz -= roundf(dz / lz) * lz;
    }
    float r = sqrtf(dx * dx + dy * dy + dz * dz);
    if (r < r_max) {
        int bin = (int)(r / bin_width);
        atomicAdd(&counts[bin], 1ULL);
    }
}

__global__ void polymer_end_to_end(const Float4* coords,
                                   const unsigned int* chain_offsets,
                                   const unsigned int* chain_indices,
                                   int n_chains,
                                   int n_atoms,
                                   int n_frames,
                                   float* out) {
    int frame = blockIdx.y;
    int chain = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || chain >= n_chains) return;
    unsigned int start = chain_offsets[chain];
    unsigned int end = chain_offsets[chain + 1];
    if (end <= start) {
        out[frame * n_chains + chain] = 0.0f;
        return;
    }
    unsigned int first = chain_indices[start];
    unsigned int last = chain_indices[end - 1];
    Float4 p = coords[frame * n_atoms + first];
    Float4 q = coords[frame * n_atoms + last];
    float dx = q.x - p.x;
    float dy = q.y - p.y;
    float dz = q.z - p.z;
    out[frame * n_chains + chain] = sqrtf(dx * dx + dy * dy + dz * dz);
}

__global__ void polymer_contour_length(const Float4* coords,
                                       const unsigned int* chain_offsets,
                                       const unsigned int* chain_indices,
                                       int n_chains,
                                       int n_atoms,
                                       int n_frames,
                                       float* out) {
    int frame = blockIdx.y;
    int chain = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || chain >= n_chains) return;
    unsigned int start = chain_offsets[chain];
    unsigned int end = chain_offsets[chain + 1];
    float sum = 0.0f;
    if (end > start + 1) {
        for (unsigned int i = start; i + 1 < end; ++i) {
            unsigned int a = chain_indices[i];
            unsigned int b = chain_indices[i + 1];
            Float4 pa = coords[frame * n_atoms + a];
            Float4 pb = coords[frame * n_atoms + b];
            float dx = pb.x - pa.x;
            float dy = pb.y - pa.y;
            float dz = pb.z - pa.z;
            sum += sqrtf(dx * dx + dy * dy + dz * dz);
        }
    }
    out[frame * n_chains + chain] = sum;
}

__global__ void polymer_chain_rg(const Float4* coords,
                                 const unsigned int* chain_offsets,
                                 const unsigned int* chain_indices,
                                 int n_chains,
                                 int n_atoms,
                                 int n_frames,
                                 float* out) {
    int frame = blockIdx.y;
    int chain = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || chain >= n_chains) return;
    unsigned int start = chain_offsets[chain];
    unsigned int end = chain_offsets[chain + 1];
    unsigned int n = (end > start) ? (end - start) : 0;
    if (n == 0) {
        out[frame * n_chains + chain] = 0.0f;
        return;
    }
    float cx = 0.0f;
    float cy = 0.0f;
    float cz = 0.0f;
    for (unsigned int i = start; i < end; ++i) {
        unsigned int idx = chain_indices[i];
        Float4 p = coords[frame * n_atoms + idx];
        cx += p.x;
        cy += p.y;
        cz += p.z;
    }
    float inv = 1.0f / (float)n;
    cx *= inv;
    cy *= inv;
    cz *= inv;
    float sum = 0.0f;
    for (unsigned int i = start; i < end; ++i) {
        unsigned int idx = chain_indices[i];
        Float4 p = coords[frame * n_atoms + idx];
        float dx = p.x - cx;
        float dy = p.y - cy;
        float dz = p.z - cz;
        sum += dx * dx + dy * dy + dz * dz;
    }
    out[frame * n_chains + chain] = sqrtf(sum * inv);
}

__global__ void polymer_bond_hist(const Float4* coords,
                                  const unsigned int* bond_pairs,
                                  int n_bonds,
                                  int n_atoms,
                                  int n_frames,
                                  float r_max,
                                  float bin_width,
                                  int bins,
                                  unsigned long long* counts) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_bonds) return;
    unsigned int a = bond_pairs[idx * 2];
    unsigned int b = bond_pairs[idx * 2 + 1];
    Float4 pa = coords[frame * n_atoms + a];
    Float4 pb = coords[frame * n_atoms + b];
    float dx = pb.x - pa.x;
    float dy = pb.y - pa.y;
    float dz = pb.z - pa.z;
    float r = sqrtf(dx * dx + dy * dy + dz * dz);
    if (r < r_max) {
        int bin = (int)(r / bin_width);
        if (bin < bins) {
            atomicAdd(&counts[bin], 1ULL);
        }
    }
}

__global__ void polymer_angle_hist(const Float4* coords,
                                   const unsigned int* angle_triplets,
                                   int n_angles,
                                   int n_atoms,
                                   int n_frames,
                                   float max_angle,
                                   float bin_width,
                                   int bins,
                                   int degrees,
                                   unsigned long long* counts) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_angles) return;
    unsigned int a = angle_triplets[idx * 3];
    unsigned int b = angle_triplets[idx * 3 + 1];
    unsigned int c = angle_triplets[idx * 3 + 2];
    Float4 pa = coords[frame * n_atoms + a];
    Float4 pb = coords[frame * n_atoms + b];
    Float4 pc = coords[frame * n_atoms + c];
    float v1x = pa.x - pb.x;
    float v1y = pa.y - pb.y;
    float v1z = pa.z - pb.z;
    float v2x = pc.x - pb.x;
    float v2y = pc.y - pb.y;
    float v2z = pc.z - pb.z;
    float n1 = sqrtf(v1x * v1x + v1y * v1y + v1z * v1z);
    float n2 = sqrtf(v2x * v2x + v2y * v2y + v2z * v2z);
    if (n1 == 0.0f || n2 == 0.0f) return;
    float dot = (v1x * v2x + v1y * v2y + v1z * v2z) / (n1 * n2);
    dot = fminf(1.0f, fmaxf(-1.0f, dot));
    float angle = acosf(dot);
    if (degrees) {
        angle *= 57.29577951308232f;
    }
    if (angle < max_angle) {
        int bin = (int)(angle / bin_width);
        if (bin < bins) {
            atomicAdd(&counts[bin], 1ULL);
        }
    }
}
}

extern "C" {
__global__ void group_com_accum(const Float4* coords,
                                const unsigned int* group_offsets,
                                const unsigned int* group_indices,
                                const float* masses,
                                int n_groups,
                                int n_atoms,
                                int n_frames,
                                int max_len,
                                float* sum_x,
                                float* sum_y,
                                float* sum_z,
                                float* mass_sum) {
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
    float m = masses[atom];
    int idx = frame * n_groups + group;
    atomicAdd(&sum_x[idx], p.x * m);
    atomicAdd(&sum_y[idx], p.y * m);
    atomicAdd(&sum_z[idx], p.z * m);
    atomicAdd(&mass_sum[idx], m);
}

__global__ void group_com_finalize(const float* sum_x,
                                   const float* sum_y,
                                   const float* sum_z,
                                   const float* mass_sum,
                                   int n_groups,
                                   int n_frames,
                                   float scale,
                                   Float4* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_groups * n_frames;
    if (idx >= total) return;
    float m = mass_sum[idx];
    if (m == 0.0f) {
        out[idx].x = 0.0f;
        out[idx].y = 0.0f;
        out[idx].z = 0.0f;
        out[idx].w = 1.0f;
        return;
    }
    out[idx].x = sum_x[idx] / m * scale;
    out[idx].y = sum_y[idx] / m * scale;
    out[idx].z = sum_z[idx] / m * scale;
    out[idx].w = 1.0f;
}

__global__ void group_dipole_accum(const Float4* coords,
                                   const unsigned int* group_offsets,
                                   const unsigned int* group_indices,
                                   const float* charges,
                                   int n_groups,
                                   int n_atoms,
                                   int n_frames,
                                   int max_len,
                                   float* sum_x,
                                   float* sum_y,
                                   float* sum_z) {
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
    float q = charges[atom];
    int idx = frame * n_groups + group;
    atomicAdd(&sum_x[idx], p.x * q);
    atomicAdd(&sum_y[idx], p.y * q);
    atomicAdd(&sum_z[idx], p.z * q);
}

__global__ void group_dipole_finalize(const float* sum_x,
                                      const float* sum_y,
                                      const float* sum_z,
                                      int n_groups,
                                      int n_frames,
                                      float scale,
                                      Float4* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_groups * n_frames;
    if (idx >= total) return;
    out[idx].x = sum_x[idx] * scale;
    out[idx].y = sum_y[idx] * scale;
    out[idx].z = sum_z[idx] * scale;
    out[idx].w = 0.0f;
}

__global__ void group_ke_accum(const Float4* coords,
                               const unsigned int* group_offsets,
                               const unsigned int* group_indices,
                               const float* masses,
                               int n_groups,
                               int n_atoms,
                               int n_frames,
                               int max_len,
                               float vel_scale,
                               float* sum_ke) {
    int atom_in_group = blockIdx.x * blockDim.x + threadIdx.x;
    int frame = blockIdx.y;
    int group = blockIdx.z;
    if (frame >= n_frames || group >= n_groups) return;
    unsigned int start = group_offsets[group];
    unsigned int end = group_offsets[group + 1];
    unsigned int len = end - start;
    if (atom_in_group >= len) return;
    unsigned int atom = group_indices[start + atom_in_group];
    Float4 v = coords[frame * n_atoms + atom];
    float vx = v.x * vel_scale;
    float vy = v.y * vel_scale;
    float vz = v.z * vel_scale;
    float m = masses[atom];
    float ke = 0.5f * m * (vx * vx + vy * vy + vz * vz);
    int idx = frame * n_groups + group;
    atomicAdd(&sum_ke[idx], ke);
}

__global__ void orientation_plane(const Float4* coords,
                                  const unsigned int* anchors,
                                  int n_groups,
                                  int n_atoms,
                                  int n_frames,
                                  float scale,
                                  Float4* out) {
    int group = blockIdx.x * blockDim.x + threadIdx.x;
    int frame = blockIdx.y;
    if (frame >= n_frames || group >= n_groups) return;
    unsigned int a = anchors[group * 3 + 0];
    unsigned int b = anchors[group * 3 + 1];
    unsigned int c = anchors[group * 3 + 2];
    Float4 pa = coords[frame * n_atoms + a];
    Float4 pb = coords[frame * n_atoms + b];
    Float4 pc = coords[frame * n_atoms + c];
    float v1x = (pa.x - pb.x) * scale;
    float v1y = (pa.y - pb.y) * scale;
    float v1z = (pa.z - pb.z) * scale;
    float v2x = (pa.x - pc.x) * scale;
    float v2y = (pa.y - pc.y) * scale;
    float v2z = (pa.z - pc.z) * scale;
    float cx = v1y * v2z - v1z * v2y;
    float cy = v1z * v2x - v1x * v2z;
    float cz = v1x * v2y - v1y * v2x;
    float norm = sqrtf(cx * cx + cy * cy + cz * cz);
    int idx = frame * n_groups + group;
    if (norm == 0.0f) {
        out[idx].x = 0.0f;
        out[idx].y = 0.0f;
        out[idx].z = 0.0f;
        out[idx].w = 0.0f;
    } else {
        out[idx].x = cx / norm;
        out[idx].y = cy / norm;
        out[idx].z = cz / norm;
        out[idx].w = 0.0f;
    }
}

__global__ void orientation_vector(const Float4* coords,
                                   const unsigned int* anchors,
                                   int n_groups,
                                   int n_atoms,
                                   int n_frames,
                                   float scale,
                                   Float4* out) {
    int group = blockIdx.x * blockDim.x + threadIdx.x;
    int frame = blockIdx.y;
    if (frame >= n_frames || group >= n_groups) return;
    unsigned int a = anchors[group * 3 + 0];
    unsigned int b = anchors[group * 3 + 1];
    Float4 pa = coords[frame * n_atoms + a];
    Float4 pb = coords[frame * n_atoms + b];
    float vx = (pb.x - pa.x) * scale;
    float vy = (pb.y - pa.y) * scale;
    float vz = (pb.z - pa.z) * scale;
    float norm = sqrtf(vx * vx + vy * vy + vz * vz);
    int idx = frame * n_groups + group;
    if (norm == 0.0f) {
        out[idx].x = 0.0f;
        out[idx].y = 0.0f;
        out[idx].z = 0.0f;
        out[idx].w = 0.0f;
    } else {
        out[idx].x = vx / norm;
        out[idx].y = vy / norm;
        out[idx].z = vz / norm;
        out[idx].w = 0.0f;
    }
}

__global__ void orientation_vector_pbc(const Float4* coords,
                                       const unsigned int* anchors,
                                       int n_groups,
                                       int n_atoms,
                                       int n_frames,
                                       const Float4* boxes,
                                       float scale,
                                       Float4* out) {
    int group = blockIdx.x * blockDim.x + threadIdx.x;
    int frame = blockIdx.y;
    if (frame >= n_frames || group >= n_groups) return;
    unsigned int a = anchors[group * 3 + 0];
    unsigned int b = anchors[group * 3 + 1];
    Float4 pa = coords[frame * n_atoms + a];
    Float4 pb = coords[frame * n_atoms + b];
    Float4 box = boxes[frame];
    float vx = (pb.x - pa.x) * scale;
    float vy = (pb.y - pa.y) * scale;
    float vz = (pb.z - pa.z) * scale;
    vx = wrap_pbc(vx, box.x);
    vy = wrap_pbc(vy, box.y);
    vz = wrap_pbc(vz, box.z);
    float norm = sqrtf(vx * vx + vy * vy + vz * vz);
    int idx = frame * n_groups + group;
    if (norm == 0.0f) {
        out[idx].x = 0.0f;
        out[idx].y = 0.0f;
        out[idx].z = 0.0f;
        out[idx].w = 0.0f;
    } else {
        out[idx].x = vx / norm;
        out[idx].y = vy / norm;
        out[idx].z = vz / norm;
        out[idx].w = 0.0f;
    }
}

__global__ void water_count(const Float4* coords,
                            const unsigned int* sel,
                            int n_sel,
                            int n_atoms,
                            int frame,
                            float center_x,
                            float center_y,
                            float center_z,
                            float box_x,
                            float box_y,
                            float box_z,
                            float region_x,
                            float region_y,
                            float region_z,
                            int dim_x,
                            int dim_y,
                            int dim_z,
                            float scale,
                            unsigned int* counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_sel) return;
    unsigned int atom = sel[idx];
    Float4 p = coords[frame * n_atoms + atom];
    float x = p.x * scale - center_x;
    float y = p.y * scale - center_y;
    float z = p.z * scale - center_z;
    if (x < 0.0f || y < 0.0f || z < 0.0f) return;
    if (x > region_x || y > region_y || z > region_z) return;
    int ix = (int)floorf(x / box_x);
    int iy = (int)floorf(y / box_y);
    int iz = (int)floorf(z / box_z);
    if (ix < 0 || iy < 0 || iz < 0) return;
    if (ix >= dim_x || iy >= dim_y || iz >= dim_z) return;
    int flat = ix + dim_x * (iy + dim_y * iz);
    atomicAdd(&counts[flat], 1);
}

__device__ __forceinline__ int gist_find_exception(unsigned int atom_i,
                                                   unsigned int atom_j,
                                                   const unsigned int* ex_i,
                                                   const unsigned int* ex_j,
                                                   int n_ex) {
    unsigned int lo = atom_i < atom_j ? atom_i : atom_j;
    unsigned int hi = atom_i < atom_j ? atom_j : atom_i;
    for (int k = 0; k < n_ex; ++k) {
        unsigned int ei = ex_i[k];
        unsigned int ej = ex_j[k];
        unsigned int elo = ei < ej ? ei : ej;
        unsigned int ehi = ei < ej ? ej : ei;
        if (elo == lo && ehi == hi) {
            return k;
        }
    }
    return -1;
}

__device__ __forceinline__ float gist_pair_energy(const Float4* coords,
                                                  int frame_base,
                                                  unsigned int atom_i,
                                                  unsigned int atom_j,
                                                  const float* charges,
                                                  const float* sigmas,
                                                  const float* epsilons,
                                                  const unsigned int* ex_i,
                                                  const unsigned int* ex_j,
                                                  const float* ex_qprod,
                                                  const float* ex_sigma,
                                                  const float* ex_epsilon,
                                                  int n_ex,
                                                  int pbc_mode,
                                                  float lx,
                                                  float ly,
                                                  float lz,
                                                  float c00,
                                                  float c01,
                                                  float c02,
                                                  float c10,
                                                  float c11,
                                                  float c12,
                                                  float c20,
                                                  float c21,
                                                  float c22,
                                                  float r00,
                                                  float r01,
                                                  float r02,
                                                  float r10,
                                                  float r11,
                                                  float r12,
                                                  float r20,
                                                  float r21,
                                                  float r22,
                                                  float cutoff,
                                                  float length_scale) {
    if (atom_i == atom_j) {
        return 0.0f;
    }
    Float4 pi = coords[frame_base + atom_i];
    Float4 pj = coords[frame_base + atom_j];
    float dx = (pi.x - pj.x) * length_scale;
    float dy = (pi.y - pj.y) * length_scale;
    float dz = (pi.z - pj.z) * length_scale;
    if (pbc_mode == 1) {
        dx = wrap_pbc(dx, lx);
        dy = wrap_pbc(dy, ly);
        dz = wrap_pbc(dz, lz);
    } else if (pbc_mode == 2) {
        Float4 c0 = {c00, c01, c02, 0.0f};
        Float4 c1 = {c10, c11, c12, 0.0f};
        Float4 c2 = {c20, c21, c22, 0.0f};
        Float4 r0 = {r00, r01, r02, 0.0f};
        Float4 r1 = {r10, r11, r12, 0.0f};
        Float4 r2 = {r20, r21, r22, 0.0f};
        wrap_triclinic(&dx, &dy, &dz, c0, c1, c2, r0, r1, r2);
    }
    float r2 = dx * dx + dy * dy + dz * dz;
    if (r2 <= 0.0f) {
        return 0.0f;
    }
    float r = sqrtf(r2);
    if (r > cutoff) {
        return 0.0f;
    }

    const float coulomb_const = 138.935456f;
    float qprod;
    float sigma;
    float epsilon;
    int ex_idx = gist_find_exception(atom_i, atom_j, ex_i, ex_j, n_ex);
    if (ex_idx >= 0) {
        qprod = ex_qprod[ex_idx];
        sigma = ex_sigma[ex_idx];
        epsilon = ex_epsilon[ex_idx];
    } else {
        qprod = charges[atom_i] * charges[atom_j];
        sigma = 0.5f * (sigmas[atom_i] + sigmas[atom_j]);
        epsilon = sqrtf(epsilons[atom_i] * epsilons[atom_j]);
    }

    float e = 0.0f;
    if (epsilon != 0.0f && sigma != 0.0f) {
        float sr = sigma / r;
        float sr2 = sr * sr;
        float sr6 = sr2 * sr2 * sr2;
        e += 4.0f * epsilon * (sr6 * sr6 - sr6);
    }
    if (qprod != 0.0f) {
        e += coulomb_const * qprod / r;
    }
    return e;
}

__global__ void gist_direct_energy(const Float4* coords,
                                   int n_atoms,
                                   int frame,
                                   const unsigned int* water_offsets,
                                   const unsigned int* water_atoms,
                                   int n_waters,
                                   const unsigned int* solute_atoms,
                                   int n_solute,
                                   const float* charges,
                                   const float* sigmas,
                                   const float* epsilons,
                                   const unsigned int* ex_i,
                                   const unsigned int* ex_j,
                                   const float* ex_qprod,
                                   const float* ex_sigma,
                                   const float* ex_epsilon,
                                   int n_ex,
                                   int pbc_mode,
                                   float lx,
                                   float ly,
                                   float lz,
                                   float c00,
                                   float c01,
                                   float c02,
                                   float c10,
                                   float c11,
                                   float c12,
                                   float c20,
                                   float c21,
                                   float c22,
                                   float r00,
                                   float r01,
                                   float r02,
                                   float r10,
                                   float r11,
                                   float r12,
                                   float r20,
                                   float r21,
                                   float r22,
                                   float cutoff,
                                   float length_scale,
                                   float* out_sw,
                                   float* out_ww) {
    int wi = blockIdx.x * blockDim.x + threadIdx.x;
    if (wi >= n_waters) return;
    int frame_base = frame * n_atoms;
    unsigned int wi_start = water_offsets[wi];
    unsigned int wi_end = water_offsets[wi + 1];

    float e_sw = 0.0f;
    if (n_solute > 0) {
        for (unsigned int ia = wi_start; ia < wi_end; ++ia) {
            unsigned int atom_i = water_atoms[ia];
            for (int sj = 0; sj < n_solute; ++sj) {
                unsigned int atom_j = solute_atoms[sj];
                e_sw += gist_pair_energy(coords,
                                         frame_base,
                                         atom_i,
                                         atom_j,
                                         charges,
                                         sigmas,
                                         epsilons,
                                         ex_i,
                                         ex_j,
                                         ex_qprod,
                                         ex_sigma,
                                         ex_epsilon,
                                         n_ex,
                                         pbc_mode,
                                         lx,
                                         ly,
                                         lz,
                                         c00,
                                         c01,
                                         c02,
                                         c10,
                                         c11,
                                         c12,
                                         c20,
                                         c21,
                                         c22,
                                         r00,
                                         r01,
                                         r02,
                                         r10,
                                         r11,
                                         r12,
                                         r20,
                                         r21,
                                         r22,
                                         cutoff,
                                         length_scale);
            }
        }
    }
    out_sw[wi] = e_sw;

    float e_ww = 0.0f;
    for (int wj = 0; wj < n_waters; ++wj) {
        if (wj == wi) continue;
        unsigned int wj_start = water_offsets[wj];
        unsigned int wj_end = water_offsets[wj + 1];
        for (unsigned int ia = wi_start; ia < wi_end; ++ia) {
            unsigned int atom_i = water_atoms[ia];
            for (unsigned int ja = wj_start; ja < wj_end; ++ja) {
                unsigned int atom_j = water_atoms[ja];
                e_ww += gist_pair_energy(coords,
                                         frame_base,
                                         atom_i,
                                         atom_j,
                                         charges,
                                         sigmas,
                                         epsilons,
                                         ex_i,
                                         ex_j,
                                         ex_qprod,
                                         ex_sigma,
                                         ex_epsilon,
                                         n_ex,
                                         pbc_mode,
                                         lx,
                                         ly,
                                         lz,
                                         c00,
                                         c01,
                                         c02,
                                         c10,
                                         c11,
                                         c12,
                                         c20,
                                         c21,
                                         c22,
                                         r00,
                                         r01,
                                         r02,
                                         r10,
                                         r11,
                                         r12,
                                         r20,
                                         r21,
                                         r22,
                                         cutoff,
                                         length_scale);
            }
        }
    }
    out_ww[wi] = 0.5f * e_ww;
}

__global__ void gist_counts_orient(const Float4* coords,
                                   int n_atoms,
                                   int frame,
                                   const unsigned int* oxy_idx,
                                   const unsigned int* h1_idx,
                                   const unsigned int* h2_idx,
                                   const unsigned int* orient_valid,
                                   int n_waters,
                                   float center_x,
                                   float center_y,
                                   float center_z,
                                   float origin_x,
                                   float origin_y,
                                   float origin_z,
                                   float spacing,
                                   int dim_x,
                                   int dim_y,
                                   int dim_z,
                                   int orientation_bins,
                                   float length_scale,
                                   unsigned int* out_cell,
                                   unsigned int* out_bin) {
    int wi = blockIdx.x * blockDim.x + threadIdx.x;
    if (wi >= n_waters) return;
    const unsigned int invalid = 0xFFFFFFFFu;
    out_cell[wi] = invalid;
    out_bin[wi] = invalid;

    unsigned int oi = oxy_idx[wi];
    if ((int)oi >= n_atoms) return;
    int frame_base = frame * n_atoms;
    Float4 po = coords[frame_base + oi];
    float ox = po.x * length_scale;
    float oy = po.y * length_scale;
    float oz = po.z * length_scale;

    float fx = (ox - origin_x) / spacing;
    float fy = (oy - origin_y) / spacing;
    float fz = (oz - origin_z) / spacing;
    if (fx < 0.0f || fy < 0.0f || fz < 0.0f) return;
    int ix = (int)floorf(fx);
    int iy = (int)floorf(fy);
    int iz = (int)floorf(fz);
    if (ix < 0 || iy < 0 || iz < 0) return;
    if (ix >= dim_x || iy >= dim_y || iz >= dim_z) return;
    unsigned int flat = (unsigned int)(ix + dim_x * (iy + dim_y * iz));
    out_cell[wi] = flat;

    if (orient_valid[wi] == 0u) return;
    unsigned int h1 = h1_idx[wi];
    unsigned int h2 = h2_idx[wi];
    if ((int)h1 >= n_atoms || (int)h2 >= n_atoms) return;

    Float4 ph1 = coords[frame_base + h1];
    Float4 ph2 = coords[frame_base + h2];
    float hmx = 0.5f * ((ph1.x + ph2.x) * length_scale);
    float hmy = 0.5f * ((ph1.y + ph2.y) * length_scale);
    float hmz = 0.5f * ((ph1.z + ph2.z) * length_scale);

    float hvx = hmx - ox;
    float hvy = hmy - oy;
    float hvz = hmz - oz;
    float rvx = ox - center_x;
    float rvy = oy - center_y;
    float rvz = oz - center_z;
    float hnorm = sqrtf(hvx * hvx + hvy * hvy + hvz * hvz);
    float rnorm = sqrtf(rvx * rvx + rvy * rvy + rvz * rvz);
    if (hnorm <= 0.0f || rnorm <= 0.0f) return;
    float cos_t = (hvx * rvx + hvy * rvy + hvz * rvz) / (hnorm * rnorm);
    if (cos_t < -1.0f) cos_t = -1.0f;
    if (cos_t > 1.0f) cos_t = 1.0f;
    int bin = (int)floorf(((cos_t + 1.0f) * 0.5f) * (float)orientation_bins);
    if (bin < 0) bin = 0;
    if (bin >= orientation_bins) bin = orientation_bins - 1;
    out_bin[wi] = (unsigned int)bin;
}

__global__ void gist_accumulate_hist(const unsigned int* cell_idx,
                                     const unsigned int* bin_idx,
                                     int n_waters,
                                     int orientation_bins,
                                     unsigned int* counts,
                                     unsigned int* orient_counts) {
    int wi = blockIdx.x * blockDim.x + threadIdx.x;
    if (wi >= n_waters) return;
    unsigned int cell = cell_idx[wi];
    if (cell == 0xFFFFFFFFu) return;
    atomicAdd(&counts[cell], 1u);
    unsigned int bin = bin_idx[wi];
    if (bin == 0xFFFFFFFFu) return;
    if ((int)bin >= orientation_bins) return;
    unsigned int flat = cell * (unsigned int)orientation_bins + bin;
    atomicAdd(&orient_counts[flat], 1u);
}

__global__ void hbond_count(const Float4* coords,
                            const unsigned int* donors,
                            const unsigned int* acceptors,
                            int n_donors,
                            int n_acceptors,
                            int n_atoms,
                            int n_frames,
                            float dist2,
                            unsigned int* counts) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int frame = blockIdx.y;
    if (frame >= n_frames) return;
    int total = n_donors * n_acceptors;
    if (pair >= total) return;
    int d = pair / n_acceptors;
    int a = pair - d * n_acceptors;
    unsigned int donor = donors[d];
    unsigned int acceptor = acceptors[a];
    if (donor == acceptor) return;
    Float4 pd = coords[frame * n_atoms + donor];
    Float4 pa = coords[frame * n_atoms + acceptor];
    float dx = pa.x - pd.x;
    float dy = pa.y - pd.y;
    float dz = pa.z - pd.z;
