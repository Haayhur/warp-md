        if (idx_cluster[off_i + k] != idx_cluster[off_j + k]) {
            return;
        }
    }
    atomicAdd(&cp_out[dt], 1);
}

__global__ void ion_pair_corr_ani(const unsigned int* idx_pair,
                                  const unsigned int* idx_cluster,
                                  int n_frames,
                                  int n_ani,
                                  int max_cluster,
                                  unsigned int* ip_out,
                                  unsigned int* cp_out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    int ani = blockIdx.z;
    if (i >= n_frames || j >= n_frames || ani >= n_ani) return;
    if (j < i) return;
    int dt = j - i;
    unsigned int a = idx_pair[i * n_ani + ani];
    unsigned int b = idx_pair[j * n_ani + ani];
    if (a == b) {
        atomicAdd(&ip_out[dt], 1);
    }
    int off_i = (i * n_ani + ani) * max_cluster;
    int off_j = (j * n_ani + ani) * max_cluster;
    for (int k = 0; k < max_cluster; k++) {
        if (idx_cluster[off_i + k] != idx_cluster[off_j + k]) {
            return;
        }
    }
    atomicAdd(&cp_out[dt], 1);
}

// warp-pack: naive O(N^2) per-atom max overlap (CPU parity baseline).
__global__ void pack_overlap_max(const Float4* pos,
                                 const float* radius,
                                 const int* mol_id,
                                 int n_atoms,
                                 Float4 box,
                                 float* out_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;
    Float4 pi = pos[i];
    int mi = mol_id[i];
    float ri = radius[i];
    float max_overlap = 0.0f;
    for (int j = 0; j < n_atoms; j++) {
        if (j == i) continue;
        if (mol_id[j] == mi) continue;
        Float4 pj = pos[j];
        float dx = pi.x - pj.x;
        float dy = pi.y - pj.y;
        float dz = pi.z - pj.z;
        dx = wrap_pbc(dx, box.x);
        dy = wrap_pbc(dy, box.y);
        dz = wrap_pbc(dz, box.z);
        float d2 = dx * dx + dy * dy + dz * dz;
        float tol = ri + radius[j];
        float tol2 = tol * tol;
        if (d2 < tol2) {
            float overlap = tol - sqrtf(d2);
            if (overlap > max_overlap) {
                max_overlap = overlap;
            }
        }
    }
    out_max[i] = max_overlap;
}

__device__ __forceinline__ int wrap_index(int v, int n) {
    int r = v % n;
    return (r < 0) ? r + n : r;
}

__global__ void pack_overlap_max_cells(const Float4* pos,
                                       const float* radius,
                                       const int* mol_id,
                                       int n_atoms,
                                       Float4 box_min,
                                       Float4 box_len,
                                       float cell_size,
                                       float inv_cell,
                                       int nx,
                                       int ny,
                                       int nz,
                                       const int* cell_offsets,
                                       const int* cell_atoms,
                                       float* out_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;
    Float4 pi = pos[i];
    int mi = mol_id[i];
    float ri = radius[i];
    bool pbc = (box_len.x > 0.0f && box_len.y > 0.0f && box_len.z > 0.0f);

    float lx = box_len.x;
    float ly = box_len.y;
    float lz = box_len.z;

    float x = pi.x - box_min.x;
    float y = pi.y - box_min.y;
    float z = pi.z - box_min.z;
    if (pbc) {
        x -= floorf(x / lx) * lx;
        y -= floorf(y / ly) * ly;
        z -= floorf(z / lz) * lz;
    }

    int ix = (int)floorf(x * inv_cell);
    int iy = (int)floorf(y * inv_cell);
    int iz = (int)floorf(z * inv_cell);
    if (pbc) {
        ix = wrap_index(ix, nx);
        iy = wrap_index(iy, ny);
        iz = wrap_index(iz, nz);
    } else {
        if (ix < 0) ix = 0;
        if (iy < 0) iy = 0;
        if (iz < 0) iz = 0;
        if (ix >= nx) ix = nx - 1;
        if (iy >= ny) iy = ny - 1;
        if (iz >= nz) iz = nz - 1;
    }

    float max_overlap = 0.0f;
    for (int dz = -1; dz <= 1; dz++) {
        int zc = iz + dz;
        if (pbc) {
            zc = wrap_index(zc, nz);
        } else if (zc < 0 || zc >= nz) {
            continue;
        }
        for (int dy = -1; dy <= 1; dy++) {
            int yc = iy + dy;
            if (pbc) {
                yc = wrap_index(yc, ny);
            } else if (yc < 0 || yc >= ny) {
                continue;
            }
            for (int dx = -1; dx <= 1; dx++) {
                int xc = ix + dx;
                if (pbc) {
                    xc = wrap_index(xc, nx);
                } else if (xc < 0 || xc >= nx) {
                    continue;
                }
                int cell_id = (zc * ny + yc) * nx + xc;
                int start = cell_offsets[cell_id];
                int end = cell_offsets[cell_id + 1];
                for (int idx = start; idx < end; idx++) {
                    int j = cell_atoms[idx];
                    if (j == i) continue;
                    if (mol_id[j] == mi) continue;
                    Float4 pj = pos[j];
                    float dx = pi.x - pj.x;
                    float dy = pi.y - pj.y;
                    float dz = pi.z - pj.z;
                    if (pbc) {
                        dx = wrap_pbc(dx, lx);
                        dy = wrap_pbc(dy, ly);
                        dz = wrap_pbc(dz, lz);
                    }
                    float d2 = dx * dx + dy * dy + dz * dz;
                    float tol = ri + radius[j];
                    float tol2 = tol * tol;
                    if (d2 < tol2) {
                        float overlap = tol - sqrtf(d2);
                        if (overlap > max_overlap) {
                            max_overlap = overlap;
                        }
                    }
                }
            }
        }
    }
    out_max[i] = max_overlap;
}

__global__ void pack_overlap_penalty_cells(const Float4* pos,
                                           const float* radius,
                                           const float* fscale,
                                           const int* mol_id,
                                           const unsigned char* movable,
                                           int n_atoms,
                                           Float4 box_min,
                                           Float4 box_len,
                                           float cell_size,
                                           float inv_cell,
                                           int nx,
                                           int ny,
                                           int nz,
                                           const int* cell_offsets,
                                           const int* cell_atoms,
                                           float* out_penalty) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;
    Float4 pi = pos[i];
    int mi = mol_id[i];
    float ri = radius[i];
    bool pbc = (box_len.x > 0.0f && box_len.y > 0.0f && box_len.z > 0.0f);

    float lx = box_len.x;
    float ly = box_len.y;
    float lz = box_len.z;

    float x = pi.x - box_min.x;
    float y = pi.y - box_min.y;
    float z = pi.z - box_min.z;
    if (pbc) {
        x -= floorf(x / lx) * lx;
        y -= floorf(y / ly) * ly;
        z -= floorf(z / lz) * lz;
    }

    int ix = (int)floorf(x * inv_cell);
    int iy = (int)floorf(y * inv_cell);
    int iz = (int)floorf(z * inv_cell);
    if (pbc) {
        ix = wrap_index(ix, nx);
        iy = wrap_index(iy, ny);
        iz = wrap_index(iz, nz);
    } else {
        if (ix < 0) ix = 0;
        if (iy < 0) iy = 0;
        if (iz < 0) iz = 0;
        if (ix >= nx) ix = nx - 1;
        if (iy >= ny) iy = ny - 1;
        if (iz >= nz) iz = nz - 1;
    }

    float penalty = 0.0f;
    for (int dz = -1; dz <= 1; dz++) {
        int zc = iz + dz;
        if (pbc) {
            zc = wrap_index(zc, nz);
        } else if (zc < 0 || zc >= nz) {
            continue;
        }
        for (int dy = -1; dy <= 1; dy++) {
            int yc = iy + dy;
            if (pbc) {
                yc = wrap_index(yc, ny);
            } else if (yc < 0 || yc >= ny) {
                continue;
            }
            for (int dx = -1; dx <= 1; dx++) {
                int xc = ix + dx;
                if (pbc) {
                    xc = wrap_index(xc, nx);
                } else if (xc < 0 || xc >= nx) {
                    continue;
                }
                int cell_id = (zc * ny + yc) * nx + xc;
                int start = cell_offsets[cell_id];
                int end = cell_offsets[cell_id + 1];
                for (int idx = start; idx < end; idx++) {
                    int j = cell_atoms[idx];
                    if (j <= i) continue;
                    if (!movable[i] && !movable[j]) continue;
                    if (mol_id[j] == mi) continue;
                    Float4 pj = pos[j];
                    float dx = pi.x - pj.x;
                    float dy = pi.y - pj.y;
                    float dz = pi.z - pj.z;
                    if (pbc) {
                        dx = wrap_pbc(dx, lx);
                        dy = wrap_pbc(dy, ly);
                        dz = wrap_pbc(dz, lz);
                    }
                    float d2 = dx * dx + dy * dy + dz * dz;
                    float tol = ri + radius[j];
                    float tol2 = tol * tol;
                    if (d2 < tol2) {
                        float diff = d2 - tol2;
                        float weight = fscale[i] * fscale[j];
                        penalty += weight * diff * diff;
                    }
                }
            }
        }
    }
    out_penalty[i] = penalty;
}

__global__ void pack_overlap_max_cells_movable(const Float4* pos,
                                               const float* radius,
                                               const int* mol_id,
                                               const unsigned char* movable,
                                               int n_atoms,
                                               Float4 box_min,
                                               Float4 box_len,
                                               float cell_size,
                                               float inv_cell,
                                               int nx,
                                               int ny,
                                               int nz,
                                               const int* cell_offsets,
                                               const int* cell_atoms,
                                               float* out_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;
    Float4 pi = pos[i];
    int mi = mol_id[i];
    float ri = radius[i];
    bool pbc = (box_len.x > 0.0f && box_len.y > 0.0f && box_len.z > 0.0f);

    float lx = box_len.x;
    float ly = box_len.y;
    float lz = box_len.z;

    float x = pi.x - box_min.x;
    float y = pi.y - box_min.y;
    float z = pi.z - box_min.z;
    if (pbc) {
        x -= floorf(x / lx) * lx;
        y -= floorf(y / ly) * ly;
        z -= floorf(z / lz) * lz;
    }

    int ix = (int)floorf(x * inv_cell);
    int iy = (int)floorf(y * inv_cell);
    int iz = (int)floorf(z * inv_cell);
    if (pbc) {
        ix = wrap_index(ix, nx);
        iy = wrap_index(iy, ny);
        iz = wrap_index(iz, nz);
    } else {
        if (ix < 0) ix = 0;
        if (iy < 0) iy = 0;
        if (iz < 0) iz = 0;
        if (ix >= nx) ix = nx - 1;
        if (iy >= ny) iy = ny - 1;
        if (iz >= nz) iz = nz - 1;
    }

    float max_overlap = 0.0f;
    for (int dz = -1; dz <= 1; dz++) {
        int zc = iz + dz;
        if (pbc) {
            zc = wrap_index(zc, nz);
        } else if (zc < 0 || zc >= nz) {
            continue;
        }
        for (int dy = -1; dy <= 1; dy++) {
            int yc = iy + dy;
            if (pbc) {
                yc = wrap_index(yc, ny);
            } else if (yc < 0 || yc >= ny) {
                continue;
            }
            for (int dx = -1; dx <= 1; dx++) {
                int xc = ix + dx;
                if (pbc) {
                    xc = wrap_index(xc, nx);
                } else if (xc < 0 || xc >= nx) {
                    continue;
                }
                int cell_id = (zc * ny + yc) * nx + xc;
                int start = cell_offsets[cell_id];
                int end = cell_offsets[cell_id + 1];
                for (int idx = start; idx < end; idx++) {
                    int j = cell_atoms[idx];
                    if (j == i) continue;
                    if (!movable[i] && !movable[j]) continue;
                    if (mol_id[j] == mi) continue;
                    Float4 pj = pos[j];
                    float dx = pi.x - pj.x;
                    float dy = pi.y - pj.y;
                    float dz = pi.z - pj.z;
                    if (pbc) {
                        dx = wrap_pbc(dx, lx);
                        dy = wrap_pbc(dy, ly);
                        dz = wrap_pbc(dz, lz);
                    }
                    float d2 = dx * dx + dy * dy + dz * dz;
                    float tol = ri + radius[j];
                    float tol2 = tol * tol;
                    if (d2 < tol2) {
                        float overlap = tol - sqrtf(d2);
                        if (overlap > max_overlap) {
                            max_overlap = overlap;
                        }
                    }
                }
            }
        }
    }
    out_max[i] = max_overlap;
}

__global__ void pack_overlap_grad_cells(const Float4* pos,
                                        const float* radius,
                                        const float* fscale,
                                        const int* mol_id,
                                        const unsigned char* movable,
                                        int n_atoms,
                                        Float4 box_min,
                                        Float4 box_len,
                                        float cell_size,
                                        float inv_cell,
                                        int nx,
                                        int ny,
                                        int nz,
                                        const int* cell_offsets,
                                        const int* cell_atoms,
                                        Float4* out_grad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;
    Float4 pi = pos[i];
    int mi = mol_id[i];
    float ri = radius[i];
    bool pbc = (box_len.x > 0.0f && box_len.y > 0.0f && box_len.z > 0.0f);

    float lx = box_len.x;
    float ly = box_len.y;
    float lz = box_len.z;

    float x = pi.x - box_min.x;
    float y = pi.y - box_min.y;
    float z = pi.z - box_min.z;
    if (pbc) {
        x -= floorf(x / lx) * lx;
        y -= floorf(y / ly) * ly;
        z -= floorf(z / lz) * lz;
    }

    int ix = (int)floorf(x * inv_cell);
    int iy = (int)floorf(y * inv_cell);
    int iz = (int)floorf(z * inv_cell);
    if (pbc) {
        ix = wrap_index(ix, nx);
        iy = wrap_index(iy, ny);
        iz = wrap_index(iz, nz);
    } else {
        if (ix < 0) ix = 0;
        if (iy < 0) iy = 0;
        if (iz < 0) iz = 0;
        if (ix >= nx) ix = nx - 1;
        if (iy >= ny) iy = ny - 1;
        if (iz >= nz) iz = nz - 1;
    }

    for (int dz = -1; dz <= 1; dz++) {
        int zc = iz + dz;
        if (pbc) {
            zc = wrap_index(zc, nz);
        } else if (zc < 0 || zc >= nz) {
            continue;
        }
        for (int dy = -1; dy <= 1; dy++) {
            int yc = iy + dy;
            if (pbc) {
                yc = wrap_index(yc, ny);
            } else if (yc < 0 || yc >= ny) {
                continue;
            }
            for (int dx = -1; dx <= 1; dx++) {
                int xc = ix + dx;
                if (pbc) {
                    xc = wrap_index(xc, nx);
                } else if (xc < 0 || xc >= nx) {
                    continue;
                }
                int cell_id = (zc * ny + yc) * nx + xc;
                int start = cell_offsets[cell_id];
                int end = cell_offsets[cell_id + 1];
                for (int idx = start; idx < end; idx++) {
                    int j = cell_atoms[idx];
                    if (j <= i) continue;
                    if (!movable[i] && !movable[j]) continue;
                    if (mol_id[j] == mi) continue;
                    Float4 pj = pos[j];
                    float dx = pi.x - pj.x;
                    float dy = pi.y - pj.y;
                    float dz = pi.z - pj.z;
                    if (pbc) {
                        dx = wrap_pbc(dx, lx);
                        dy = wrap_pbc(dy, ly);
                        dz = wrap_pbc(dz, lz);
                    }
                    float d2 = dx * dx + dy * dy + dz * dz;
                    float tol = ri + radius[j];
                    float tol2 = tol * tol;
                    if (d2 < tol2) {
                        float diff = d2 - tol2;
                        float weight = fscale[i] * fscale[j];
                        float coeff = weight * 4.0f * diff;
                        float gx = dx * coeff;
                        float gy = dy * coeff;
                        float gz = dz * coeff;
                        atomicAdd(&out_grad[i].x, gx);
                        atomicAdd(&out_grad[i].y, gy);
                        atomicAdd(&out_grad[i].z, gz);
                        atomicAdd(&out_grad[j].x, -gx);
                        atomicAdd(&out_grad[j].y, -gy);
                        atomicAdd(&out_grad[j].z, -gz);
                    }
                }
            }
        }
    }
}

__global__ void pack_short_tol_penalty_grad_cells(const Float4* pos,
                                                  const float* radius,
                                                  const float* short_radius,
                                                  const float* fscale,
                                                  const float* short_scale,
                                                  const unsigned char* use_short,
                                                  const int* mol_id,
                                                  const unsigned char* movable,
                                                  int n_atoms,
                                                  Float4 box_min,
                                                  Float4 box_len,
                                                  float cell_size,
                                                  float inv_cell,
                                                  int nx,
                                                  int ny,
                                                  int nz,
                                                  const int* cell_offsets,
                                                  const int* cell_atoms,
                                                  float* out_penalty,
                                                  Float4* out_grad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;
    Float4 pi = pos[i];
    int mi = mol_id[i];
    float ri = radius[i];
    bool pbc = (box_len.x > 0.0f && box_len.y > 0.0f && box_len.z > 0.0f);

    float lx = box_len.x;
    float ly = box_len.y;
    float lz = box_len.z;

    float x = pi.x - box_min.x;
    float y = pi.y - box_min.y;
    float z = pi.z - box_min.z;
    if (pbc) {
        x -= floorf(x / lx) * lx;
        y -= floorf(y / ly) * ly;
        z -= floorf(z / lz) * lz;
    }

    int ix = (int)floorf(x * inv_cell);
    int iy = (int)floorf(y * inv_cell);
    int iz = (int)floorf(z * inv_cell);
    if (pbc) {
        ix = wrap_index(ix, nx);
        iy = wrap_index(iy, ny);
        iz = wrap_index(iz, nz);
    } else {
        if (ix < 0) ix = 0;
        if (iy < 0) iy = 0;
        if (iz < 0) iz = 0;
        if (ix >= nx) ix = nx - 1;
        if (iy >= ny) iy = ny - 1;
        if (iz >= nz) iz = nz - 1;
    }

    for (int dz = -1; dz <= 1; dz++) {
        int zc = iz + dz;
        if (pbc) {
            zc = wrap_index(zc, nz);
        } else if (zc < 0 || zc >= nz) {
            continue;
        }
        for (int dy = -1; dy <= 1; dy++) {
            int yc = iy + dy;
            if (pbc) {
                yc = wrap_index(yc, ny);
            } else if (yc < 0 || yc >= ny) {
                continue;
            }
            for (int dx = -1; dx <= 1; dx++) {
                int xc = ix + dx;
                if (pbc) {
                    xc = wrap_index(xc, nx);
                } else if (xc < 0 || xc >= nx) {
                    continue;
                }
                int cell_id = (zc * ny + yc) * nx + xc;
                int start = cell_offsets[cell_id];
                int end = cell_offsets[cell_id + 1];
                for (int idx = start; idx < end; idx++) {
                    int j = cell_atoms[idx];
                    if (j <= i) continue;
                    if (!movable[i] && !movable[j]) continue;
                    if (mol_id[j] == mi) continue;
                    if (!use_short[i] && !use_short[j]) continue;
                    Float4 pj = pos[j];
                    float dx = pi.x - pj.x;
                    float dy = pi.y - pj.y;
                    float dz = pi.z - pj.z;
                    if (pbc) {
                        dx = wrap_pbc(dx, lx);
                        dy = wrap_pbc(dy, ly);
                        dz = wrap_pbc(dz, lz);
                    }
                    float d2 = dx * dx + dy * dy + dz * dz;
                    float tol = ri + radius[j];
                    float tol2 = tol * tol;
                    if (d2 < tol2) {
                        continue;
                    }
                    float short_r = short_radius[i] + short_radius[j];
                    if (short_r <= 0.0f) {
                        continue;
                    }
                    float short2 = short_r * short_r;
                    if (d2 < short2) {
                        float diff = d2 - short2;
                        float scale = sqrtf(short_scale[i] * short_scale[j]);
                        float weight = fscale[i] * fscale[j] * scale;
                        out_penalty[i] += weight * diff * diff;
                        float coeff = weight * 4.0f * diff;
                        float gx = dx * coeff;
                        float gy = dy * coeff;
                        float gz = dz * coeff;
                        atomicAdd(&out_grad[i].x, gx);
                        atomicAdd(&out_grad[i].y, gy);
                        atomicAdd(&out_grad[i].z, gz);
                        atomicAdd(&out_grad[j].x, -gx);
                        atomicAdd(&out_grad[j].y, -gy);
                        atomicAdd(&out_grad[j].z, -gz);
                    }
                }
            }
        }
    }
}

__global__ void pack_constraint_penalty(const Float4* pos,
                                        const unsigned char* types,
                                        const unsigned char* modes,
                                        const Float4* data0,
                                        const Float4* data1,
                                        const int* atom_offsets,
                                        const int* atom_constraints,
                                        int n_atoms,
                                        float* out_sum,
                                        Float4* out_grad,
                                        float* out_max_val,
                                        float* out_max_violation) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;
    Float4 p = pos[i];
    float sum = 0.0f;
    float gx = 0.0f;
    float gy = 0.0f;
    float gz = 0.0f;
    float max_val = 0.0f;
    float max_violation = 0.0f;
    int start = atom_offsets[i];
    int end = atom_offsets[i + 1];
    for (int k = start; k < end; k++) {
        int c = atom_constraints[k];
        unsigned char t = types[c];
        unsigned char m = modes[c];
        Float4 a = data0[c];
        Float4 b = data1[c];
        float value = 0.0f;
        float vx = 0.0f;
        float vy = 0.0f;
        float vz = 0.0f;
        float violation = 0.0f;

        if (t == 0) {
            // Box: data0 = (min.x, min.y, min.z, max.x), data1 = (max.y, max.z, 0, 0)
            float minx = a.x;
            float miny = a.y;
            float minz = a.z;
            float maxx = a.w;
            float maxy = b.x;
            float maxz = b.y;
            if (m == 0) {
                float dx_low = p.x - minx;
                if (dx_low < 0.0f) {
                    value += dx_low * dx_low;
                    vx += 2.0f * dx_low;
                    violation = fmaxf(violation, -dx_low);
                }
                float dx_high = p.x - maxx;
                if (dx_high > 0.0f) {
                    value += dx_high * dx_high;
                    vx += 2.0f * dx_high;
                    violation = fmaxf(violation, dx_high);
                }
                float dy_low = p.y - miny;
                if (dy_low < 0.0f) {
                    value += dy_low * dy_low;
                    vy += 2.0f * dy_low;
                    violation = fmaxf(violation, -dy_low);
                }
                float dy_high = p.y - maxy;
                if (dy_high > 0.0f) {
                    value += dy_high * dy_high;
                    vy += 2.0f * dy_high;
                    violation = fmaxf(violation, dy_high);
                }
                float dz_low = p.z - minz;
                if (dz_low < 0.0f) {
                    value += dz_low * dz_low;
                    vz += 2.0f * dz_low;
                    violation = fmaxf(violation, -dz_low);
                }
                float dz_high = p.z - maxz;
                if (dz_high > 0.0f) {
                    value += dz_high * dz_high;
                    vz += 2.0f * dz_high;
                    violation = fmaxf(violation, dz_high);
                }
            } else if (m == 1) {
                bool inside = (p.x > minx && p.x < maxx && p.y > miny && p.y < maxy && p.z > minz && p.z < maxz);
                if (inside) {
                    float midx = (minx + maxx) * 0.5f;
                    float midy = (miny + maxy) * 0.5f;
                    float midz = (minz + maxz) * 0.5f;
                    float dx = (p.x <= midx) ? (p.x - minx) : (maxx - p.x);
                    float sx = (p.x <= midx) ? 1.0f : -1.0f;
                    value += dx;
                    vx += sx;
                    violation = fmaxf(violation, dx);
                    float dy = (p.y <= midy) ? (p.y - miny) : (maxy - p.y);
                    float sy = (p.y <= midy) ? 1.0f : -1.0f;
                    value += dy;
                    vy += sy;
                    violation = fmaxf(violation, dy);
                    float dz = (p.z <= midz) ? (p.z - minz) : (maxz - p.z);
                    float sz = (p.z <= midz) ? 1.0f : -1.0f;
                    value += dz;
                    vz += sz;
                    violation = fmaxf(violation, dz);
                }
            }
        } else if (t == 1) {
            // Sphere: data0 = (center.x, center.y, center.z, radius)
            float cx = a.x;
            float cy = a.y;
            float cz = a.z;
            float r = a.w;
            float dx = p.x - cx;
            float dy = p.y - cy;
            float dz = p.z - cz;
            float d2 = dx * dx + dy * dy + dz * dz;
            if (m == 0) {
                float w = d2 - r * r;
                if (w > 0.0f) {
                    value = w * w;
                    float coeff = 4.0f * w;
                    vx = dx * coeff;
                    vy = dy * coeff;
                    vz = dz * coeff;
                    violation = sqrtf(w);
                }
            } else if (m == 1) {
                if (d2 < r * r) {
                    float dist = sqrtf(d2);
                    float w = r - dist;
                    value = w * w;
                    if (dist > 1.0e-6f) {
                        float coeff = -2.0f * w / dist;
                        vx = dx * coeff;
                        vy = dy * coeff;
                        vz = dz * coeff;
                    }
                    violation = w;
                }
            }
        } else if (t == 2) {
            // Ellipsoid: data0 = (center.x, center.y, center.z, rx), data1 = (ry, rz, 0, 0)
            float cx = a.x;
            float cy = a.y;
            float cz = a.z;
            float rx = a.w;
            float ry = b.x;
            float rz = b.y;
            float dx = (p.x - cx) / rx;
            float dy = (p.y - cy) / ry;
            float dz = (p.z - cz) / rz;
            float s = dx * dx + dy * dy + dz * dz;
            if (m == 0) {
                float w = s - 1.0f;
                if (w > 0.0f) {
                    value = w * w;
                    vx = 2.0f * w * dx / rx;
                    vy = 2.0f * w * dy / ry;
                    vz = 2.0f * w * dz / rz;
                    violation = w;
                }
            } else if (m == 1) {
                float w = 1.0f - s;
                if (w > 0.0f) {
                    value = w * w;
                    vx = -2.0f * w * dx / rx;
                    vy = -2.0f * w * dy / ry;
                    vz = -2.0f * w * dz / rz;
                    violation = w;
                }
            }
        } else if (t == 3) {
            // Cylinder: data0 = (base.x, base.y, base.z, radius), data1 = (axis.x, axis.y, axis.z, height)
            float bx = a.x;
            float by = a.y;
            float bz = a.z;
            float rad = a.w;
            float ax = b.x;
            float ay = b.y;
            float az = b.z;
            float height = b.w;
            float axis_len = sqrtf(ax * ax + ay * ay + az * az);
            if (axis_len > 1.0e-6f) {
                float ux = ax / axis_len;
                float uy = ay / axis_len;
                float uz = az / axis_len;
                float vx0 = p.x - bx;
                float vy0 = p.y - by;
                float vz0 = p.z - bz;
                float proj = vx0 * ux + vy0 * uy + vz0 * uz;
                if (m == 0) {
                    if (proj < 0.0f) {
                        float w = proj;
                        value += w * w;
                        vx += 2.0f * w * ux;
                        vy += 2.0f * w * uy;
                        vz += 2.0f * w * uz;
                        violation = fmaxf(violation, -w);
                    } else if (proj > height) {
                        float w = proj - height;
                        value += w * w;
                        vx += 2.0f * w * ux;
                        vy += 2.0f * w * uy;
                        vz += 2.0f * w * uz;
                        violation = fmaxf(violation, w);
                    }
                    float rx = vx0 - proj * ux;
                    float ry = vy0 - proj * uy;
                    float rz = vz0 - proj * uz;
                    float dist = sqrtf(rx * rx + ry * ry + rz * rz);
                    if (dist > rad) {
                        float w = dist - rad;
                        value += w * w;
                        if (dist > 1.0e-6f) {
                            float coeff = 2.0f * w / dist;
                            vx += rx * coeff;
                            vy += ry * coeff;
                            vz += rz * coeff;
                        }
                        violation = fmaxf(violation, w);
                    }
                } else if (m == 1) {
                    if (proj >= 0.0f && proj <= height) {
                        float rx = vx0 - proj * ux;
                        float ry = vy0 - proj * uy;
                        float rz = vz0 - proj * uz;
                        float dist = sqrtf(rx * rx + ry * ry + rz * rz);
                        if (dist < rad) {
                            float w = rad - dist;
                            value = w * w;
                            if (dist > 1.0e-6f) {
                                float coeff = -2.0f * w / dist;
                                vx = rx * coeff;
                                vy = ry * coeff;
                                vz = rz * coeff;
                            }
                            violation = w;
                        }
                    }
                }
            }
        } else if (t == 4) {
            // Plane: data0 = (point.x, point.y, point.z, 0), data1 = (normal.x, normal.y, normal.z, 0)
            float px0 = a.x;
            float py0 = a.y;
            float pz0 = a.z;
            float nx = b.x;
            float ny = b.y;
            float nz = b.z;
            float w = (p.x - px0) * nx + (p.y - py0) * ny + (p.z - pz0) * nz;
            if (m == 0 || m == 2) {
                if (w < 0.0f) {
                    value = w * w;
                    float coeff = 2.0f * w;
                    vx = nx * coeff;
                    vy = ny * coeff;
                    vz = nz * coeff;
                    violation = -w;
                }
            } else if (m == 1 || m == 3) {
                if (w > 0.0f) {
                    value = w * w;
                    float coeff = 2.0f * w;
                    vx = nx * coeff;
                    vy = ny * coeff;
                    vz = nz * coeff;
                    violation = w;
                }
            }
        } else if (t == 5) {
            // XyGauss: data0 = (center.x, center.y, z0, amplitude), data1 = (sigma.x, sigma.y, 0, 0)
            float cx = a.x;
            float cy = a.y;
            float z0 = a.z;
            float amp = a.w;
            float sx = b.x;
            float sy = b.y;
            float dx = p.x - cx;
            float dy = p.y - cy;
            float expo = -(dx * dx) / (2.0f * sx * sx) - (dy * dy) / (2.0f * sy * sy);
            float surf = (expo <= -50.0f) ? z0 : (z0 + amp * expf(expo));
            float w = surf - p.z;
            if (m == 0 || m == 2) {
                if (w > 0.0f) {
                    value = w * w;
                    vz = -2.0f * w;
                    violation = w;
                }
            } else if (m == 1 || m == 3) {
                if (w < 0.0f) {
                    value = w * w;
                    vz = -2.0f * w;
                    violation = -w;
                }
            }
        }

        if (value > 0.0f) {
            sum += value;
            gx += vx;
            gy += vy;
            gz += vz;
            if (value > max_val) {
                max_val = value;
            }
            float v = violation;
            if (value > v) v = value;
            if (v > max_violation) {
                max_violation = v;
            }
        }
    }
    out_sum[i] = sum;
    out_grad[i].x = gx;
    out_grad[i].y = gy;
    out_grad[i].z = gz;
    out_max_val[i] = max_val;
    out_max_violation[i] = max_violation;
}

__device__ __forceinline__ void pack_fallback_dir(int i, int j, float* dx, float* dy, float* dz) {
    unsigned long long seed = ((unsigned long long)i) * 6364136223846793005ull;
    seed += (unsigned long long)j;
    seed += 1442695040888963407ull;
    float x = ((seed & 0xFF) / 255.0f) * 2.0f - 1.0f;
    float y = (((seed >> 8) & 0xFF) / 255.0f) * 2.0f - 1.0f;
    float z = (((seed >> 16) & 0xFF) / 255.0f) * 2.0f - 1.0f;
    float n = sqrtf(x * x + y * y + z * z);
    if (n > 1.0e-6f) {
        *dx = x / n;
        *dy = y / n;
        *dz = z / n;
    } else {
        *dx = 1.0f;
        *dy = 0.0f;
        *dz = 0.0f;
    }
}

__global__ void pack_relax_accum(const Float4* pos,
                                 const float* radius,
                                 const int* mol_id,
                                 const unsigned char* mol_movable,
                                 int n_atoms,
                                 int n_mols,
                                 Float4 box_min,
                                 float cell_size,
                                 float inv_cell,
                                 int nx,
                                 int ny,
                                 int nz,
                                 const int* cell_offsets,
                                 const int* cell_atoms,
                                 Float4* out_disp,
                                 float* out_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_atoms) return;
    int mol_i = mol_id[i] - 1;
    if (mol_i < 0 || mol_i >= n_mols) return;
    Float4 pi = pos[i];
    float ri = radius[i];

    float x = pi.x - box_min.x;
    float y = pi.y - box_min.y;
    float z = pi.z - box_min.z;

    int ix = (int)floorf(x * inv_cell);
    int iy = (int)floorf(y * inv_cell);
    int iz = (int)floorf(z * inv_cell);
    if (ix < 0) ix = 0;
    if (iy < 0) iy = 0;
    if (iz < 0) iz = 0;
    if (ix >= nx) ix = nx - 1;
    if (iy >= ny) iy = ny - 1;
    if (iz >= nz) iz = nz - 1;

    for (int dz = -1; dz <= 1; dz++) {
        int zc = iz + dz;
        if (zc < 0 || zc >= nz) {
            continue;
        }
        for (int dy = -1; dy <= 1; dy++) {
            int yc = iy + dy;
            if (yc < 0 || yc >= ny) {
                continue;
            }
            for (int dx = -1; dx <= 1; dx++) {
                int xc = ix + dx;
                if (xc < 0 || xc >= nx) {
                    continue;
                }
                int cell_id = (zc * ny + yc) * nx + xc;
                int start = cell_offsets[cell_id];
                int end = cell_offsets[cell_id + 1];
                for (int idx = start; idx < end; idx++) {
                    int j = cell_atoms[idx];
                    if (j >= i) continue;
                    int mol_j = mol_id[j] - 1;
                    if (mol_j < 0 || mol_j >= n_mols) continue;
                    if (mol_i == mol_j) continue;
                    Float4 pj = pos[j];
                    float dx = pi.x - pj.x;
                    float dy = pi.y - pj.y;
                    float dz = pi.z - pj.z;
                    float d2 = dx * dx + dy * dy + dz * dz;
                    float rj = radius[j];
                    float rij = ri + rj;
                    float min2 = rij * rij;
                    if (d2 >= min2) {
                        continue;
                    }
                    float dist = sqrtf(d2);
                    float overlap = rij - dist;
                    atomicMaxFloat(out_max, overlap);
                    float ux, uy, uz;
                    if (dist > 1.0e-6f) {
                        float inv = 1.0f / dist;
                        ux = dx * inv;
                        uy = dy * inv;
                        uz = dz * inv;
                    } else {
                        pack_fallback_dir(i, j, &ux, &uy, &uz);
                    }
                    bool move_i = mol_movable[mol_i] != 0;
                    bool move_j = mol_movable[mol_j] != 0;
                    if (move_i && move_j) {
                        float s = overlap * 0.5f;
                        atomicAdd(&out_disp[mol_i].x, ux * s);
                        atomicAdd(&out_disp[mol_i].y, uy * s);
                        atomicAdd(&out_disp[mol_i].z, uz * s);
                        atomicAdd(&out_disp[mol_j].x, -ux * s);
                        atomicAdd(&out_disp[mol_j].y, -uy * s);
                        atomicAdd(&out_disp[mol_j].z, -uz * s);
                    } else if (move_i) {
                        atomicAdd(&out_disp[mol_i].x, ux * overlap);
                        atomicAdd(&out_disp[mol_i].y, uy * overlap);
                        atomicAdd(&out_disp[mol_i].z, uz * overlap);
                    } else if (move_j) {
                        atomicAdd(&out_disp[mol_j].x, -ux * overlap);
                        atomicAdd(&out_disp[mol_j].y, -uy * overlap);
                        atomicAdd(&out_disp[mol_j].z, -uz * overlap);
                    }
                }
            }
        }
    }
}

}
