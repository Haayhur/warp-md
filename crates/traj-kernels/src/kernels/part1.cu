
extern "C" {
struct Float4 { float x; float y; float z; float w; };

__device__ __forceinline__ float wrap_pbc(float d, float box) {
    if (box > 0.0f) {
        d -= roundf(d / box) * box;
    }
    return d;
}

__device__ __forceinline__ void wrap_triclinic(float* dx, float* dy, float* dz,
                                               Float4 c0, Float4 c1, Float4 c2,
                                               Float4 r0, Float4 r1, Float4 r2) {
    float x = *dx;
    float y = *dy;
    float z = *dz;
    float f0 = r0.x * x + r1.x * y + r2.x * z;
    float f1 = r0.y * x + r1.y * y + r2.y * z;
    float f2 = r0.z * x + r1.z * y + r2.z * z;
    f0 -= roundf(f0);
    f1 -= roundf(f1);
    f2 -= roundf(f2);
    *dx = f0 * c0.x + f1 * c1.x + f2 * c2.x;
    *dy = f0 * c0.y + f1 * c1.y + f2 * c2.y;
    *dz = f0 * c0.z + f1 * c1.z + f2 * c2.z;
}

__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
    int* addr_as_i = (int*)addr;
    int old = *addr_as_i;
    int assumed;
    while (value < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(addr_as_i, assumed, __float_as_int(value));
        if (assumed == old) break;
    }
    return __int_as_float(old);
}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    int* addr_as_i = (int*)addr;
    int old = *addr_as_i;
    int assumed;
    while (value > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(addr_as_i, assumed, __float_as_int(value));
        if (assumed == old) break;
    }
    return __int_as_float(old);
}

__device__ __forceinline__ float angle_from_vectors(float v1x, float v1y, float v1z,
                                                    float v2x, float v2y, float v2z,
                                                    int degrees) {
    float dot = v1x * v2x + v1y * v2y + v1z * v2z;
    float n1 = sqrtf(v1x * v1x + v1y * v1y + v1z * v1z);
    float n2 = sqrtf(v2x * v2x + v2y * v2y + v2z * v2z);
    if (n1 == 0.0f || n2 == 0.0f) return 0.0f;
    float cosv = dot / (n1 * n2);
    if (cosv > 1.0f) cosv = 1.0f;
    if (cosv < -1.0f) cosv = -1.0f;
    float angle = acosf(cosv);
    if (degrees) angle *= 57.29577951308232f;
    return angle;
}

__device__ __forceinline__ float dihedral_from_vectors(float b0x, float b0y, float b0z,
                                                       float b1x, float b1y, float b1z,
                                                       float b2x, float b2y, float b2z,
                                                       int degrees, int range360) {
    float n = sqrtf(b1x * b1x + b1y * b1y + b1z * b1z);
    if (n == 0.0f) return 0.0f;
    float b1nx = b1x / n;
    float b1ny = b1y / n;
    float b1nz = b1z / n;
    float dot_b0 = b0x * b1nx + b0y * b1ny + b0z * b1nz;
    float dot_b2 = b2x * b1nx + b2y * b1ny + b2z * b1nz;
    float vx = b0x - dot_b0 * b1nx;
    float vy = b0y - dot_b0 * b1ny;
    float vz = b0z - dot_b0 * b1nz;
    float wx = b2x - dot_b2 * b1nx;
    float wy = b2y - dot_b2 * b1ny;
    float wz = b2z - dot_b2 * b1nz;
    float x = vx * wx + vy * wy + vz * wz;
    float mx = b1ny * vz - b1nz * vy;
    float my = b1nz * vx - b1nx * vz;
    float mz = b1nx * vy - b1ny * vx;
    float y = mx * wx + my * wy + mz * wz;
    float angle = atan2f(y, x);
    if (degrees) {
        angle *= 57.29577951308232f;
        if (range360 && angle < 0.0f) angle += 360.0f;
    } else if (range360 && angle < 0.0f) {
        angle += 6.283185307179586f;
    }
    return angle;
}

__global__ void angle_from_com(const Float4* coms,
                               int n_frames,
                               const Float4* boxes,
                               int degrees,
                               float* out) {
    int frame = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames) return;
    int base = frame * 3;
    Float4 a = coms[base];
    Float4 b = coms[base + 1];
    Float4 c = coms[base + 2];
    float v1x = a.x - b.x;
    float v1y = a.y - b.y;
    float v1z = a.z - b.z;
    float v2x = c.x - b.x;
    float v2y = c.y - b.y;
    float v2z = c.z - b.z;
    Float4 box = boxes[frame];
    v1x = wrap_pbc(v1x, box.x);
    v1y = wrap_pbc(v1y, box.y);
    v1z = wrap_pbc(v1z, box.z);
    v2x = wrap_pbc(v2x, box.x);
    v2y = wrap_pbc(v2y, box.y);
    v2z = wrap_pbc(v2z, box.z);
    out[frame] = angle_from_vectors(v1x, v1y, v1z, v2x, v2y, v2z, degrees);
}

__global__ void dihedral_from_com(const Float4* coms,
                                  int n_frames,
                                  const Float4* boxes,
                                  int degrees,
                                  int range360,
                                  float* out) {
    int frame = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames) return;
    int base = frame * 4;
    Float4 a = coms[base];
    Float4 b = coms[base + 1];
    Float4 c = coms[base + 2];
    Float4 d = coms[base + 3];
    float b0x = a.x - b.x;
    float b0y = a.y - b.y;
    float b0z = a.z - b.z;
    float b1x = c.x - b.x;
    float b1y = c.y - b.y;
    float b1z = c.z - b.z;
    float b2x = d.x - c.x;
    float b2y = d.y - c.y;
    float b2z = d.z - c.z;
    Float4 box = boxes[frame];
    b0x = wrap_pbc(b0x, box.x);
    b0y = wrap_pbc(b0y, box.y);
    b0z = wrap_pbc(b0z, box.z);
    b1x = wrap_pbc(b1x, box.x);
    b1y = wrap_pbc(b1y, box.y);
    b1z = wrap_pbc(b1z, box.z);
    b2x = wrap_pbc(b2x, box.x);
    b2y = wrap_pbc(b2y, box.y);
    b2z = wrap_pbc(b2z, box.z);
    out[frame] = dihedral_from_vectors(b0x, b0y, b0z, b1x, b1y, b1z, b2x, b2y, b2z, degrees, range360);
}

__global__ void distance_from_com(const Float4* coms,
                                  int n_frames,
                                  const Float4* boxes,
                                  float* out) {
    int frame = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames) return;
    int base = frame * 2;
    Float4 a = coms[base];
    Float4 b = coms[base + 1];
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float dz = b.z - a.z;
    Float4 box = boxes[frame];
    dx = wrap_pbc(dx, box.x);
    dy = wrap_pbc(dy, box.y);
    dz = wrap_pbc(dz, box.z);
    out[frame] = sqrtf(dx * dx + dy * dy + dz * dz);
}

__global__ void distance_from_com_min(const Float4* coms,
                                      int n_frames,
                                      const Float4* boxes,
                                      float* out_min) {
    int frame = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames) return;
    int base = frame * 2;
    Float4 a = coms[base];
    Float4 b = coms[base + 1];
    float dx = b.x - a.x;
    float dy = b.y - a.y;
    float dz = b.z - a.z;
    Float4 box = boxes[frame];
    dx = wrap_pbc(dx, box.x);
    dy = wrap_pbc(dy, box.y);
    dz = wrap_pbc(dz, box.z);
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    atomicMinFloat(out_min, dist);
}

__global__ void rg_accum(const Float4* coords,
                         const unsigned int* sel,
                         const float* masses,
                         int n_sel,
                         int n_atoms,
                         int n_frames,
                         int mass_weighted,
                         float* sum_x,
                         float* sum_y,
                         float* sum_z,
                         float* mass_sum) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_sel) return;
    unsigned int atom = sel[idx];
    Float4 p = coords[frame * n_atoms + atom];
    float m = mass_weighted ? masses[idx] : 1.0f;
    atomicAdd(&sum_x[frame], p.x * m);
    atomicAdd(&sum_y[frame], p.y * m);
    atomicAdd(&sum_z[frame], p.z * m);
    atomicAdd(&mass_sum[frame], m);
}

__global__ void rg_sumsq(const Float4* coords,
                         const unsigned int* sel,
                         const float* masses,
                         int n_sel,
                         int n_atoms,
                         int n_frames,
                         int mass_weighted,
                         const float* sum_x,
                         const float* sum_y,
                         const float* sum_z,
                         const float* mass_sum,
                         float* sum_sq) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_sel) return;
    float msum = mass_sum[frame];
    if (msum == 0.0f) return;
    float cx = sum_x[frame] / msum;
    float cy = sum_y[frame] / msum;
    float cz = sum_z[frame] / msum;
    unsigned int atom = sel[idx];
    Float4 p = coords[frame * n_atoms + atom];
    float dx = p.x - cx;
    float dy = p.y - cy;
    float dz = p.z - cz;
    float m = mass_weighted ? masses[idx] : 1.0f;
    atomicAdd(&sum_sq[frame], m * (dx * dx + dy * dy + dz * dz));
}

__global__ void rg_finalize(const float* mass_sum,
                            const float* sum_sq,
                            int n_frames,
                            float* out) {
    int frame = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames) return;
    float m = mass_sum[frame];
    if (m == 0.0f) {
        out[frame] = 0.0f;
    } else {
        out[frame] = sqrtf(sum_sq[frame] / m);
    }
}

__global__ void msd_accum(const Float4* coords,
                          const Float4* origin,
                          const unsigned int* sel,
                          int n_sel,
                          int n_atoms,
                          int n_frames,
                          float* sum_sq) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_sel) return;
    unsigned int atom = sel[idx];
    Float4 p = coords[frame * n_atoms + atom];
    Float4 r0 = origin[idx];
    float dx = p.x - r0.x;
    float dy = p.y - r0.y;
    float dz = p.z - r0.z;
    atomicAdd(&sum_sq[frame], dx * dx + dy * dy + dz * dz);
}

__global__ void msd_finalize(const float* sum_sq,
                             int n_frames,
                             int n_sel,
                             float* out) {
    int frame = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames) return;
    if (n_sel == 0) {
        out[frame] = 0.0f;
    } else {
        out[frame] = sum_sq[frame] / (float)n_sel;
    }
}

__global__ void rmsd_centroid(const Float4* coords,
                              const Float4* ref,
                              const unsigned int* sel,
                              int n_sel,
                              int n_atoms,
                              int n_frames,
                              float* sum_x,
                              float* sum_y) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_sel) return;
    unsigned int atom = sel[idx];
    Float4 p = coords[frame * n_atoms + atom];
    Float4 r = ref[idx];
    atomicAdd(&sum_x[frame * 3 + 0], p.x);
    atomicAdd(&sum_x[frame * 3 + 1], p.y);
    atomicAdd(&sum_x[frame * 3 + 2], p.z);
    atomicAdd(&sum_y[frame * 3 + 0], r.x);
    atomicAdd(&sum_y[frame * 3 + 1], r.y);
    atomicAdd(&sum_y[frame * 3 + 2], r.z);
}

__global__ void rmsd_cov(const Float4* coords,
                         const Float4* ref,
                         const unsigned int* sel,
                         int n_sel,
                         int n_atoms,
                         int n_frames,
                         const float* sum_x,
                         const float* sum_y,
                         float* cov,
                         float* sum_x2,
                         float* sum_y2) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_sel) return;
    float cx = sum_x[frame * 3 + 0] / (float)n_sel;
    float cy = sum_x[frame * 3 + 1] / (float)n_sel;
    float cz = sum_x[frame * 3 + 2] / (float)n_sel;
    float rx = sum_y[frame * 3 + 0] / (float)n_sel;
    float ry = sum_y[frame * 3 + 1] / (float)n_sel;
    float rz = sum_y[frame * 3 + 2] / (float)n_sel;
    unsigned int atom = sel[idx];
    Float4 p = coords[frame * n_atoms + atom];
    Float4 r = ref[idx];
    float x0 = p.x - cx;
    float x1 = p.y - cy;
    float x2 = p.z - cz;
    float y0 = r.x - rx;
    float y1 = r.y - ry;
    float y2 = r.z - rz;
    atomicAdd(&cov[frame * 9 + 0], x0 * y0);
    atomicAdd(&cov[frame * 9 + 1], x0 * y1);
    atomicAdd(&cov[frame * 9 + 2], x0 * y2);
    atomicAdd(&cov[frame * 9 + 3], x1 * y0);
    atomicAdd(&cov[frame * 9 + 4], x1 * y1);
    atomicAdd(&cov[frame * 9 + 5], x1 * y2);
    atomicAdd(&cov[frame * 9 + 6], x2 * y0);
    atomicAdd(&cov[frame * 9 + 7], x2 * y1);
    atomicAdd(&cov[frame * 9 + 8], x2 * y2);
    atomicAdd(&sum_x2[frame], x0 * x0 + x1 * x1 + x2 * x2);
    atomicAdd(&sum_y2[frame], y0 * y0 + y1 * y1 + y2 * y2);
}

__global__ void rmsd_raw_accum(const Float4* coords,
                               const Float4* ref,
                               const unsigned int* sel,
                               int n_sel,
                               int n_atoms,
                               int n_frames,
                               float* sum_sq) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_sel) return;
    unsigned int atom = sel[idx];
    Float4 p = coords[frame * n_atoms + atom];
    Float4 r = ref[idx];
    float dx = p.x - r.x;
    float dy = p.y - r.y;
    float dz = p.z - r.z;
    atomicAdd(&sum_sq[frame], dx * dx + dy * dy + dz * dz);
}

__global__ void rmsd_finalize(const float* sum_sq,
                              int n_frames,
                              int n_sel,
                              float* out) {
    int frame = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames) return;
    if (n_sel == 0) {
        out[frame] = 0.0f;
    } else {
        out[frame] = sqrtf(sum_sq[frame] / (float)n_sel);
    }
}

__global__ void distance_to_point(const Float4* coords,
                                  const unsigned int* sel,
                                  int n_sel,
                                  int n_atoms,
                                  int n_frames,
                                  Float4 point,
                                  const Float4* boxes,
                                  float* out) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_sel) return;
    unsigned int atom = sel[idx];
    Float4 p = coords[frame * n_atoms + atom];
    Float4 box = boxes[frame];
    float dx = p.x - point.x;
    float dy = p.y - point.y;
    float dz = p.z - point.z;
    dx = wrap_pbc(dx, box.x);
    dy = wrap_pbc(dy, box.y);
    dz = wrap_pbc(dz, box.z);
    out[frame * n_sel + idx] = sqrtf(dx * dx + dy * dy + dz * dz);
}

__global__ void distance_to_reference(const Float4* coords,
                                      const Float4* ref,
                                      const unsigned int* sel,
                                      int n_sel,
                                      int n_atoms,
                                      int n_frames,
                                      const Float4* boxes,
                                      float* out) {
    int frame = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= n_frames || idx >= n_sel) return;
    unsigned int atom = sel[idx];
    Float4 p = coords[frame * n_atoms + atom];
    Float4 r = ref[idx];
    Float4 box = boxes[frame];
    float dx = p.x - r.x;
    float dy = p.y - r.y;
    float dz = p.z - r.z;
    dx = wrap_pbc(dx, box.x);
    dy = wrap_pbc(dy, box.y);
    dz = wrap_pbc(dz, box.z);
    out[frame * n_sel + idx] = sqrtf(dx * dx + dy * dy + dz * dz);
}

__global__ void pairwise_distance(const Float4* coords,
                                  const unsigned int* sel_a,
                                  const unsigned int* sel_b,
                                  int n_sel_a,
                                  int n_sel_b,
                                  int n_atoms,
                                  int n_frames,
                                  const Float4* boxes,
                                  float* out) {
    int frame = blockIdx.y;
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int n_pairs = n_sel_a * n_sel_b;
    if (frame >= n_frames || pair >= n_pairs) return;
    int a = pair / n_sel_b;
    int b = pair - a * n_sel_b;
    unsigned int atom_a = sel_a[a];
    unsigned int atom_b = sel_b[b];
    Float4 pa = coords[frame * n_atoms + atom_a];
    Float4 pb = coords[frame * n_atoms + atom_b];
    Float4 box = boxes[frame];
    float dx = pb.x - pa.x;
    float dy = pb.y - pa.y;
    float dz = pb.z - pa.z;
    dx = wrap_pbc(dx, box.x);
    dy = wrap_pbc(dy, box.y);
    dz = wrap_pbc(dz, box.z);
    out[frame * n_pairs + pair] = sqrtf(dx * dx + dy * dy + dz * dz);
}

__global__ void mindist_pairs(const Float4* coords,
                              const unsigned int* sel_a,
                              const unsigned int* sel_b,
                              int n_sel_a,
                              int n_sel_b,
                              int n_atoms,
                              int n_frames,
                              const Float4* boxes,
                              float* out) {
    int frame = blockIdx.x;
    int tid = threadIdx.x;
    int n_pairs = n_sel_a * n_sel_b;
    if (frame >= n_frames) return;
    float min_val = 1.0e30f;
    for (int pair = tid; pair < n_pairs; pair += blockDim.x) {
        int a = pair / n_sel_b;
        int b = pair - a * n_sel_b;
        unsigned int atom_a = sel_a[a];
        unsigned int atom_b = sel_b[b];
        Float4 pa = coords[frame * n_atoms + atom_a];
        Float4 pb = coords[frame * n_atoms + atom_b];
        Float4 box = boxes[frame];
        float dx = pb.x - pa.x;
        float dy = pb.y - pa.y;
        float dz = pb.z - pa.z;
        dx = wrap_pbc(dx, box.x);
        dy = wrap_pbc(dy, box.y);
        dz = wrap_pbc(dz, box.z);
        float dist = sqrtf(dx * dx + dy * dy + dz * dz);
        if (dist < min_val) min_val = dist;
