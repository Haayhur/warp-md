    float r2 = dx * dx + dy * dy + dz * dz;
    if (r2 <= dist2) {
        atomicAdd(&counts[frame], 1);
    }
}

__global__ void hbond_count_angle(const Float4* coords,
                                  const unsigned int* donors,
                                  const unsigned int* hydrogens,
                                  const unsigned int* acceptors,
                                  int n_donors,
                                  int n_acceptors,
                                  int n_atoms,
                                  int n_frames,
                                  float dist2,
                                  float cos_cutoff,
                                  unsigned int* counts) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int frame = blockIdx.y;
    if (frame >= n_frames) return;
    int total = n_donors * n_acceptors;
    if (pair >= total) return;
    int d = pair / n_acceptors;
    int a = pair - d * n_acceptors;
    unsigned int donor = donors[d];
    unsigned int hydrogen = hydrogens[d];
    unsigned int acceptor = acceptors[a];
    if (donor == acceptor) return;
    Float4 pd = coords[frame * n_atoms + donor];
    Float4 ph = coords[frame * n_atoms + hydrogen];
    Float4 pa = coords[frame * n_atoms + acceptor];
    float dx = pa.x - pd.x;
    float dy = pa.y - pd.y;
    float dz = pa.z - pd.z;
    float r2 = dx * dx + dy * dy + dz * dz;
    if (r2 > dist2) return;
    float v1x = pd.x - ph.x;
    float v1y = pd.y - ph.y;
    float v1z = pd.z - ph.z;
    float v2x = pa.x - ph.x;
    float v2y = pa.y - ph.y;
    float v2z = pa.z - ph.z;
    float n1 = v1x * v1x + v1y * v1y + v1z * v1z;
    float n2 = v2x * v2x + v2y * v2y + v2z * v2z;
    if (n1 == 0.0f || n2 == 0.0f) return;
    float cos_val = (v1x * v2x + v1y * v2y + v1z * v2z) / sqrtf(n1 * n2);
    if (cos_val >= cos_cutoff) {
        atomicAdd(&counts[frame], 1);
    }
}

__global__ void msd_time_lag(const float* com,
                             const float* times,
                             const unsigned int* type_ids,
                             const unsigned int* type_counts,
                             int n_groups,
                             int n_types,
                             int n_frames,
                             int ndframe,
                             float dt0,
                             float eps_num,
                             float eps_add,
                             int frame_dec_start,
                             int frame_dec_stride,
                             int frame_dec_enabled,
                             int dt_cut1,
                             int dt_stride1,
                             int dt_cut2,
                             int dt_stride2,
                             int dt_dec_enabled,
                             int axis_enabled,
                             float ax,
                             float ay,
                             float az,
                             float* out,
                             unsigned int* n_diff) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    int g = blockIdx.z;
    if (i >= n_frames || j >= n_frames || g >= n_groups) return;
    if (j <= i) return;
    if (frame_dec_enabled) {
        int idx1 = i + 1;
        if (idx1 > frame_dec_start && (idx1 % frame_dec_stride) != 0) return;
    }
    float dt = times[j] - times[i];
    int idt = (int)floorf((dt + eps_num) / dt0 + eps_add);
    if (idt <= 0 || idt > ndframe) return;
    if (dt_dec_enabled) {
        if (idt > dt_cut2 && (idt % dt_stride2) != 0) return;
        if (idt > dt_cut1 && (idt % dt_stride1) != 0) return;
    }
    if (g == 0) {
        atomicAdd(&n_diff[idt], 1);
    }
    int idx_i = (i * n_groups + g) * 3;
    int idx_j = (j * n_groups + g) * 3;
    float dx = com[idx_i] - com[idx_j];
    float dy = com[idx_i + 1] - com[idx_j + 1];
    float dz = com[idx_i + 2] - com[idx_j + 2];
    float msd_x = dx * dx;
    float msd_y = dy * dy;
    float msd_z = dz * dz;
    float msd_tot = msd_x + msd_y + msd_z;
    unsigned int t = type_ids[g];
    float inv_type = 0.0f;
    if (type_counts[t] > 0) {
        inv_type = 1.0f / (float)type_counts[t];
    }
    float inv_groups = 1.0f / (float)n_groups;
    int components = axis_enabled ? 5 : 4;
    int stride = n_types + 1;
    int base = idt * (components * stride);
    int offset_type = base + t;
    int offset_total = base + n_types;

    atomicAdd(&out[offset_type], msd_x * inv_type);
    atomicAdd(&out[offset_total], msd_x * inv_groups);
    atomicAdd(&out[offset_type + stride], msd_y * inv_type);
    atomicAdd(&out[offset_total + stride], msd_y * inv_groups);
    atomicAdd(&out[offset_type + 2 * stride], msd_z * inv_type);
    atomicAdd(&out[offset_total + 2 * stride], msd_z * inv_groups);
    int comp = 3;
    if (axis_enabled) {
        float proj = dx * ax + dy * ay + dz * az;
        float msd_axis = proj * proj;
        atomicAdd(&out[offset_type + 3 * stride], msd_axis * inv_type);
        atomicAdd(&out[offset_total + 3 * stride], msd_axis * inv_groups);
        comp = 4;
    }
    atomicAdd(&out[offset_type + comp * stride], msd_tot * inv_type);
    atomicAdd(&out[offset_total + comp * stride], msd_tot * inv_groups);
}

__global__ void rotacf_time_lag(const float* orient,
                                const float* times,
                                const unsigned int* type_ids,
                                const unsigned int* type_counts,
                                int n_groups,
                                int n_types,
                                int n_frames,
                                int ndframe,
                                float dt0,
                                float eps_num,
                                float eps_add,
                                int frame_dec_start,
                                int frame_dec_stride,
                                int frame_dec_enabled,
                                int dt_cut1,
                                int dt_stride1,
                                int dt_cut2,
                                int dt_stride2,
                                int dt_dec_enabled,
                                float* corr,
                                float* corr_p2,
                                unsigned int* n_diff) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    int g = blockIdx.z;
    if (i >= n_frames || j >= n_frames || g >= n_groups) return;
    if (j <= i) return;
    if (frame_dec_enabled) {
        int idx1 = i + 1;
        if (idx1 > frame_dec_start && (idx1 % frame_dec_stride) != 0) return;
    }
    float dt = times[j] - times[i];
    int idt = (int)floorf((dt + eps_num) / dt0 + eps_add);
    if (idt <= 0 || idt > ndframe) return;
    if (dt_dec_enabled) {
        if (idt > dt_cut2 && (idt % dt_stride2) != 0) return;
        if (idt > dt_cut1 && (idt % dt_stride1) != 0) return;
    }
    if (g == 0) {
        atomicAdd(&n_diff[idt], 1);
    }
    int idx_i = (i * n_groups + g) * 3;
    int idx_j = (j * n_groups + g) * 3;
    float v1x = orient[idx_i];
    float v1y = orient[idx_i + 1];
    float v1z = orient[idx_i + 2];
    float v2x = orient[idx_j];
    float v2y = orient[idx_j + 1];
    float v2z = orient[idx_j + 2];
    float dot = v1x * v2x + v1y * v2y + v1z * v2z;
    float dot2 = dot * dot;
    unsigned int t = type_ids[g];
    float inv_type = 0.0f;
    if (type_counts[t] > 0) {
        inv_type = 1.0f / (float)type_counts[t];
    }
    float inv_groups = 1.0f / (float)n_groups;
    int stride = n_types + 1;
    int base = idt * stride;
    atomicAdd(&corr[base + t], dot * inv_type);
    atomicAdd(&corr[base + n_types], dot * inv_groups);
    atomicAdd(&corr_p2[base + t], dot2 * inv_type);
    atomicAdd(&corr_p2[base + n_types], dot2 * inv_groups);
}

__global__ void xcorr_time_lag(const float* vec_a,
                               const float* vec_b,
                               int n_frames,
                               int ndframe,
                               float* out,
                               unsigned int* n_diff) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    if (i >= n_frames || j >= n_frames) return;
    if (j <= i) return;
    int lag = j - i;
    if (lag <= 0 || lag > ndframe) return;
    atomicAdd(&n_diff[lag], 1);
    int idx_a = j * 3;
    int idx_b = i * 3;
    float dot = vec_a[idx_a] * vec_b[idx_b]
        + vec_a[idx_a + 1] * vec_b[idx_b + 1]
        + vec_a[idx_a + 2] * vec_b[idx_b + 2];
    atomicAdd(&out[lag], dot);
}

__global__ void timecorr_series_lag(const float* vec_a,
                                    const float* vec_b,
                                    int n_frames,
                                    int n_items,
                                    int ndframe,
                                    float* out,
                                    unsigned int* n_diff) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    int item = blockIdx.z;
    if (i >= n_frames || j >= n_frames || item >= n_items) return;
    if (j <= i) return;
    int lag = j - i;
    if (lag <= 0 || lag > ndframe) return;
    if (item == 0) {
        atomicAdd(&n_diff[lag], 1);
    }
    int idx_j = (j * n_items + item) * 3;
    int idx_i = (i * n_items + item) * 3;
    float dot = vec_a[idx_j] * vec_b[idx_i]
        + vec_a[idx_j + 1] * vec_b[idx_i + 1]
        + vec_a[idx_j + 2] * vec_b[idx_i + 2];
    atomicAdd(&out[lag], dot);
}

__global__ void conductivity_total(const float* com,
                                   const float* times,
                                   const float* charges,
                                   int n_groups,
                                   int n_frames,
                                   int ndframe,
                                   float dt0,
                                   float eps_num,
                                   float eps_add,
                                   int frame_dec_start,
                                   int frame_dec_stride,
                                   int frame_dec_enabled,
                                   int dt_cut1,
                                   int dt_stride1,
                                   int dt_cut2,
                                   int dt_stride2,
                                   int dt_dec_enabled,
                                   int cols,
                                   float* out,
                                   unsigned int* n_diff) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    if (i >= n_frames || j >= n_frames) return;
    if (j <= i) return;
    if (frame_dec_enabled) {
        int idx1 = i + 1;
        if (idx1 > frame_dec_start && (idx1 % frame_dec_stride) != 0) return;
    }
    float dt = times[j] - times[i];
    int idt = (int)floorf((dt + eps_num) / dt0 + eps_add);
    if (idt <= 0 || idt > ndframe) return;
    if (dt_dec_enabled) {
        if (idt > dt_cut2 && (idt % dt_stride2) != 0) return;
        if (idt > dt_cut1 && (idt % dt_stride1) != 0) return;
    }
    atomicAdd(&n_diff[idt], 1);
    float sx = 0.0f;
    float sy = 0.0f;
    float sz = 0.0f;
    for (int g = 0; g < n_groups; g++) {
        int idx_i = (i * n_groups + g) * 3;
        int idx_j = (j * n_groups + g) * 3;
        float dx = (com[idx_i] - com[idx_j]) * charges[g];
        float dy = (com[idx_i + 1] - com[idx_j + 1]) * charges[g];
        float dz = (com[idx_i + 2] - com[idx_j + 2]) * charges[g];
        sx += dx;
        sy += dy;
        sz += dz;
    }
    float dot = sx * sx + sy * sy + sz * sz;
    int base = idt * cols;
    atomicAdd(&out[base + (cols - 1)], dot);
}

__global__ void conductivity_transference(const float* com,
                                          const float* times,
                                          const float* charges,
                                          const unsigned int* type_ids,
                                          const float* type_charge,
                                          int n_groups,
                                          int n_types,
                                          int n_frames,
                                          int ndframe,
                                          float dt0,
                                          float eps_num,
                                          float eps_add,
                                          int frame_dec_start,
                                          int frame_dec_stride,
                                          int frame_dec_enabled,
                                          int dt_cut1,
                                          int dt_stride1,
                                          int dt_cut2,
                                          int dt_stride2,
                                          int dt_dec_enabled,
                                          int cols,
                                          float* out) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    int j = blockIdx.z;
    int total_pairs = n_groups * n_groups;
    if (pair >= total_pairs || i >= n_frames || j >= n_frames) return;
    if (j <= i) return;
    if (frame_dec_enabled) {
        int idx1 = i + 1;
        if (idx1 > frame_dec_start && (idx1 % frame_dec_stride) != 0) return;
    }
    float dt = times[j] - times[i];
    int idt = (int)floorf((dt + eps_num) / dt0 + eps_add);
    if (idt <= 0 || idt > ndframe) return;
    if (dt_dec_enabled) {
        if (idt > dt_cut2 && (idt % dt_stride2) != 0) return;
        if (idt > dt_cut1 && (idt % dt_stride1) != 0) return;
    }
    int k = pair / n_groups;
    int l = pair - k * n_groups;
    if (l < k) return;
    unsigned int type_k = type_ids[k];
    unsigned int type_l = type_ids[l];
    if (type_charge[type_k] == 0.0f || type_charge[type_l] == 0.0f) return;
    int idx_i_k = (i * n_groups + k) * 3;
    int idx_j_k = (j * n_groups + k) * 3;
    int idx_i_l = (i * n_groups + l) * 3;
    int idx_j_l = (j * n_groups + l) * 3;
    float v1x = (com[idx_i_k] - com[idx_j_k]) * charges[k];
    float v1y = (com[idx_i_k + 1] - com[idx_j_k + 1]) * charges[k];
    float v1z = (com[idx_i_k + 2] - com[idx_j_k + 2]) * charges[k];
    float v2x = (com[idx_i_l] - com[idx_j_l]) * charges[l];
    float v2y = (com[idx_i_l + 1] - com[idx_j_l + 1]) * charges[l];
    float v2z = (com[idx_i_l + 2] - com[idx_j_l + 2]) * charges[l];
    float dot = v1x * v2x + v1y * v2y + v1z * v2z;
    float factor = (l == k) ? 1.0f : 2.0f;
    int idx = type_l + type_k * n_types;
    int base = idt * cols;
    atomicAdd(&out[base + idx], factor * dot);
}

__global__ void ion_pair_frame_cat(const float* com,
                                   const float* box_l,
                                   const unsigned int* cat_indices,
                                   const unsigned int* ani_indices,
                                   int n_cat,
                                   int n_ani,
                                   int n_groups,
                                   int n_frames,
                                   int max_cluster,
                                   float rclust,
                                   unsigned int* idx_pair,
                                   unsigned int* idx_cluster) {
    int cat = blockIdx.x * blockDim.x + threadIdx.x;
    int frame = blockIdx.y;
    if (cat >= n_cat || frame >= n_frames) return;
    int pair_offset = frame * n_cat + cat;
    idx_pair[pair_offset] = 0xFFFFFFFFu;
    int cluster_offset = (frame * n_cat + cat) * max_cluster;
    for (int k = 0; k < max_cluster; k++) {
        idx_cluster[cluster_offset + k] = 0xFFFFFFFFu;
    }
    unsigned int cat_group = cat_indices[cat];
    int idx_cat = (frame * n_groups + cat_group) * 3;
    float cx = com[idx_cat];
    float cy = com[idx_cat + 1];
    float cz = com[idx_cat + 2];
    float lx = box_l[frame * 3 + 0];
    float ly = box_l[frame * 3 + 1];
    float lz = box_l[frame * 3 + 2];
    float best = 1.0e30f;
    for (int a = 0; a < n_ani; a++) {
        unsigned int ani_group = ani_indices[a];
        int idx_ani = (frame * n_groups + ani_group) * 3;
        float dx = com[idx_ani] - cx;
        float dy = com[idx_ani + 1] - cy;
        float dz = com[idx_ani + 2] - cz;
        if (lx > 0.0f) {
            dx -= roundf(dx / lx) * lx;
            dy -= roundf(dy / ly) * ly;
            dz -= roundf(dz / lz) * lz;
        }
        float r2 = dx * dx + dy * dy + dz * dz;
        float r = sqrtf(r2);
        if (r < rclust) {
            for (int k = 0; k < max_cluster; k++) {
                unsigned int* slot = &idx_cluster[cluster_offset + k];
                if (*slot == 0xFFFFFFFFu) {
                    *slot = ani_group;
                    break;
                }
            }
            if (r < best) {
                best = r;
                idx_pair[pair_offset] = ani_group;
            }
        }
    }
}

__global__ void ion_pair_frame_ani(const float* com,
                                   const float* box_l,
                                   const unsigned int* cat_indices,
                                   const unsigned int* ani_indices,
                                   int n_cat,
                                   int n_ani,
                                   int n_groups,
                                   int n_frames,
                                   int max_cluster,
                                   float rclust,
                                   unsigned int* idx_pair,
                                   unsigned int* idx_cluster) {
    int ani = blockIdx.x * blockDim.x + threadIdx.x;
    int frame = blockIdx.y;
    if (ani >= n_ani || frame >= n_frames) return;
    int pair_offset = frame * n_ani + ani;
    idx_pair[pair_offset] = 0xFFFFFFFFu;
    int cluster_offset = (frame * n_ani + ani) * max_cluster;
    for (int k = 0; k < max_cluster; k++) {
        idx_cluster[cluster_offset + k] = 0xFFFFFFFFu;
    }
    unsigned int ani_group = ani_indices[ani];
    int idx_ani = (frame * n_groups + ani_group) * 3;
    float cx = com[idx_ani];
    float cy = com[idx_ani + 1];
    float cz = com[idx_ani + 2];
    float lx = box_l[frame * 3 + 0];
    float ly = box_l[frame * 3 + 1];
    float lz = box_l[frame * 3 + 2];
    float best = 1.0e30f;
    for (int a = 0; a < n_cat; a++) {
        unsigned int cat_group = cat_indices[a];
        int idx_cat = (frame * n_groups + cat_group) * 3;
        float dx = com[idx_cat] - cx;
        float dy = com[idx_cat + 1] - cy;
        float dz = com[idx_cat + 2] - cz;
        if (lx > 0.0f) {
            dx -= roundf(dx / lx) * lx;
            dy -= roundf(dy / ly) * ly;
            dz -= roundf(dz / lz) * lz;
        }
        float r2 = dx * dx + dy * dy + dz * dz;
        float r = sqrtf(r2);
        if (r < rclust) {
            for (int k = 0; k < max_cluster; k++) {
                unsigned int* slot = &idx_cluster[cluster_offset + k];
                if (*slot == 0xFFFFFFFFu) {
                    *slot = cat_group;
                    break;
                }
            }
            if (r < best) {
                best = r;
                idx_pair[pair_offset] = cat_group;
            }
        }
    }
}

__global__ void ion_pair_corr_cat(const unsigned int* idx_pair,
                                  const unsigned int* idx_cluster,
                                  int n_frames,
                                  int n_cat,
                                  int max_cluster,
                                  unsigned int* ip_out,
                                  unsigned int* cp_out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    int cat = blockIdx.z;
    if (i >= n_frames || j >= n_frames || cat >= n_cat) return;
    if (j < i) return;
    int dt = j - i;
    unsigned int a = idx_pair[i * n_cat + cat];
    unsigned int b = idx_pair[j * n_cat + cat];
    if (a == b) {
        atomicAdd(&ip_out[dt], 1);
    }
    int off_i = (i * n_cat + cat) * max_cluster;
    int off_j = (j * n_cat + cat) * max_cluster;
    for (int k = 0; k < max_cluster; k++) {
