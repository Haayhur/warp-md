use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig,
    PushKernelArg, ValidAsZeroBits,
};
use cudarc::nvrtc::compile_ptx;

use traj_core::error::{TrajError, TrajResult};
use traj_kernels::KERNELS_SRC;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Float4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

unsafe impl DeviceRepr for Float4 {}
unsafe impl ValidAsZeroBits for Float4 {}

pub fn convert_coords(coords: &[[f32; 4]]) -> Vec<Float4> {
    coords
        .iter()
        .map(|c| Float4 {
            x: c[0],
            y: c[1],
            z: c[2],
            w: c[3],
        })
        .collect()
}

pub fn coords_as_float4(coords: &[[f32; 4]]) -> &[Float4] {
    debug_assert_eq!(
        std::mem::size_of::<[f32; 4]>(),
        std::mem::size_of::<Float4>()
    );
    debug_assert_eq!(
        std::mem::align_of::<[f32; 4]>(),
        std::mem::align_of::<Float4>()
    );
    // SAFETY:
    // - `[f32; 4]` and `Float4` have identical size/alignment.
    // - `Float4` is `#[repr(C)]` of four `f32` fields.
    // - The returned slice borrows the same memory and preserves length.
    unsafe { std::slice::from_raw_parts(coords.as_ptr() as *const Float4, coords.len()) }
}

#[derive(Clone)]
pub struct GpuContext {
    inner: Arc<GpuContextInner>,
}

struct GpuContextInner {
    stream: Arc<CudaStream>,
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    kernels: Kernels,
}

struct Kernels {
    rg_accum: Arc<CudaFunction>,
    rg_sumsq: Arc<CudaFunction>,
    rg_finalize: Arc<CudaFunction>,
    msd_accum: Arc<CudaFunction>,
    msd_finalize: Arc<CudaFunction>,
    rmsd_centroid: Arc<CudaFunction>,
    rmsd_cov: Arc<CudaFunction>,
    rmsd_raw_accum: Arc<CudaFunction>,
    rmsd_finalize: Arc<CudaFunction>,
    distance_to_point: Arc<CudaFunction>,
    distance_to_reference: Arc<CudaFunction>,
    pairwise_distance: Arc<CudaFunction>,
    distance_from_com_min: Arc<CudaFunction>,
    mindist_pairs: Arc<CudaFunction>,
    closest_atom_point: Arc<CudaFunction>,
    search_neighbors_count: Arc<CudaFunction>,
    min_dist_a: Arc<CudaFunction>,
    min_dist_b: Arc<CudaFunction>,
    min_dist_a_triclinic: Arc<CudaFunction>,
    min_dist_b_triclinic: Arc<CudaFunction>,
    min_dist_points: Arc<CudaFunction>,
    max_dist_points: Arc<CudaFunction>,
    multipucker_histogram: Arc<CudaFunction>,
    multipucker_distances: Arc<CudaFunction>,
    multipucker_histogram_from_distances: Arc<CudaFunction>,
    multipucker_normalize_rows: Arc<CudaFunction>,
    hausdorff_reduce: Arc<CudaFunction>,
    atom_map_pairs: Arc<CudaFunction>,
    closest_min_dist: Arc<CudaFunction>,
    closest_topk: Arc<CudaFunction>,
    rotate_dihedral_apply: Arc<CudaFunction>,
    image_coords: Arc<CudaFunction>,
    chirality_volume: Arc<CudaFunction>,
    torsion_diffusion_counts: Arc<CudaFunction>,
    bbox_minmax: Arc<CudaFunction>,
    bbox_area: Arc<CudaFunction>,
    sasa_approx: Arc<CudaFunction>,
    volume_orthorhombic: Arc<CudaFunction>,
    volume_cell: Arc<CudaFunction>,
    randomize_ions_apply: Arc<CudaFunction>,
    shift_coords: Arc<CudaFunction>,
    translate_coords: Arc<CudaFunction>,
    scale_coords: Arc<CudaFunction>,
    transform_coords: Arc<CudaFunction>,
    native_contacts_count: Arc<CudaFunction>,
    gather_selection: Arc<CudaFunction>,
    replicate_cell: Arc<CudaFunction>,
    rmsf_accum: Arc<CudaFunction>,
    mean_structure_accum: Arc<CudaFunction>,
    align_centroid: Arc<CudaFunction>,
    align_cov: Arc<CudaFunction>,
    inertia_tensor: Arc<CudaFunction>,
    rmsd_per_res_accum: Arc<CudaFunction>,
    rdf_hist: Arc<CudaFunction>,
    polymer_end_to_end: Arc<CudaFunction>,
    polymer_contour_length: Arc<CudaFunction>,
    polymer_chain_rg: Arc<CudaFunction>,
    polymer_bond_hist: Arc<CudaFunction>,
    polymer_angle_hist: Arc<CudaFunction>,
    group_com_accum: Arc<CudaFunction>,
    group_com_finalize: Arc<CudaFunction>,
    group_dipole_accum: Arc<CudaFunction>,
    group_dipole_finalize: Arc<CudaFunction>,
    group_ke_accum: Arc<CudaFunction>,
    orientation_plane: Arc<CudaFunction>,
    orientation_vector: Arc<CudaFunction>,
    orientation_vector_pbc: Arc<CudaFunction>,
    angle_from_com: Arc<CudaFunction>,
    dihedral_from_com: Arc<CudaFunction>,
    distance_from_com: Arc<CudaFunction>,
    water_count: Arc<CudaFunction>,
    gist_direct_energy: Arc<CudaFunction>,
    gist_counts_orient: Arc<CudaFunction>,
    gist_accumulate_hist: Arc<CudaFunction>,
    hbond_count: Arc<CudaFunction>,
    hbond_count_angle: Arc<CudaFunction>,
    msd_time_lag: Arc<CudaFunction>,
    rotacf_time_lag: Arc<CudaFunction>,
    xcorr_time_lag: Arc<CudaFunction>,
    timecorr_series_lag: Arc<CudaFunction>,
    conductivity_total: Arc<CudaFunction>,
    conductivity_transference: Arc<CudaFunction>,
    ion_pair_frame_cat: Arc<CudaFunction>,
    ion_pair_frame_ani: Arc<CudaFunction>,
    ion_pair_corr_cat: Arc<CudaFunction>,
    ion_pair_corr_ani: Arc<CudaFunction>,
    pack_overlap_max: Arc<CudaFunction>,
    pack_overlap_max_cells: Arc<CudaFunction>,
    pack_overlap_max_cells_movable: Arc<CudaFunction>,
    pack_overlap_penalty_cells: Arc<CudaFunction>,
    pack_overlap_grad_cells: Arc<CudaFunction>,
    pack_short_tol_penalty_grad_cells: Arc<CudaFunction>,
    pack_constraint_penalty: Arc<CudaFunction>,
    pack_relax_accum: Arc<CudaFunction>,
}

impl Kernels {
    fn load(module: &Arc<CudaModule>) -> TrajResult<Self> {
        let load = |name: &str| -> TrajResult<Arc<CudaFunction>> {
            module
                .load_function(name)
                .map_err(|err| {
                    TrajError::Unsupported(format!("cuda kernel load '{name}' failed: {err}"))
                })
                .map(Arc::new)
        };
        Ok(Self {
            rg_accum: load("rg_accum")?,
            rg_sumsq: load("rg_sumsq")?,
            rg_finalize: load("rg_finalize")?,
            msd_accum: load("msd_accum")?,
            msd_finalize: load("msd_finalize")?,
            rmsd_centroid: load("rmsd_centroid")?,
            rmsd_cov: load("rmsd_cov")?,
            rmsd_raw_accum: load("rmsd_raw_accum")?,
            rmsd_finalize: load("rmsd_finalize")?,
            distance_to_point: load("distance_to_point")?,
            distance_to_reference: load("distance_to_reference")?,
            pairwise_distance: load("pairwise_distance")?,
            distance_from_com_min: load("distance_from_com_min")?,
            mindist_pairs: load("mindist_pairs")?,
            closest_atom_point: load("closest_atom_point")?,
            search_neighbors_count: load("search_neighbors_count")?,
            min_dist_a: load("min_dist_a")?,
            min_dist_b: load("min_dist_b")?,
            min_dist_a_triclinic: load("min_dist_a_triclinic")?,
            min_dist_b_triclinic: load("min_dist_b_triclinic")?,
            min_dist_points: load("min_dist_points")?,
            max_dist_points: load("max_dist_points")?,
            multipucker_histogram: load("multipucker_histogram")?,
            multipucker_distances: load("multipucker_distances")?,
            multipucker_histogram_from_distances: load("multipucker_histogram_from_distances")?,
            multipucker_normalize_rows: load("multipucker_normalize_rows")?,
            hausdorff_reduce: load("hausdorff_reduce")?,
            atom_map_pairs: load("atom_map_pairs")?,
            closest_min_dist: load("closest_min_dist")?,
            closest_topk: load("closest_topk")?,
            rotate_dihedral_apply: load("rotate_dihedral_apply")?,
            image_coords: load("image_coords")?,
            chirality_volume: load("chirality_volume")?,
            torsion_diffusion_counts: load("torsion_diffusion_counts")?,
            bbox_minmax: load("bbox_minmax")?,
            bbox_area: load("bbox_area")?,
            sasa_approx: load("sasa_approx")?,
            volume_orthorhombic: load("volume_orthorhombic")?,
            volume_cell: load("volume_cell")?,
            randomize_ions_apply: load("randomize_ions_apply")?,
            shift_coords: load("shift_coords")?,
            translate_coords: load("translate_coords")?,
            scale_coords: load("scale_coords")?,
            transform_coords: load("transform_coords")?,
            native_contacts_count: load("native_contacts_count")?,
            gather_selection: load("gather_selection")?,
            replicate_cell: load("replicate_cell")?,
            rmsf_accum: load("rmsf_accum")?,
            mean_structure_accum: load("mean_structure_accum")?,
            align_centroid: load("align_centroid")?,
            align_cov: load("align_cov")?,
            inertia_tensor: load("inertia_tensor")?,
            rmsd_per_res_accum: load("rmsd_per_res_accum")?,
            rdf_hist: load("rdf_hist")?,
            polymer_end_to_end: load("polymer_end_to_end")?,
            polymer_contour_length: load("polymer_contour_length")?,
            polymer_chain_rg: load("polymer_chain_rg")?,
            polymer_bond_hist: load("polymer_bond_hist")?,
            polymer_angle_hist: load("polymer_angle_hist")?,
            group_com_accum: load("group_com_accum")?,
            group_com_finalize: load("group_com_finalize")?,
            group_dipole_accum: load("group_dipole_accum")?,
            group_dipole_finalize: load("group_dipole_finalize")?,
            group_ke_accum: load("group_ke_accum")?,
            orientation_plane: load("orientation_plane")?,
            orientation_vector: load("orientation_vector")?,
            orientation_vector_pbc: load("orientation_vector_pbc")?,
            angle_from_com: load("angle_from_com")?,
            dihedral_from_com: load("dihedral_from_com")?,
            distance_from_com: load("distance_from_com")?,
            water_count: load("water_count")?,
            gist_direct_energy: load("gist_direct_energy")?,
            gist_counts_orient: load("gist_counts_orient")?,
            gist_accumulate_hist: load("gist_accumulate_hist")?,
            hbond_count: load("hbond_count")?,
            hbond_count_angle: load("hbond_count_angle")?,
            msd_time_lag: load("msd_time_lag")?,
            rotacf_time_lag: load("rotacf_time_lag")?,
            xcorr_time_lag: load("xcorr_time_lag")?,
            timecorr_series_lag: load("timecorr_series_lag")?,
            conductivity_total: load("conductivity_total")?,
            conductivity_transference: load("conductivity_transference")?,
            ion_pair_frame_cat: load("ion_pair_frame_cat")?,
            ion_pair_frame_ani: load("ion_pair_frame_ani")?,
            ion_pair_corr_cat: load("ion_pair_corr_cat")?,
            ion_pair_corr_ani: load("ion_pair_corr_ani")?,
            pack_overlap_max: load("pack_overlap_max")?,
            pack_overlap_max_cells: load("pack_overlap_max_cells")?,
            pack_overlap_max_cells_movable: load("pack_overlap_max_cells_movable")?,
            pack_overlap_penalty_cells: load("pack_overlap_penalty_cells")?,
            pack_overlap_grad_cells: load("pack_overlap_grad_cells")?,
            pack_short_tol_penalty_grad_cells: load("pack_short_tol_penalty_grad_cells")?,
            pack_constraint_penalty: load("pack_constraint_penalty")?,
            pack_relax_accum: load("pack_relax_accum")?,
        })
    }
}

mod ops_part1;
mod ops_part2;
mod ops_part3;
mod ops_part4;
mod ops_part5;
mod ops_part6;
mod ops_part7;
mod ops_part8;

pub struct GpuSelection {
    sel: CudaSlice<u32>,
    masses: CudaSlice<f32>,
    n_sel: usize,
}

impl GpuSelection {
    pub fn n_sel(&self) -> usize {
        self.n_sel
    }
}

pub struct GpuReference {
    coords: CudaSlice<Float4>,
    n_sel: usize,
}

impl GpuReference {
    pub fn n_sel(&self) -> usize {
        self.n_sel
    }
}

pub struct GpuPolymer {
    chain_offsets: CudaSlice<u32>,
    chain_indices: CudaSlice<u32>,
    n_chains: usize,
    bond_pairs: Option<CudaSlice<u32>>,
    n_bonds: usize,
    angle_triplets: Option<CudaSlice<u32>>,
    n_angles: usize,
}

pub struct GpuBufferF32 {
    inner: CudaSlice<f32>,
}

pub struct GpuBufferU32 {
    #[allow(dead_code)]
    inner: CudaSlice<u32>,
}

pub struct GpuPairs {
    pairs: CudaSlice<u32>,
    n_pairs: usize,
}

pub struct GpuGroups {
    offsets: CudaSlice<u32>,
    indices: CudaSlice<u32>,
    n_groups: usize,
    max_len: usize,
}

pub struct GpuAnchors {
    anchors: CudaSlice<u32>,
    n_groups: usize,
}

pub struct GpuCountsU32 {
    inner: CudaSlice<u32>,
}

pub struct GpuCoords {
    inner: CudaSlice<Float4>,
}

pub struct RmsdCovariance {
    pub cov: Vec<[f32; 9]>,
    pub sum_x2: Vec<f32>,
    pub sum_y2: Vec<f32>,
}

pub struct RmsfAccum {
    pub sum_x: Vec<f32>,
    pub sum_y: Vec<f32>,
    pub sum_z: Vec<f32>,
    pub sum_sq: Vec<f32>,
}

pub struct MeanStructureAccum {
    pub sum_x: Vec<f32>,
    pub sum_y: Vec<f32>,
    pub sum_z: Vec<f32>,
}

pub struct AlignCovariance {
    pub cov: Vec<[f32; 9]>,
    pub sum_x: Vec<[f32; 3]>,
    pub sum_y: Vec<[f32; 3]>,
    pub sum_w: Vec<f32>,
}

pub struct GpuCounts {
    inner: CudaSlice<u64>,
}

fn ceil_div(value: usize, block: u32) -> u32 {
    if value == 0 {
        1
    } else {
        ((value as u32) + block - 1) / block
    }
}

fn map_driver_err(err: cudarc::driver::DriverError) -> TrajError {
    TrajError::Unsupported(format!("cuda driver error: {err}"))
}

fn map_compile_err(err: cudarc::nvrtc::CompileError) -> TrajError {
    TrajError::Unsupported(format!("cuda compile error: {err}"))
}
