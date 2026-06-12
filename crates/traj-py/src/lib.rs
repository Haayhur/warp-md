use std::cell::RefCell;
use std::sync::Arc;

use numpy::ndarray::{Array2, Array3, Array4};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use warp_pack::PackConfig as PackConfigInput;
use warp_structure::{AtomRecord, AtomRecordKind, OutputSpec, PackOutput, Vec3};

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunkBuilder};
use traj_core::interner::StringInterner;
use traj_core::selection::Selection;
use traj_core::system::{AtomTable, System};
use traj_engine::{
    AlignPlan, AlignPrincipalAxisPlan, AnalyzeModesPlan, AnglePlan, AtomMapPlan, AtomicAdpPlan,
    AtomicCorrelationPlan, AtomicFluctPlan, AutoImagePlan, AverageFramePlan, BfactorsPlan,
    BondAngleDistributionPlan, BondLengthDistributionPlan, BondiFfvPlan, BundlePlan, CenterMode,
    CenterOfGeometryPlan, CenterOfMassPlan, CenterTrajectoryPlan, ChainRgPlan, CheckChiralityPlan,
    CheckStructurePlan, ClosestAtomPlan, ClosestPlan, ClusterMethod, ConductivityPlan,
    ContourLengthPlan, CountInVoxelPlan, CrossCorrelationPlan, CurrentPlan, DensityMapPlan,
    DensityMapUnit, DensityPlan, DielectricPlan, DihedralPlan, DihedralRmsPlan,
    DipoleAlignmentPlan, DistancePlan, DistanceRmsdPlan, DistanceToPointPlan,
    DistanceToReferencePlan, DockingPlan, DsspPlan, DtDecimation, EndToEndPlan, EquipartitionPlan,
    Executor, FixImageBondsPlan, FrameDecimation, FreeVolumePlan, GetVelocityPlan, GistDirectPlan,
    GistGridPlan, GroupBy, HausdorffPlan, HbondPlan, HelixOrientationPlan, HelixPlan,
    HydrationOrderPlan, ImagePlan, IonPairCorrelationPlan, LagMode, LowestCurvePlan,
    MakeStructurePlan, MatrixMode, MatrixPlan, MdmatPlan, MeanStructurePlan, MindistPlan,
    MolSurfPlan, MsdPlan, MultiDihedralPlan, MultiPuckerMode, MultiPuckerPlan, NativeContactsPlan,
    NmrIredPlan, OrientationSpec, PairDistPlan, PairwiseDistancePlan, PairwiseMetric,
    PairwiseRmsdPlan, PbcMode, PcaPlan, PermuteDihedralsPlan, PersistenceLengthPlan, Plan,
    PlanOutput, PotentialPlan, PrincipalAxesPlan, ProjectionPlan, PuckerMetric, PuckerPlan,
    RadgyrTensorPlan, RamaPlan, RandomizeIonsPlan, RdfPlan, ReferenceMode, ReplicateCellPlan,
    RgPlan, RmsdPerResPlan, RmsdPlan, RmsfPlan, RotAcfPlan, RotateDihedralPlan, RotatePlan,
    RotationMatrixPlan, SaltBridgePlan, ScalePlan, SearchNeighborsPlan, SelectedFrame,
    SetDihedralPlan, SetVelocityPlan, SolventOrientationPlan, SolventPolarizationPlan, StripPlan,
    StructureFactorPlan, SuperposePlan, SurfAlgorithm, SurfPlan, SymmRmsdPlan, TimeBinning,
    ToroidalDiffusionPlan, TorsionDiffusionPlan, TrajectoryClusterPlan, TransformPlan,
    TranslatePlan, VanHovePlan, VectorPlan, VelocityAutoCorrPlan, VolmapPlan, VolumePlan,
    WaterCountPlan, WaterOrderPlan, WatershellPlan, WaveletPlan, XtalSymmPlan,
};
use traj_io::cpt::{CptReader, CptWriter};
use traj_io::dcd::{DcdReader, DcdWriter};
use traj_io::gro_traj::{GroTrajReader, GroTrajWriter};
use traj_io::gromos96_traj::{Gromos96TrajReader, Gromos96TrajWriter};
use traj_io::h5md::{H5mdReader, H5mdWriter};
use traj_io::pdb_traj::PdbTrajReader;
use traj_io::tng::{TngReader, TngWriter};
use traj_io::trr::{TrrReader, TrrWriter};
use traj_io::xtc::{XtcReader, XtcWriter};
use traj_io::TrajReader;

mod analysis;
mod contract;
mod correlation;
mod geometry;
mod io;
mod pack;
mod structure;
mod transform;

pub(crate) use self::io::{
    coords_to_vec, py_box_to_box3, PySelection, PySystem, PyTrajectory, TrajKind,
};
pub(crate) use self::pack::{
    bondi_ffv_to_py, bundle_to_py, clustering_to_py, current_to_py, density_map_to_py,
    dielectric_to_py, get_attr_or_item, get_attr_or_item_opt, grid_to_py, helix_orientation_to_py,
    helix_to_py, heuristic_chunk_frames, hist_to_py, hydration_order_to_py, json_value_to_py,
    matrix_to_py, mdmat_to_py, parse_axis, parse_density_map_unit, parse_group_by, parse_lag_mode,
    parse_matrix_mode, parse_pairwise_metric, parse_pbc, parse_reference, pca_to_py,
    potential_to_py, reset_traj, resolve_chunk_frames_for_streaming, run_plan,
    run_plan_with_frame_subset, saltbr_class_name, solvent_orientation_to_py,
    solvent_polarization_to_py, split_rama_phi_psi, structure_factor_to_py, timeseries_to_py,
    to_py_err, traj_n_frames_hint, vanhove_to_py, water_order_to_py,
};

#[pyfunction]
fn qm_cli(argv: Vec<String>) -> PyResult<i32> {
    warp_qm::cli::run_from_args(argv)
        .map(i32::from)
        .map_err(PyRuntimeError::new_err)
}

#[pymodule]
fn traj_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "__rust_build_profile__",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )?;
    m.add("__rust_cuda_enabled__", cfg!(feature = "cuda"))?;
    io::register(m)?;
    structure::register(m)?;
    geometry::register(m)?;
    transform::register(m)?;
    correlation::register(m)?;
    analysis::register(m)?;
    pack::register(m)?;
    contract::register(m)?;
    m.add_function(pyo3::wrap_pyfunction!(qm_cli, m)?)?;
    Ok(())
}
