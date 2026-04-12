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
use traj_core::pdb_gro::{parse_gro_reader, parse_pdb_reader, PdbParseOptions, PdbRecordKind};
use traj_core::selection::Selection;
use traj_core::system::{AtomTable, System};
use traj_engine::executor::SelectedFrame;
use traj_engine::plans::analysis::msd::{DtDecimation, FrameDecimation, TimeBinning};
use traj_engine::plans::analysis::rotacf::OrientationSpec;
use traj_engine::{
    AlignPlan, AlignPrincipalAxisPlan, AnalyzeModesPlan, AnglePlan, AtomMapPlan, AtomicAdpPlan,
    AtomicCorrPlan, AtomicFluctPlan, AutoImagePlan, AverageFramePlan, BfactorsPlan,
    BondAngleDistributionPlan, BondLengthDistributionPlan, BondiFfvPlan, BundlePlan, CenterMode,
    CenterOfGeometryPlan, CenterOfMassPlan, CenterTrajectoryPlan, ChainRgPlan, CheckChiralityPlan,
    CheckStructurePlan, ClosestAtomPlan, ClosestPlan, ClusterMethod, ConductivityPlan,
    ContourLengthPlan, CountInVoxelPlan, CurrentPlan, DensityMapPlan, DensityMapUnit, DensityPlan,
    DielectricPlan, DihedralPlan, DihedralRmsPlan, DipoleAlignmentPlan, DistancePlan,
    DistanceRmsdPlan, DistanceToPointPlan, DistanceToReferencePlan, DockingPlan, DsspPlan,
    EndToEndPlan, EquipartitionPlan, Executor, FixImageBondsPlan, FreeVolumePlan, GetVelocityPlan,
    GistDirectPlan, GistGridPlan, GroupBy, H2OrderPlan, HausdorffPlan, HbondPlan, HelixOrientPlan,
    HelixPlan, HydOrderPlan, ImagePlan, IonPairCorrelationPlan, LagMode, LowestCurvePlan,
    MakeStructurePlan, MatrixMode, MatrixPlan, MdmatPlan, MeanStructurePlan, MindistPlan,
    MolSurfPlan, MsdPlan, MultiDihedralPlan, MultiPuckerMode, MultiPuckerPlan, NativeContactsPlan,
    NmrIredPlan, PairDistPlan, PairwiseDistancePlan, PairwiseMetric, PairwiseRmsdPlan, PbcMode,
    PcaPlan, PermuteDihedralsPlan, PersistenceLengthPlan, Plan, PlanOutput, PotentialPlan,
    PrincipalAxesPlan, ProjectionPlan, PuckerMetric, PuckerPlan, RadgyrTensorPlan, RamaPlan,
    RandomizeIonsPlan, RdfPlan, ReferenceMode, ReplicateCellPlan, RgPlan, RmsdPerResPlan, RmsdPlan,
    RmsfPlan, RotAcfPlan, RotateDihedralPlan, RotatePlan, RotationMatrixPlan, SOrientPlan,
    SaltBridgePlan, ScalePlan, SearchNeighborsPlan, SetDihedralPlan, SetVelocityPlan, SpolPlan,
    StripPlan, StructureFactorPlan, SuperposePlan, SurfAlgorithm, SurfPlan, SymmRmsdPlan,
    ToroidalDiffusionPlan, TorsionDiffusionPlan, TrajectoryClusterPlan, TransformPlan,
    TranslatePlan, VanHovePlan, VectorPlan, VelocityAutoCorrPlan, VolmapPlan, VolumePlan,
    WaterCountPlan, WatershellPlan, WaveletPlan, XcorrPlan, XtalSymmPlan,
};
use traj_io::cpt::{CptReader, CptWriter};
use traj_io::dcd::{DcdReader, DcdWriter};
use traj_io::gro::GroReader;
use traj_io::gro_traj::{GroTrajReader, GroTrajWriter};
use traj_io::gromos96_traj::{Gromos96TrajReader, Gromos96TrajWriter};
use traj_io::h5md::{H5mdReader, H5mdWriter};
use traj_io::pdb::{PdbReader, PdbqtReader};
use traj_io::pdb_traj::PdbTrajReader;
use traj_io::tng::{TngReader, TngWriter};
use traj_io::trr::{TrrReader, TrrWriter};
use traj_io::xtc::{XtcReader, XtcWriter};
use traj_io::{TopologyReader, TrajReader};

include!("py_part1.rs");
include!("py_part2.rs");
include!("py_part3.rs");
include!("py_part4.rs");
include!("py_part5.rs");
include!("py_part6.rs");
include!("py_part7.rs");
include!("py_part8.rs");
include!("py_part9.rs");
include!("py_agent_contract.rs");
include!("py_part10.rs");
