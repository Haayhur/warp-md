use std::cell::RefCell;
use std::sync::Arc;

use numpy::ndarray::{Array2, Array3, Array4};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use warp_pack::PackConfig as PackConfigInput;

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunkBuilder};
use traj_core::selection::Selection;
use traj_core::system::System;
use traj_engine::plans::analysis::msd::{DtDecimation, FrameDecimation, TimeBinning};
use traj_engine::plans::analysis::rotacf::OrientationSpec;
use traj_engine::{
    AlignPlan, AlignPrincipalAxisPlan, AnalyzeModesPlan, AnglePlan, AtomMapPlan, AtomicCorrPlan,
    AtomicFluctPlan, AutoImagePlan, AverageFramePlan, BfactorsPlan, BondAngleDistributionPlan,
    BondLengthDistributionPlan, CenterMode, CenterOfGeometryPlan, CenterOfMassPlan,
    CenterTrajectoryPlan, ChainRgPlan, CheckChiralityPlan, CheckStructurePlan, ClosestAtomPlan,
    ClosestPlan, ClusterMethod, ConductivityPlan, ContourLengthPlan, CountInVoxelPlan, DensityPlan,
    DielectricPlan, DihedralPlan, DihedralRmsPlan, DipoleAlignmentPlan, DistancePlan,
    DistanceRmsdPlan, DistanceToPointPlan, DistanceToReferencePlan, DockingPlan, DsspPlan,
    EndToEndPlan, EquipartitionPlan, Executor, FixImageBondsPlan, GetVelocityPlan, GistDirectPlan,
    GistGridPlan, GroupBy, HausdorffPlan, HbondPlan, ImagePlan, IonPairCorrelationPlan, LagMode,
    LowestCurvePlan, MakeStructurePlan, MatrixMode, MatrixPlan, MeanStructurePlan, MindistPlan,
    MolSurfPlan, MsdPlan, MultiDihedralPlan, MultiPuckerMode, MultiPuckerPlan, NativeContactsPlan,
    NmrIredPlan, PairDistPlan, PairwiseDistancePlan, PairwiseMetric, PairwiseRmsdPlan, PbcMode,
    PcaPlan, PermuteDihedralsPlan, PersistenceLengthPlan, Plan, PlanOutput, PrincipalAxesPlan,
    ProjectionPlan, PuckerMetric, PuckerPlan, RadgyrTensorPlan, RandomizeIonsPlan, RdfPlan,
    ReferenceMode, ReplicateCellPlan, RgPlan, RmsdPerResPlan, RmsdPlan, RmsfPlan, RotAcfPlan,
    RotateDihedralPlan, RotatePlan, RotationMatrixPlan, ScalePlan, SearchNeighborsPlan,
    SetDihedralPlan, SetVelocityPlan, StripPlan, StructureFactorPlan, SuperposePlan, SurfAlgorithm,
    SurfPlan, SymmRmsdPlan, ToroidalDiffusionPlan, TorsionDiffusionPlan, TrajectoryClusterPlan,
    TransformPlan, TranslatePlan, VectorPlan, VelocityAutoCorrPlan, VolmapPlan, VolumePlan,
    WaterCountPlan, WatershellPlan, WaveletPlan, XcorrPlan, XtalSymmPlan,
};
use traj_io::dcd::DcdReader;
use traj_io::gro::GroReader;
use traj_io::pdb::{PdbReader, PdbqtReader};
use traj_io::pdb_traj::PdbTrajReader;
use traj_io::xtc::XtcReader;
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
