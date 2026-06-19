pub mod analysis;
pub mod geometry;
pub mod matrix;
pub mod polymer;
pub mod rdf;
pub mod rg;
pub mod rmsd;

pub use analysis::{
    ClusterMethod, DsspPlan, DtDecimation, FrameDecimation, KabschSanderPlan, LinearDensityNorm,
    LinearDensityPlan, LinearDensityWeight, MsdPlan, NematicOrderPlan, OrientationSpec, RotAcfPlan,
    TimeBinning, TrajectoryClusterPlan,
};
pub use geometry::{
    AlignPlan, AlignPrincipalAxisPlan, AnglePlan, AtomMapPlan, AtomicAdpPlan, AtomicFluctPlan,
    AutoImagePlan, AutoImageTrajectoryPlan, AverageFramePlan, BfactorsPlan, CenterMode,
    CenterOfGeometryPlan, CenterOfMassPlan, CenterTrajectoryOutputPlan, CenterTrajectoryPlan,
    CheckChiralityPlan, CheckStructurePlan, ClosestAtomPlan, ClosestCoordsPlan, ClosestPlan,
    DihedralPlan, DihedralRmsPlan, DistanceCenterToPointPlan, DistanceCenterToReferencePlan,
    DistancePlan, DistanceToPointPlan, DistanceToReferencePlan, DistanceVectorPlan, DridPlan,
    FixImageBondsPlan, FixImageBondsTrajectoryPlan, GetVelocityPlan, HausdorffPlan, ImagePlan,
    ImageTrajectoryPlan, LowestCurvePlan, MakeStructurePlan, MeanStructurePlan, MindistPlan,
    MultiAnglePlan, MultiDihedralPlan, MultiDistancePlan, MultiVectorCommand, MultiVectorPlan,
    NativeContactsPlan, PairListDistancePlan, PairwiseDistancePlan, PermuteDihedralsPlan,
    PrincipalAxesPlan, PuckerMetric, PuckerPlan, RandomizeIonsPlan, RandomizeIonsTrajectoryPlan,
    ReplicateCellPlan, RmsdPerResPlan, RmsfPlan, RotateDihedralPlan, RotateDihedralTrajectoryPlan,
    RotatePlan, RotationMatrixPlan, RunningAveragePlan, RunningAverageTrajectoryPlan, ScalePlan,
    SearchNeighborListPlan, SearchNeighborsPlan, SetDihedralPlan, SetDihedralTrajectoryPlan,
    SetVelocityPlan, StripPlan, StripTrajectoryPlan, SuperposePlan, SuperposeTrajectoryPlan,
    TransformPlan, TransformTrajectoryPlan, TranslatePlan, VectorPlan, VolumePlan, WatershellPlan,
    XtalSymmPlan,
};
pub use matrix::{AnalyzeModesPlan, MatrixMode, MatrixPlan, PcaPlan, ProjectionPlan};
pub use polymer::{
    BondAngleDistributionPlan, BondLengthDistributionPlan, ChainRgPlan, ContourLengthPlan,
    EndToEndPlan, PersistenceLengthPlan,
};
pub use rdf::{
    PairDistDynamicPlan, PairDistPlan, PairDistanceExtremaMode, PairDistanceExtremaPlan,
    RdfDimension, RdfPlan,
};
pub use rg::{RadgyrPlan, RadgyrTensorPlan, RgPlan, ShapeDescriptorsPlan};
pub use rmsd::{DistanceRmsdPlan, PairwiseMetric, PairwiseRmsdPlan, RmsdPlan, SymmRmsdPlan};

#[derive(Debug, Clone, Copy)]
pub enum ReferenceMode {
    Topology,
    Frame0,
}

#[derive(Debug, Clone, Copy)]
pub enum PbcMode {
    None,
    Orthorhombic,
}
