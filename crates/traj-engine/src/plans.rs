pub mod analysis;
pub mod geometry;
pub mod matrix;
pub mod polymer;
pub mod rdf;
pub mod rg;
pub mod rmsd;

pub use analysis::{
    ClusterMethod, DsspPlan, DtDecimation, FrameDecimation, MsdPlan, OrientationSpec, RotAcfPlan,
    TimeBinning, TrajectoryClusterPlan,
};
pub use geometry::{
    AlignPlan, AlignPrincipalAxisPlan, AnglePlan, AtomMapPlan, AtomicAdpPlan, AtomicFluctPlan,
    AutoImagePlan, AverageFramePlan, BfactorsPlan, CenterMode, CenterOfGeometryPlan,
    CenterOfMassPlan, CenterTrajectoryPlan, CheckChiralityPlan, CheckStructurePlan,
    ClosestAtomPlan, ClosestPlan, DihedralPlan, DihedralRmsPlan, DistanceCenterToPointPlan,
    DistanceCenterToReferencePlan, DistancePlan, DistanceToPointPlan, DistanceToReferencePlan,
    FixImageBondsPlan, GetVelocityPlan, HausdorffPlan, ImagePlan, LowestCurvePlan,
    MakeStructurePlan, MeanStructurePlan, MindistPlan, MultiDihedralPlan, MultiDistancePlan,
    NativeContactsPlan, PairListDistancePlan, PairwiseDistancePlan, PermuteDihedralsPlan,
    PrincipalAxesPlan, PuckerMetric, PuckerPlan, RandomizeIonsPlan, ReplicateCellPlan,
    RmsdPerResPlan, RmsfPlan, RotateDihedralPlan, RotatePlan, RotationMatrixPlan, ScalePlan,
    SearchNeighborsPlan, SetDihedralPlan, SetVelocityPlan, StripPlan, SuperposePlan, TransformPlan,
    TranslatePlan, VectorPlan, VolumePlan, WatershellPlan, XtalSymmPlan,
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
pub use rg::{RadgyrPlan, RadgyrTensorPlan, RgPlan};
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
