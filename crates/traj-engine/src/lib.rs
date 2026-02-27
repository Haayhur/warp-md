pub mod correlators;
pub mod executor;
pub mod feature_store;
pub mod plans;

pub use correlators::{LagMode, LagSettings};
pub use executor::{
    collect_selected_frames, count_frames, normalize_frame_indices, ClusteringOutput, Device,
    DielectricOutput, Executor, GridOutput, PcaOutput, PersistenceOutput, Plan, PlanOutput,
    RdfOutput, SelectedFrame, SelectedFramesReader, StructureFactorOutput,
};
pub use feature_store::{
    ChunkIndex, FeatureIndex, FeatureSchema, FeatureStoreReader, FeatureStoreWriter,
};
pub use plans::analysis::{
    AtomicCorrPlan, ClusterMethod, ConductivityPlan, CountInVoxelPlan, DensityPlan, DielectricPlan,
    DipoleAlignmentPlan, DockingPlan, DsspPlan, EquipartitionPlan, GistDirectPlan, GistGridPlan,
    GroupBy, GroupMap, GroupSpec, HbondPlan, IonPairCorrelationPlan, MolSurfPlan, MsdPlan,
    MultiPuckerMode, MultiPuckerPlan, NmrIredPlan, RotAcfPlan, StructureFactorPlan, SurfAlgorithm,
    SurfPlan, ToroidalDiffusionPlan, TorsionDiffusionPlan, TorsionStat, TrajectoryClusterPlan,
    VelocityAutoCorrPlan, VolmapPlan, WaterCountPlan, WaveletPlan, XcorrPlan,
};
pub use plans::{
    AlignPlan, AlignPrincipalAxisPlan, AnalyzeModesPlan, AnglePlan, AtomMapPlan, AtomicFluctPlan,
    AutoImagePlan, AverageFramePlan, BfactorsPlan, BondAngleDistributionPlan,
    BondLengthDistributionPlan, CenterMode, CenterOfGeometryPlan, CenterOfMassPlan,
    CenterTrajectoryPlan, ChainRgPlan, CheckChiralityPlan, CheckStructurePlan, ClosestAtomPlan,
    ClosestPlan, ContourLengthPlan, DihedralPlan, DihedralRmsPlan, DistancePlan, DistanceRmsdPlan,
    DistanceToPointPlan, DistanceToReferencePlan, EndToEndPlan, FixImageBondsPlan, GetVelocityPlan,
    HausdorffPlan, ImagePlan, LowestCurvePlan, MakeStructurePlan, MatrixMode, MatrixPlan,
    MeanStructurePlan, MindistPlan, MultiDihedralPlan, NativeContactsPlan, PairDistPlan,
    PairwiseDistancePlan, PairwiseMetric, PairwiseRmsdPlan, PbcMode, PcaPlan, PermuteDihedralsPlan,
    PersistenceLengthPlan, PrincipalAxesPlan, ProjectionPlan, PuckerMetric, PuckerPlan,
    RadgyrTensorPlan, RandomizeIonsPlan, RdfPlan, ReferenceMode, ReplicateCellPlan, RgPlan,
    RmsdPerResPlan, RmsdPlan, RmsfPlan, RotateDihedralPlan, RotatePlan, RotationMatrixPlan,
    ScalePlan, SearchNeighborsPlan, SetDihedralPlan, SetVelocityPlan, StripPlan, SuperposePlan,
    SymmRmsdPlan, TransformPlan, TranslatePlan, VectorPlan, VolumePlan, WatershellPlan,
    XtalSymmPlan,
};

#[cfg(test)]
mod tests;
