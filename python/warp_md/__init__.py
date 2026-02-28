from pkgutil import extend_path

# Allow local source package + installed extension module to coexist.
__path__ = extend_path(__path__, __name__)

try:
    from . import traj_py as traj_py
    from .traj_py import (
        PySystem as System,
        PySelection as Selection,
        PyTrajectory as Trajectory,
        PyRgPlan as RgPlan,
        PyRadgyrTensorPlan as RadgyrTensorPlan,
        PyRmsdPlan as RmsdPlan,
        PySymmRmsdPlan as SymmRmsdPlan,
        PyDistanceRmsdPlan as DistanceRmsdPlan,
        PyPairwiseRmsdPlan as PairwiseRmsdPlan,
        PyTrajectoryClusterPlan as TrajectoryClusterPlan,
        PyMatrixPlan as MatrixPlan,
        PyPcaPlan as PcaPlan,
        PyAnalyzeModesPlan as AnalyzeModesPlan,
        PyProjectionPlan as ProjectionPlan,
        PyRmsdPerResPlan as RmsdPerResPlan,
        PyRmsfPlan as RmsfPlan,
        PyBfactorsPlan as BfactorsPlan,
        PyAtomicFluctPlan as AtomicFluctPlan,
        PyDistancePlan as DistancePlan,
        PyLowestCurvePlan as LowestCurvePlan,
        PyVectorPlan as VectorPlan,
        PyGetVelocityPlan as GetVelocityPlan,
        PySetVelocityPlan as SetVelocityPlan,
        PyPairwiseDistancePlan as PairwiseDistancePlan,
        PyPairDistPlan as PairDistPlan,
        PyAnglePlan as AnglePlan,
        PyDihedralPlan as DihedralPlan,
        PyMultiDihedralPlan as MultiDihedralPlan,
        PyPermuteDihedralsPlan as PermuteDihedralsPlan,
        PyDihedralRmsPlan as DihedralRmsPlan,
        PyPuckerPlan as PuckerPlan,
        PyRotateDihedralPlan as RotateDihedralPlan,
        PySetDihedralPlan as SetDihedralPlan,
        PyCheckChiralityPlan as CheckChiralityPlan,
        PyMindistPlan as MindistPlan,
        PyHausdorffPlan as HausdorffPlan,
        PyCheckStructurePlan as CheckStructurePlan,
        PyAtomMapPlan as AtomMapPlan,
        PyFixImageBondsPlan as FixImageBondsPlan,
        PyRandomizeIonsPlan as RandomizeIonsPlan,
        PyClosestAtomPlan as ClosestAtomPlan,
        PySearchNeighborsPlan as SearchNeighborsPlan,
        PyWatershellPlan as WatershellPlan,
        PyClosestPlan as ClosestPlan,
        PyNativeContactsPlan as NativeContactsPlan,
        PyCenterTrajectoryPlan as CenterTrajectoryPlan,
        PyTranslatePlan as TranslatePlan,
        PyTransformPlan as TransformPlan,
        PyRotatePlan as RotatePlan,
        PyScalePlan as ScalePlan,
        PyImagePlan as ImagePlan,
        PyAutoImagePlan as AutoImagePlan,
        PyReplicateCellPlan as ReplicateCellPlan,
        PyXtalSymmPlan as XtalSymmPlan,
        PyVolumePlan as VolumePlan,
        PyStripPlan as StripPlan,
        PyMeanStructurePlan as MeanStructurePlan,
        PyAverageFramePlan as AverageFramePlan,
        PyMakeStructurePlan as MakeStructurePlan,
        PyCenterOfMassPlan as CenterOfMassPlan,
        PyCenterOfGeometryPlan as CenterOfGeometryPlan,
        PyDistanceToPointPlan as DistanceToPointPlan,
        PyDistanceToReferencePlan as DistanceToReferencePlan,
        PyPrincipalAxesPlan as PrincipalAxesPlan,
        PyAlignPlan as AlignPlan,
        PySuperposePlan as SuperposePlan,
        PyRotationMatrixPlan as RotationMatrixPlan,
        PyAlignPrincipalAxisPlan as AlignPrincipalAxisPlan,
        PyMsdPlan as MsdPlan,
        PyAtomicCorrPlan as AtomicCorrPlan,
        PyVelocityAutoCorrPlan as VelocityAutoCorrPlan,
        PyRotAcfPlan as RotAcfPlan,
        PyConductivityPlan as ConductivityPlan,
        PyDielectricPlan as DielectricPlan,
        PyDipoleAlignmentPlan as DipoleAlignmentPlan,
        PyIonPairCorrelationPlan as IonPairCorrelationPlan,
        PyStructureFactorPlan as StructureFactorPlan,
        PyWaterCountPlan as WaterCountPlan,
        PyCountInVoxelPlan as CountInVoxelPlan,
        PyDensityPlan as DensityPlan,
        PyVolmapPlan as VolmapPlan,
        PyEquipartitionPlan as EquipartitionPlan,
        PyXcorrPlan as XcorrPlan,
        PyWaveletPlan as WaveletPlan,
        PySurfPlan as SurfPlan,
        PyMolSurfPlan as MolSurfPlan,
        PyTorsionDiffusionPlan as TorsionDiffusionPlan,
        PyToroidalDiffusionPlan as ToroidalDiffusionPlan,
        PyMultiPuckerPlan as MultiPuckerPlan,
        PyNmrIredPlan as NmrIredPlan,
        PyHbondPlan as HbondPlan,
        PyRdfPlan as RdfPlan,
        PyEndToEndPlan as EndToEndPlan,
        PyContourLengthPlan as ContourLengthPlan,
        PyChainRgPlan as ChainRgPlan,
        PyBondLengthDistributionPlan as BondLengthDistributionPlan,
        PyBondAngleDistributionPlan as BondAngleDistributionPlan,
        PyPersistenceLengthPlan as PersistenceLengthPlan,
    )
    try:
        from .traj_py import PyDockingPlan as DockingPlan
    except Exception:  # pragma: no cover - older extension builds
        class _MissingDockingPlan:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "DockingPlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
                )

        DockingPlan = _MissingDockingPlan
    try:
        from .traj_py import PyGistGridPlan as GistGridPlan
    except Exception:  # pragma: no cover - older extension builds
        class _MissingGistGridPlan:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "GistGridPlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
                )

        GistGridPlan = _MissingGistGridPlan
    try:
        from .traj_py import PyGistDirectPlan as GistDirectPlan
    except Exception:  # pragma: no cover - older extension builds
        class _MissingGistDirectPlan:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "GistDirectPlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
                )

        GistDirectPlan = _MissingGistDirectPlan
    try:
        from .traj_py import gist_apply_pme_scaling as gist_apply_pme_scaling
    except Exception:  # pragma: no cover - older extension builds
        def gist_apply_pme_scaling(*args, **kwargs):
            raise RuntimeError(
                "gist_apply_pme_scaling binding unavailable in this build. Rebuild bindings with `maturin develop`."
            )
    __rust_build_profile__ = getattr(traj_py, "__rust_build_profile__", "unknown")
    __rust_cuda_enabled__ = bool(getattr(traj_py, "__rust_cuda_enabled__", False))
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - allow CLI help without bindings
    _IMPORT_ERROR = exc
    traj_py = None
    __rust_build_profile__ = "unavailable"
    __rust_cuda_enabled__ = False

    def _missing(*_args, **_kwargs):
        raise RuntimeError(
            "warp-md Python bindings are unavailable. Run `maturin develop` or install warp-md."
        ) from _IMPORT_ERROR

    class _Missing:
        def __init__(self, *args, **kwargs):
            _missing(*args, **kwargs)

    System = _Missing
    Selection = _Missing
    Trajectory = _Missing
    RgPlan = _Missing
    RadgyrTensorPlan = _Missing
    RmsdPlan = _Missing
    SymmRmsdPlan = _Missing
    DistanceRmsdPlan = _Missing
    PairwiseRmsdPlan = _Missing
    TrajectoryClusterPlan = _Missing
    MatrixPlan = _Missing
    PcaPlan = _Missing
    AnalyzeModesPlan = _Missing
    ProjectionPlan = _Missing
    RmsdPerResPlan = _Missing
    RmsfPlan = _Missing
    BfactorsPlan = _Missing
    AtomicFluctPlan = _Missing
    DistancePlan = _Missing
    LowestCurvePlan = _Missing
    VectorPlan = _Missing
    GetVelocityPlan = _Missing
    SetVelocityPlan = _Missing
    PairwiseDistancePlan = _Missing
    PairDistPlan = _Missing
    AnglePlan = _Missing
    DihedralPlan = _Missing
    MultiDihedralPlan = _Missing
    PermuteDihedralsPlan = _Missing
    DihedralRmsPlan = _Missing
    PuckerPlan = _Missing
    RotateDihedralPlan = _Missing
    SetDihedralPlan = _Missing
    CheckChiralityPlan = _Missing
    MindistPlan = _Missing
    HausdorffPlan = _Missing
    CheckStructurePlan = _Missing
    AtomMapPlan = _Missing
    FixImageBondsPlan = _Missing
    RandomizeIonsPlan = _Missing
    ClosestAtomPlan = _Missing
    SearchNeighborsPlan = _Missing
    WatershellPlan = _Missing
    ClosestPlan = _Missing
    NativeContactsPlan = _Missing
    CenterTrajectoryPlan = _Missing
    TranslatePlan = _Missing
    TransformPlan = _Missing
    RotatePlan = _Missing
    ScalePlan = _Missing
    ImagePlan = _Missing
    AutoImagePlan = _Missing
    ReplicateCellPlan = _Missing
    XtalSymmPlan = _Missing
    VolumePlan = _Missing
    StripPlan = _Missing
    MeanStructurePlan = _Missing
    AverageFramePlan = _Missing
    MakeStructurePlan = _Missing
    CenterOfMassPlan = _Missing
    CenterOfGeometryPlan = _Missing
    DistanceToPointPlan = _Missing
    DistanceToReferencePlan = _Missing
    PrincipalAxesPlan = _Missing
    AlignPlan = _Missing
    SuperposePlan = _Missing
    RotationMatrixPlan = _Missing
    AlignPrincipalAxisPlan = _Missing
    MsdPlan = _Missing
    AtomicCorrPlan = _Missing
    VelocityAutoCorrPlan = _Missing
    RotAcfPlan = _Missing
    ConductivityPlan = _Missing
    DielectricPlan = _Missing
    DipoleAlignmentPlan = _Missing
    IonPairCorrelationPlan = _Missing
    StructureFactorPlan = _Missing
    DockingPlan = _Missing
    GistGridPlan = _Missing
    GistDirectPlan = _Missing
    WaterCountPlan = _Missing
    CountInVoxelPlan = _Missing
    DensityPlan = _Missing
    VolmapPlan = _Missing
    EquipartitionPlan = _Missing
    XcorrPlan = _Missing
    WaveletPlan = _Missing
    SurfPlan = _Missing
    MolSurfPlan = _Missing
    TorsionDiffusionPlan = _Missing
    ToroidalDiffusionPlan = _Missing
    MultiPuckerPlan = _Missing
    NmrIredPlan = _Missing
    HbondPlan = _Missing
    RdfPlan = _Missing
    EndToEndPlan = _Missing
    ContourLengthPlan = _Missing
    ChainRgPlan = _Missing
    BondLengthDistributionPlan = _Missing
    BondAngleDistributionPlan = _Missing
    PersistenceLengthPlan = _Missing
    gist_apply_pme_scaling = _missing
from .builder import (
    charges_from_selections,
    charges_from_table,
    group_indices,
    group_types_from_selections,
)
from .analysis import (
    acorr,
    align,
    align_principal_axis,
    superpose,
    ArrayTrajectory,
    crank,
    dssp,
    dssp_allatoms,
    dssp_allresidues,
    energy_analysis,
    ene_decomp,
    esander,
    gist,
    GistConfig,
    GistResult,
    infraredspec,
    lipidscd,
    rmsf,
    atomicfluct,
    bfactors,
    lie,
    distance_rmsd,
    pairwise_rmsd,
    cluster_trajectory,
    center,
    rotate,
    scale,
    transform,
    translate,
    covar,
    mwcovar,
    dist,
    correl,
    wavelet,
    mean_structure,
    make_structure,
    get_average_frame,
    strip,
    radgyr_tensor,
    get_velocity,
    ired_vector_and_matrix,
    jcoupling,
    nh_order_parameters,
    calc_ired_vector_and_matrix,
    calc_nh_order_parameters,
    ti,
    timecorr,
    velocity_autocorrelation,
    xcorr,
    pca,
    projection,
    analyze_modes,
    vector,
    vector_mask,
    rotation_matrix,
    angle,
    dihedral,
    search_neighbors,
    symmrmsd,
    set_velocity,
    autoimage,
    fiximagedbonds,
    native_contacts,
    atom_map,
    check_structure,
    closest,
    closest_atom,
    count_in_voxel,
    surf,
    molsurf,
    docking,
    docking_ligplot_svg,
    pucker,
    rotdif,
    multipucker,
    xtalsymm,
)
from .pack import Box as PackBox
from .pack import Constraint as PackConstraint
from .pack import OutputSpec as PackOutputSpec
from .pack import PackConfig, PackResult, Structure as PackStructure
from .pack import export as pack_export
from .pack import parse_inp as pack_parse_inp
from .pack import run as pack_run
from .pack import run_inp as pack_run_inp
from .io import open_trajectory_auto, open_trajectory

__all__ = [
    "traj_py",
    "System",
    "Selection",
    "Trajectory",
    "open_trajectory_auto",
    "open_trajectory",
    "ArrayTrajectory",
    "RgPlan",
    "RadgyrTensorPlan",
    "RmsdPlan",
    "SymmRmsdPlan",
    "DistanceRmsdPlan",
    "PairwiseRmsdPlan",
    "TrajectoryClusterPlan",
    "MatrixPlan",
    "PcaPlan",
    "AnalyzeModesPlan",
    "ProjectionPlan",
    "RmsdPerResPlan",
    "RmsfPlan",
    "BfactorsPlan",
    "AtomicFluctPlan",
    "DistancePlan",
    "LowestCurvePlan",
    "VectorPlan",
    "GetVelocityPlan",
    "SetVelocityPlan",
    "PairwiseDistancePlan",
    "PairDistPlan",
    "AnglePlan",
    "DihedralPlan",
    "MultiDihedralPlan",
    "PermuteDihedralsPlan",
    "DihedralRmsPlan",
    "PuckerPlan",
    "RotateDihedralPlan",
    "SetDihedralPlan",
    "CheckChiralityPlan",
    "MindistPlan",
    "HausdorffPlan",
    "CheckStructurePlan",
    "AtomMapPlan",
    "FixImageBondsPlan",
    "RandomizeIonsPlan",
    "ClosestAtomPlan",
    "SearchNeighborsPlan",
    "WatershellPlan",
    "ClosestPlan",
    "NativeContactsPlan",
    "CenterTrajectoryPlan",
    "TranslatePlan",
    "TransformPlan",
    "RotatePlan",
    "ScalePlan",
    "ImagePlan",
    "AutoImagePlan",
    "ReplicateCellPlan",
    "XtalSymmPlan",
    "VolumePlan",
    "StripPlan",
    "MeanStructurePlan",
    "AverageFramePlan",
    "MakeStructurePlan",
    "CenterOfMassPlan",
    "CenterOfGeometryPlan",
    "DistanceToPointPlan",
    "DistanceToReferencePlan",
    "PrincipalAxesPlan",
    "AlignPlan",
    "SuperposePlan",
    "RotationMatrixPlan",
    "AlignPrincipalAxisPlan",
    "MsdPlan",
    "AtomicCorrPlan",
    "VelocityAutoCorrPlan",
    "RotAcfPlan",
    "ConductivityPlan",
    "DielectricPlan",
    "DipoleAlignmentPlan",
    "IonPairCorrelationPlan",
    "StructureFactorPlan",
    "DockingPlan",
    "GistGridPlan",
    "GistDirectPlan",
    "WaterCountPlan",
    "CountInVoxelPlan",
    "DensityPlan",
    "VolmapPlan",
    "EquipartitionPlan",
    "XcorrPlan",
    "WaveletPlan",
    "SurfPlan",
    "MolSurfPlan",
    "TorsionDiffusionPlan",
    "ToroidalDiffusionPlan",
    "MultiPuckerPlan",
    "NmrIredPlan",
    "HbondPlan",
    "RdfPlan",
    "EndToEndPlan",
    "ContourLengthPlan",
    "ChainRgPlan",
    "BondLengthDistributionPlan",
    "BondAngleDistributionPlan",
    "PersistenceLengthPlan",
    "group_indices",
    "group_types_from_selections",
    "charges_from_selections",
    "charges_from_table",
    "crank",
    "align",
    "align_principal_axis",
    "superpose",
    "acorr",
    "xcorr",
    "timecorr",
    "velocity_autocorrelation",
    "dssp",
    "dssp_allatoms",
    "dssp_allresidues",
    "energy_analysis",
    "esander",
    "lie",
    "ene_decomp",
    "gist",
    "GistConfig",
    "GistResult",
    "infraredspec",
    "lipidscd",
    "rmsf",
    "atomicfluct",
    "bfactors",
    "distance_rmsd",
    "pairwise_rmsd",
    "cluster_trajectory",
    "center",
    "translate",
    "transform",
    "rotate",
    "scale",
    "covar",
    "mwcovar",
    "dist",
    "correl",
    "wavelet",
    "mean_structure",
    "make_structure",
    "get_average_frame",
    "strip",
    "radgyr_tensor",
    "get_velocity",
    "ired_vector_and_matrix",
    "jcoupling",
    "nh_order_parameters",
    "calc_ired_vector_and_matrix",
    "calc_nh_order_parameters",
    "ti",
    "pca",
    "projection",
    "analyze_modes",
    "vector",
    "vector_mask",
    "rotation_matrix",
    "angle",
    "dihedral",
    "search_neighbors",
    "symmrmsd",
    "set_velocity",
    "autoimage",
    "fiximagedbonds",
    "native_contacts",
    "atom_map",
    "check_structure",
    "closest",
    "closest_atom",
    "count_in_voxel",
    "surf",
    "molsurf",
    "docking",
    "docking_ligplot_svg",
    "pucker",
    "rotdif",
    "multipucker",
    "xtalsymm",
    "PackBox",
    "PackConstraint",
    "PackOutputSpec",
    "PackConfig",
    "PackResult",
    "PackStructure",
    "pack_run",
    "pack_run_inp",
    "pack_parse_inp",
    "pack_export",
    "energy_decomposition",
]

energy_decomposition = ene_decomp
