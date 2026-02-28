from __future__ import annotations

from typing import Any, Dict

from .cli_api import (
    BondAngleDistributionPlan,
    BondLengthDistributionPlan,
    ChainRgPlan,
    ConductivityPlan,
    ContourLengthPlan,
    DielectricPlan,
    DipoleAlignmentPlan,
    EndToEndPlan,
    EquipartitionPlan,
    HbondPlan,
    IonPairCorrelationPlan,
    MsdPlan,
    PersistenceLengthPlan,
    RdfPlan,
    RgPlan,
    RmsdPlan,
    RotAcfPlan,
    StructureFactorPlan,
    System,
    WaterCountPlan,
    _select,
)
from .cli_utils import _as_tuple, _pick, _resolve_charges, _resolve_group_types


class _CallablePlan:
    def __init__(self, fn, kwargs: Dict[str, Any]):
        self._fn = fn
        self._kwargs = dict(kwargs)

    def run(self, traj, system, chunk_frames=None, device="auto"):
        kwargs = dict(self._kwargs)
        kwargs["chunk_frames"] = kwargs.get("chunk_frames", chunk_frames)
        kwargs["device"] = kwargs.get("device", device)
        return self._fn(traj, system, **kwargs)


def _build_rg(system: System, spec: Dict[str, Any]):
    sel = _select(system, spec.get("selection"), "rg.selection")
    kwargs = _pick(spec, ["mass_weighted"])
    return RgPlan(sel, **kwargs)


def _build_rmsd(system: System, spec: Dict[str, Any]):
    sel = _select(system, spec.get("selection"), "rmsd.selection")
    kwargs = _pick(spec, ["reference", "align"])
    return RmsdPlan(sel, **kwargs)


def _build_msd(system: System, spec: Dict[str, Any]):
    sel = _select(system, spec.get("selection"), "msd.selection")
    group_by = spec.get("group_by", "resid")
    group_types = _resolve_group_types(system, sel, group_by, spec.get("group_types"))
    kwargs = _pick(
        spec,
        [
            "axis",
            "length_scale",
            "frame_decimation",
            "dt_decimation",
            "time_binning",
            "lag_mode",
            "max_lag",
            "memory_budget_bytes",
            "multi_tau_m",
            "multi_tau_levels",
        ],
    )
    if "axis" in kwargs:
        kwargs["axis"] = _as_tuple(kwargs["axis"], 3, "axis")
    if "frame_decimation" in kwargs:
        kwargs["frame_decimation"] = _as_tuple(kwargs["frame_decimation"], 2, "frame_decimation")
    if "dt_decimation" in kwargs:
        kwargs["dt_decimation"] = _as_tuple(kwargs["dt_decimation"], 4, "dt_decimation")
    if "time_binning" in kwargs:
        kwargs["time_binning"] = _as_tuple(kwargs["time_binning"], 2, "time_binning")
    if group_types is not None:
        kwargs["group_types"] = group_types
    return MsdPlan(sel, group_by=group_by, **kwargs)


def _build_rotacf(system: System, spec: Dict[str, Any]):
    sel = _select(system, spec.get("selection"), "rotacf.selection")
    group_by = spec.get("group_by", "resid")
    group_types = _resolve_group_types(system, sel, group_by, spec.get("group_types"))
    kwargs = _pick(
        spec,
        [
            "orientation",
            "p2_legendre",
            "length_scale",
            "frame_decimation",
            "dt_decimation",
            "time_binning",
            "lag_mode",
            "max_lag",
            "memory_budget_bytes",
            "multi_tau_m",
            "multi_tau_levels",
        ],
    )
    if "orientation" in kwargs:
        orient = kwargs["orientation"]
        if not isinstance(orient, (list, tuple)) or len(orient) not in (2, 3):
            raise ValueError("rotacf.orientation must be length 2 or 3")
    if "frame_decimation" in kwargs:
        kwargs["frame_decimation"] = _as_tuple(kwargs["frame_decimation"], 2, "frame_decimation")
    if "dt_decimation" in kwargs:
        kwargs["dt_decimation"] = _as_tuple(kwargs["dt_decimation"], 4, "dt_decimation")
    if "time_binning" in kwargs:
        kwargs["time_binning"] = _as_tuple(kwargs["time_binning"], 2, "time_binning")
    if group_types is not None:
        kwargs["group_types"] = group_types
    return RotAcfPlan(sel, group_by=group_by, **kwargs)


def _build_conductivity(system: System, spec: Dict[str, Any]):
    sel = _select(system, spec.get("selection"), "conductivity.selection")
    group_by = spec.get("group_by", "resid")
    charges_spec = spec.get("charges")
    if charges_spec is None:
        raise ValueError("conductivity.charges is required")
    charges = _resolve_charges(system, charges_spec)
    temperature = spec.get("temperature")
    if temperature is None:
        raise ValueError("conductivity.temperature is required")
    group_types = _resolve_group_types(system, sel, group_by, spec.get("group_types"))
    kwargs = _pick(
        spec,
        [
            "transference",
            "length_scale",
            "frame_decimation",
            "dt_decimation",
            "time_binning",
            "lag_mode",
            "max_lag",
            "memory_budget_bytes",
            "multi_tau_m",
            "multi_tau_levels",
        ],
    )
    if "frame_decimation" in kwargs:
        kwargs["frame_decimation"] = _as_tuple(kwargs["frame_decimation"], 2, "frame_decimation")
    if "dt_decimation" in kwargs:
        kwargs["dt_decimation"] = _as_tuple(kwargs["dt_decimation"], 4, "dt_decimation")
    if "time_binning" in kwargs:
        kwargs["time_binning"] = _as_tuple(kwargs["time_binning"], 2, "time_binning")
    if group_types is not None:
        kwargs["group_types"] = group_types
    return ConductivityPlan(sel, charges, temperature, group_by=group_by, **kwargs)


def _build_dielectric(system: System, spec: Dict[str, Any]):
    sel = _select(system, spec.get("selection"), "dielectric.selection")
    group_by = spec.get("group_by", "resid")
    charges_spec = spec.get("charges")
    if charges_spec is None:
        raise ValueError("dielectric.charges is required")
    charges = _resolve_charges(system, charges_spec)
    group_types = _resolve_group_types(system, sel, group_by, spec.get("group_types"))
    kwargs = _pick(spec, ["length_scale"])
    if group_types is not None:
        kwargs["group_types"] = group_types
    return DielectricPlan(sel, charges, group_by=group_by, **kwargs)


def _build_dipole_alignment(system: System, spec: Dict[str, Any]):
    sel = _select(system, spec.get("selection"), "dipole_alignment.selection")
    group_by = spec.get("group_by", "resid")
    charges_spec = spec.get("charges")
    if charges_spec is None:
        raise ValueError("dipole_alignment.charges is required")
    charges = _resolve_charges(system, charges_spec)
    group_types = _resolve_group_types(system, sel, group_by, spec.get("group_types"))
    kwargs = _pick(spec, ["length_scale"])
    if group_types is not None:
        kwargs["group_types"] = group_types
    return DipoleAlignmentPlan(sel, charges, group_by=group_by, **kwargs)


def _build_ion_pair(system: System, spec: Dict[str, Any]):
    sel = _select(system, spec.get("selection"), "ion_pair_correlation.selection")
    group_by = spec.get("group_by", "resid")
    rclust_cat = spec.get("rclust_cat")
    rclust_ani = spec.get("rclust_ani")
    if rclust_cat is None or rclust_ani is None:
        raise ValueError("ion_pair_correlation.rclust_cat and rclust_ani are required")
    group_types = _resolve_group_types(system, sel, group_by, spec.get("group_types"))
    kwargs = _pick(
        spec,
        [
            "cation_type",
            "anion_type",
            "max_cluster",
            "length_scale",
            "lag_mode",
            "max_lag",
            "memory_budget_bytes",
            "multi_tau_m",
            "multi_tau_levels",
        ],
    )
    if group_types is not None:
        kwargs["group_types"] = group_types
    return IonPairCorrelationPlan(sel, rclust_cat, rclust_ani, group_by=group_by, **kwargs)


def _build_structure_factor(system: System, spec: Dict[str, Any]):
    sel = _select(system, spec.get("selection"), "structure_factor.selection")
    bins = spec.get("bins")
    r_max = spec.get("r_max")
    q_bins = spec.get("q_bins")
    q_max = spec.get("q_max")
    if None in (bins, r_max, q_bins, q_max):
        raise ValueError("structure_factor requires bins, r_max, q_bins, q_max")
    kwargs = _pick(spec, ["pbc", "length_scale"])
    return StructureFactorPlan(sel, bins, r_max, q_bins, q_max, **kwargs)


def _build_water_count(system: System, spec: Dict[str, Any]):
    water_sel = _select(system, spec.get("water_selection"), "water_count.water_selection")
    center_sel = _select(system, spec.get("center_selection"), "water_count.center_selection")
    box_unit = spec.get("box_unit")
    region_size = spec.get("region_size")
    if box_unit is None or region_size is None:
        raise ValueError("water_count requires box_unit and region_size")
    kwargs = _pick(spec, ["shift", "length_scale"])
    kwargs["box_unit"] = _as_tuple(box_unit, 3, "box_unit")
    kwargs["region_size"] = _as_tuple(region_size, 3, "region_size")
    if "shift" in kwargs:
        kwargs["shift"] = _as_tuple(kwargs["shift"], 3, "shift")
    return WaterCountPlan(water_sel, center_sel, **kwargs)


def _build_equipartition(system: System, spec: Dict[str, Any]):
    sel = _select(system, spec.get("selection"), "equipartition.selection")
    group_by = spec.get("group_by", "resid")
    group_types = _resolve_group_types(system, sel, group_by, spec.get("group_types"))
    kwargs = _pick(spec, ["velocity_scale", "length_scale"])
    if group_types is not None:
        kwargs["group_types"] = group_types
    return EquipartitionPlan(sel, group_by=group_by, **kwargs)


def _build_hbond(system: System, spec: Dict[str, Any]):
    donors = _select(system, spec.get("donors"), "hbond.donors")
    acceptors = _select(system, spec.get("acceptors"), "hbond.acceptors")
    dist_cutoff = spec.get("dist_cutoff")
    if dist_cutoff is None:
        raise ValueError("hbond.dist_cutoff is required")
    hydrogens_expr = spec.get("hydrogens")
    angle_cutoff = spec.get("angle_cutoff")
    if hydrogens_expr:
        hydrogens = _select(system, hydrogens_expr, "hbond.hydrogens")
        return HbondPlan(donors, acceptors, dist_cutoff, hydrogens=hydrogens, angle_cutoff=angle_cutoff)
    return HbondPlan(donors, acceptors, dist_cutoff)


def _build_rdf(system: System, spec: Dict[str, Any]):
    sel_a = _select(system, spec.get("sel_a"), "rdf.sel_a")
    sel_b = _select(system, spec.get("sel_b"), "rdf.sel_b")
    bins = spec.get("bins")
    r_max = spec.get("r_max")
    if bins is None or r_max is None:
        raise ValueError("rdf requires bins and r_max")
    kwargs = _pick(spec, ["pbc"])
    return RdfPlan(sel_a, sel_b, bins, r_max, **kwargs)


def _build_end_to_end(system: System, spec: Dict[str, Any]):
    sel = _select(system, spec.get("selection"), "end_to_end.selection")
    return EndToEndPlan(sel)


def _build_contour_length(system: System, spec: Dict[str, Any]):
    sel = _select(system, spec.get("selection"), "contour_length.selection")
    return ContourLengthPlan(sel)


def _build_chain_rg(system: System, spec: Dict[str, Any]):
    sel = _select(system, spec.get("selection"), "chain_rg.selection")
    return ChainRgPlan(sel)


def _build_bond_length(system: System, spec: Dict[str, Any]):
    sel = _select(system, spec.get("selection"), "bond_length_distribution.selection")
    bins = spec.get("bins")
    r_max = spec.get("r_max")
    if bins is None or r_max is None:
        raise ValueError("bond_length_distribution requires bins and r_max")
    return BondLengthDistributionPlan(sel, bins, r_max)


def _build_bond_angle(system: System, spec: Dict[str, Any]):
    sel = _select(system, spec.get("selection"), "bond_angle_distribution.selection")
    bins = spec.get("bins")
    if bins is None:
        raise ValueError("bond_angle_distribution requires bins")
    kwargs = _pick(spec, ["degrees"])
    return BondAngleDistributionPlan(sel, bins, **kwargs)


def _build_persistence(system: System, spec: Dict[str, Any]):
    sel = _select(system, spec.get("selection"), "persistence_length.selection")
    return PersistenceLengthPlan(sel)


def _build_docking(system: System, spec: Dict[str, Any]):
    receptor_mask = spec.get("receptor_mask")
    ligand_mask = spec.get("ligand_mask")
    if receptor_mask is None:
        raise ValueError("docking.receptor_mask is required")
    if ligand_mask is None:
        raise ValueError("docking.ligand_mask is required")
    kwargs = _pick(
        spec,
        [
            "close_contact_cutoff",
            "hydrophobic_cutoff",
            "hydrogen_bond_cutoff",
            "clash_cutoff",
            "salt_bridge_cutoff",
            "halogen_bond_cutoff",
            "metal_coordination_cutoff",
            "cation_pi_cutoff",
            "pi_pi_cutoff",
            "hbond_min_angle_deg",
            "donor_hydrogen_cutoff",
            "allow_missing_hydrogen",
            "length_scale",
            "frame_indices",
            "max_events_per_frame",
            "chunk_frames",
            "device",
        ],
    )
    kwargs["receptor_mask"] = receptor_mask
    kwargs["ligand_mask"] = ligand_mask
    from .analysis.docking import docking as docking_analysis

    return _CallablePlan(docking_analysis, kwargs)


def _build_dssp(system: System, spec: Dict[str, Any]):
    from .analysis.dssp import dssp as dssp_analysis
    kwargs = _pick(spec, ["mask", "simplified", "chunk_frames"])
    return _CallablePlan(dssp_analysis, kwargs)


def _build_diffusion(system: System, spec: Dict[str, Any]):
    from .analysis.diffusion import diffusion as diffusion_analysis
    kwargs = _pick(spec, ["mask", "tstep", "individual", "frame_indices", "chunk_frames"])
    return _CallablePlan(diffusion_analysis, kwargs)


def _build_pca(system: System, spec: Dict[str, Any]):
    from .analysis.pca import pca as pca_analysis
    mask = spec.get("mask", "")
    if not mask:
        raise ValueError("pca.mask is required")
    kwargs = _pick(spec, ["mask", "n_vecs", "fit", "ref", "ref_mask", "chunk_frames"])
    return _CallablePlan(pca_analysis, kwargs)


def _build_rmsf(system: System, spec: Dict[str, Any]):
    from .analysis.fluct import rmsf as rmsf_analysis
    kwargs = _pick(spec, ["mask", "byres", "bymask", "calcadp", "frame_indices", "chunk_frames", "length_scale"])
    return _CallablePlan(rmsf_analysis, kwargs)


def _build_density(system: System, spec: Dict[str, Any]):
    from .analysis.density import density as density_analysis
    kwargs = _pick(spec, ["mask", "density_type", "delta", "direction", "cutoff", "center", "mass", "restrict", "frame_indices", "chunk_frames"])
    return _CallablePlan(density_analysis, kwargs)


def _build_native_contacts(system: System, spec: Dict[str, Any]):
    from .analysis.native_contacts import native_contacts as native_contacts_analysis
    kwargs = _pick(spec, ["mask", "mask2", "ref", "distance", "mindist", "maxdist", "image", "frame_indices", "chunk_frames"])
    return _CallablePlan(native_contacts_analysis, kwargs)


def _build_volmap(system: System, spec: Dict[str, Any]):
    from .analysis.volmap import volmap as volmap_analysis
    kwargs = _pick(spec, ["mask", "grid_spacing", "size", "center", "buffer", "centermask", "radscale", "peakcut", "dtype", "frame_indices", "chunk_frames"])
    return _CallablePlan(volmap_analysis, kwargs)


def _build_surf(system: System, spec: Dict[str, Any]):
    from .analysis.surf import surf as surf_analysis
    kwargs = _pick(spec, ["mask", "algorithm", "probe_radius", "n_sphere_points", "radii", "frame_indices", "chunk_frames", "device"])
    return _CallablePlan(surf_analysis, kwargs)


def _build_molsurf(system: System, spec: Dict[str, Any]):
    from .analysis.surf import molsurf as molsurf_analysis
    kwargs = _pick(spec, ["mask", "algorithm", "probe_radius", "n_sphere_points", "radii", "frame_indices", "chunk_frames", "device"])
    return _CallablePlan(molsurf_analysis, kwargs)


def _build_watershell(system: System, spec: Dict[str, Any]):
    from .analysis.watershell import watershell as watershell_analysis
    solute_mask = spec.get("solute_mask", "")
    if not solute_mask:
        raise ValueError("watershell.solute_mask is required")
    kwargs = _pick(spec, ["solute_mask", "solvent_mask", "lower", "upper", "image", "frame_indices", "chunk_frames"])
    return _CallablePlan(watershell_analysis, kwargs)


def _build_tordiff(system: System, spec: Dict[str, Any]):
    from .analysis.diffusion import tordiff as tordiff_analysis
    mask = spec.get("mask", "")
    if not mask:
        raise ValueError("tordiff.mask is required")
    kwargs = _pick(spec, ["mask", "tstep", "chunk_frames"])
    return _CallablePlan(tordiff_analysis, kwargs)


def _build_projection(system: System, spec: Dict[str, Any]):
    from .analysis.pca import projection as projection_analysis
    mask = spec.get("mask", "")
    if not mask:
        raise ValueError("projection.mask is required")
    kwargs = _pick(spec, ["mask", "eigenvec", "n_vecs", "fit", "ref", "ref_mask", "chunk_frames"])
    return _CallablePlan(projection_analysis, kwargs)


def _build_nmr(system: System, spec: Dict[str, Any]):
    from .analysis.nmr import nh_order_parameters as nmr_analysis
    mask = spec.get("selection", "")
    if not mask:
        raise ValueError("nmr.selection is required")
    # map schema 'selection' to 'selection' arg, but allow 'vector_pairs' if provided
    kwargs = _pick(spec, ["selection", "vector_pairs", "method", "order", "tstep", "tcorr", "length_scale", "pbc", "chunk_frames", "frame_indices"])
    return _CallablePlan(nmr_analysis, kwargs)


def _build_jcoupling(system: System, spec: Dict[str, Any]):
    from .analysis.nmr import jcoupling as jcoupling_analysis
    dihedrals = spec.get("dihedrals")
    if not dihedrals:
        raise ValueError("jcoupling.dihedrals is required")
    # map schema 'dihedrals' to 'dihedral_indices'
    kwargs = _pick(spec, ["dihedrals", "karplus", "kfile", "phase_deg", "length_scale", "pbc", "chunk_frames", "frame_indices", "device"])
    if "dihedrals" in kwargs:
        kwargs["dihedral_indices"] = kwargs.pop("dihedrals")
    return _CallablePlan(jcoupling_analysis, kwargs)


def _build_gist(system: System, spec: Dict[str, Any]):
    # Placeholder for gist until OpenMM object construction is supported in runner
    raise NotImplementedError(
        "GIST analysis requires OpenMM system construction which is not yet supported in the agent runner."
    )


PLAN_BUILDERS = {
    "rg": _build_rg,
    "rmsd": _build_rmsd,
    "msd": _build_msd,
    "rotacf": _build_rotacf,
    "conductivity": _build_conductivity,
    "dielectric": _build_dielectric,
    "dipole_alignment": _build_dipole_alignment,
    "ion_pair_correlation": _build_ion_pair,
    "structure_factor": _build_structure_factor,
    "water_count": _build_water_count,
    "equipartition": _build_equipartition,
    "hbond": _build_hbond,
    "rdf": _build_rdf,
    "end_to_end": _build_end_to_end,
    "contour_length": _build_contour_length,
    "chain_rg": _build_chain_rg,
    "bond_length_distribution": _build_bond_length,
    "bond_angle_distribution": _build_bond_angle,
    "persistence_length": _build_persistence,
    "docking": _build_docking,
    # New analyses
    "dssp": _build_dssp,
    "diffusion": _build_diffusion,
    "pca": _build_pca,
    "rmsf": _build_rmsf,
    "density": _build_density,
    "native_contacts": _build_native_contacts,
    # Additional analyses
    "volmap": _build_volmap,
    "surf": _build_surf,
    "molsurf": _build_molsurf,
    "watershell": _build_watershell,
    "tordiff": _build_tordiff,
    "projection": _build_projection,
    # High priority analyses
    "nmr": _build_nmr,
    "jcoupling": _build_jcoupling,
    "gist": _build_gist,
}


CLI_TO_PLAN = {
    "rg": "rg",
    "rmsd": "rmsd",
    "msd": "msd",
    "rotacf": "rotacf",
    "conductivity": "conductivity",
    "dielectric": "dielectric",
    "dipole-alignment": "dipole_alignment",
    "ion-pair-correlation": "ion_pair_correlation",
    "structure-factor": "structure_factor",
    "water-count": "water_count",
    "equipartition": "equipartition",
    "hbond": "hbond",
    "rdf": "rdf",
    "end-to-end": "end_to_end",
    "contour-length": "contour_length",
    "chain-rg": "chain_rg",
    "bond-length-distribution": "bond_length_distribution",
    "bond-angle-distribution": "bond_angle_distribution",
    "persistence-length": "persistence_length",
    "dssp": "dssp",
    "diffusion": "diffusion",
    "pca": "pca",
    "rmsf": "rmsf",
    "density": "density",
    "native-contacts": "native_contacts",
    # Additional analyses
    "volmap": "volmap",
    "surf": "surf",
    "molsurf": "molsurf",
    "watershell": "watershell",
    "tordiff": "tordiff",
    "projection": "projection",
    # High priority analyses
    "nmr": "nmr",
    "jcoupling": "jcoupling",
    "gist": "gist",
}

