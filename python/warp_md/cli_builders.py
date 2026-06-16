from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .cli_api import (
    BondiFfvPlan,
    BondAngleDistributionPlan,
    BondLengthDistributionPlan,
    ChainRgPlan,
    ConductivityPlan,
    ContourLengthPlan,
    DielectricPlan,
    DipoleAlignmentPlan,
    EndToEndPlan,
    EquipartitionPlan,
    FreeVolumePlan,
    HydrophobicDefectPlan,
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
from .cli_analysis_registry import ANALYSIS_REGISTRY
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


def _load_array_spec(value: Any, *, field: str, default_npz_key: str = "values"):
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an array or array path")
    path = Path(value)
    if not path.exists():
        if "," in value:
            return np.asarray([int(part.strip()) for part in value.split(",") if part.strip()])
        raise ValueError(f"{field} array path does not exist: {value}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            key = default_npz_key if default_npz_key in data else next(iter(data.keys()))
            return np.asarray(data[key])
    if suffix == ".csv":
        return np.loadtxt(path, delimiter=",")
    if suffix == ".json":
        return np.asarray(json.loads(path.read_text()))
    raise ValueError(f"{field} array path must end in .npy, .npz, .csv, or .json")


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
    kwargs = _pick(spec, ["length_scale", "temperature", "make_whole"])
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


def _build_free_volume(system: System, spec: Dict[str, Any]):
    sel = _select(system, spec.get("selection"), "free_volume.selection")
    center_sel = _select(system, spec.get("center_selection"), "free_volume.center_selection")
    box_unit = spec.get("box_unit")
    region_size = spec.get("region_size")
    kwargs = _pick(spec, ["probe_radius", "shift", "length_scale"])
    if box_unit is not None:
        kwargs["box_unit"] = _as_tuple(box_unit, 3, "box_unit")
    if region_size is not None:
        kwargs["region_size"] = _as_tuple(region_size, 3, "region_size")
    if "shift" in kwargs:
        kwargs["shift"] = _as_tuple(kwargs["shift"], 3, "shift")
    return FreeVolumePlan(sel, center_sel, **kwargs)


def _build_hydrophobic_defects(system: System, spec: Dict[str, Any]):
    lipid_sel = _select(system, spec.get("lipid_selection"), "hydrophobic_defects.lipid_selection")
    ref_sel = _select(system, spec.get("reference_selection"), "hydrophobic_defects.reference_selection")
    kwargs = _pick(
        spec,
        [
            "voxel_size",
            "z_bounds",
            "probe_radius",
            "defect_radius",
            "length_scale",
            "grid_mode",
            "leaflet",
            "leaflet_bins",
        ],
    )
    if "z_bounds" in kwargs:
        kwargs["z_bounds"] = _as_tuple(kwargs["z_bounds"], 2, "z_bounds")
    if spec.get("midplane_selection"):
        kwargs["midplane_selection"] = _select(
            system,
            spec.get("midplane_selection"),
            "hydrophobic_defects.midplane_selection",
        )
    return HydrophobicDefectPlan(lipid_sel, ref_sel, **kwargs)


def _build_bondi_ffv(system: System, spec: Dict[str, Any]):
    sel = _select(system, spec.get("selection"), "bondi_ffv.selection")
    kwargs = _pick(spec, ["bondi_scale", "probe_radius", "seed", "ninsert_per_nm3", "length_scale"])
    return BondiFfvPlan(sel, **kwargs)


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
    selection = spec.get("selection", spec.get("mask", ""))
    center_selection = spec.get("center_selection", "")
    if not selection:
        raise ValueError("density.selection is required")
    if not center_selection:
        raise ValueError("density.center_selection is required")
    kwargs = _pick(
        spec,
        [
            "box_unit",
            "region_size",
            "shift",
            "length_scale",
            "frame_indices",
            "chunk_frames",
            "device",
        ],
    )
    kwargs["selection"] = selection
    kwargs["center_selection"] = center_selection
    return _CallablePlan(density_analysis, kwargs)


def _build_native_contacts(system: System, spec: Dict[str, Any]):
    from .analysis.native_contacts import native_contacts as native_contacts_analysis
    kwargs = _pick(spec, ["mask", "mask2", "ref", "distance", "mindist", "maxdist", "image", "frame_indices", "chunk_frames"])
    return _CallablePlan(native_contacts_analysis, kwargs)


def _build_volmap(system: System, spec: Dict[str, Any]):
    from .analysis.volmap import volmap as volmap_analysis
    selection = spec.get("selection", spec.get("mask", ""))
    center_selection = spec.get("center_selection", "")
    if not selection:
        raise ValueError("volmap.selection is required")
    if not center_selection:
        raise ValueError("volmap.center_selection is required")
    kwargs = _pick(
        spec,
        [
            "box_unit",
            "region_size",
            "shift",
            "length_scale",
            "frame_indices",
            "chunk_frames",
            "device",
        ],
    )
    kwargs["selection"] = selection
    kwargs["center_selection"] = center_selection
    return _CallablePlan(volmap_analysis, kwargs)


def _build_surf(system: System, spec: Dict[str, Any]):
    from .analysis.surf import surf as surf_analysis

    kwargs = _pick(
        spec,
        [
            "mask",
            "algorithm",
            "probe_radius",
            "offset",
            "nbrcut",
            "solutemask",
            "n_sphere_points",
            "radii",
            "frame_indices",
            "chunk_frames",
            "device",
        ],
    )
    return _CallablePlan(surf_analysis, kwargs)


def _build_molsurf(system: System, spec: Dict[str, Any]):
    from .analysis.surf import molsurf as molsurf_analysis

    kwargs = _pick(
        spec,
        [
            "mask",
            "algorithm",
            "probe",
            "probe_radius",
            "offset",
            "n_sphere_points",
            "radii",
            "frame_indices",
            "chunk_frames",
            "device",
        ],
    )
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
    kwargs = _pick(
        spec,
        [
            "mask",
            "eigenvectors",
            "eigenvalues",
            "scalar_type",
            "average_coords",
            "frame_indices",
            "dtype",
            "chunk_frames",
        ],
    )
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
    # The agent runner does not yet construct the OpenMM objects GIST requires.
    raise NotImplementedError(
        "GIST analysis requires OpenMM system construction which is not yet supported in the agent runner."
    )


def _lipid_kwargs(spec: Dict[str, Any], names: list[str]):
    return _pick(spec, [*names, "length_scale", "frame_indices", "chunk_frames", "device"])


def _build_lipid_leaflets(system: System, spec: Dict[str, Any]):
    from .analysis.lipid import lipid_leaflets
    kwargs = _lipid_kwargs(spec, ["selection", "midplane_selection", "midplane_cutoff", "bins"])
    if not kwargs.get("selection"):
        raise ValueError("lipid_leaflets.selection is required")
    return _CallablePlan(lipid_leaflets, kwargs)


def _build_lipid_curved_leaflets(system: System, spec: Dict[str, Any]):
    from .analysis.lipid import lipid_curved_leaflets
    kwargs = _lipid_kwargs(
        spec,
        ["selection", "cutoff", "midplane_selection", "midplane_cutoff"],
    )
    if not kwargs.get("selection"):
        raise ValueError("lipid_curved_leaflets.selection is required")
    return _CallablePlan(lipid_curved_leaflets, kwargs)


def _build_lipid_z_positions(system: System, spec: Dict[str, Any]):
    from .analysis.lipid import lipid_z_positions
    kwargs = _lipid_kwargs(spec, ["membrane_selection", "height_selection", "bins"])
    if not kwargs.get("membrane_selection") or not kwargs.get("height_selection"):
        raise ValueError("lipid_z_positions requires membrane_selection and height_selection")
    return _CallablePlan(lipid_z_positions, kwargs)


def _build_lipid_z_thickness(system: System, spec: Dict[str, Any]):
    from .analysis.lipid import lipid_z_thickness
    kwargs = _lipid_kwargs(spec, ["selection"])
    if not kwargs.get("selection"):
        raise ValueError("lipid_z_thickness.selection is required")
    return _CallablePlan(lipid_z_thickness, kwargs)


def _build_lipid_z_angles(system: System, spec: Dict[str, Any]):
    from .analysis.lipid import lipid_z_angles
    kwargs = _lipid_kwargs(spec, ["atom_a", "atom_b", "degrees"])
    if not kwargs.get("atom_a") or not kwargs.get("atom_b"):
        raise ValueError("lipid_z_angles requires atom_a and atom_b")
    return _CallablePlan(lipid_z_angles, kwargs)


def _build_lipid_area(system: System, spec: Dict[str, Any]):
    from .analysis.lipid import lipid_area
    kwargs = _lipid_kwargs(spec, ["selection"])
    if not kwargs.get("selection"):
        raise ValueError("lipid_area.selection is required")
    kwargs["leaflets"] = _load_array_spec(spec.get("leaflets"), field="lipid_area.leaflets")
    return _CallablePlan(lipid_area, kwargs)


def _build_lipid_flip_flop(system: System, spec: Dict[str, Any]):
    from .analysis.lipid import lipid_flip_flop
    kwargs = _pick(spec, ["frame_cutoff", "chunk_frames", "device"])
    kwargs["leaflets"] = _load_array_spec(spec.get("leaflets"), field="lipid_flip_flop.leaflets")
    if spec.get("residue_ids") is not None:
        kwargs["residue_ids"] = _load_array_spec(
            spec.get("residue_ids"),
            field="lipid_flip_flop.residue_ids",
            default_npz_key="residue_ids",
        ).astype(np.int32)
    return _CallablePlan(lipid_flip_flop, kwargs)


def _build_lipid_neighbours(system: System, spec: Dict[str, Any]):
    from .analysis.lipid import lipid_neighbours
    kwargs = _lipid_kwargs(spec, ["selection", "cutoff"])
    if not kwargs.get("selection"):
        raise ValueError("lipid_neighbours.selection is required")
    return _CallablePlan(lipid_neighbours, kwargs)


def _build_lipid_neighbour_matrix(system: System, spec: Dict[str, Any]):
    from .analysis.lipid import lipid_neighbour_matrix
    kwargs = _lipid_kwargs(spec, ["selection", "cutoff"])
    if not kwargs.get("selection"):
        raise ValueError("lipid_neighbour_matrix.selection is required")
    return _CallablePlan(lipid_neighbour_matrix, kwargs)


def _build_lipid_largest_cluster(system: System, spec: Dict[str, Any]):
    from .analysis.lipid import lipid_largest_cluster
    kwargs = _lipid_kwargs(spec, ["selection", "cutoff"])
    if not kwargs.get("selection"):
        raise ValueError("lipid_largest_cluster.selection is required")
    return _CallablePlan(lipid_largest_cluster, kwargs)


def _build_lipid_membrane_thickness(system: System, spec: Dict[str, Any]):
    from .analysis.lipid import lipid_membrane_thickness
    kwargs = _lipid_kwargs(spec, ["selection", "bins"])
    if not kwargs.get("selection"):
        raise ValueError("lipid_membrane_thickness.selection is required")
    kwargs["leaflets"] = _load_array_spec(
        spec.get("leaflets"),
        field="lipid_membrane_thickness.leaflets",
    )
    return _CallablePlan(lipid_membrane_thickness, kwargs)


def _build_lipid_registration(system: System, spec: Dict[str, Any]):
    from .analysis.lipid import lipid_registration
    kwargs = _lipid_kwargs(spec, ["upper_selection", "lower_selection", "bins", "gaussian_sd"])
    if not kwargs.get("upper_selection") or not kwargs.get("lower_selection"):
        raise ValueError("lipid_registration requires upper_selection and lower_selection")
    kwargs["leaflets"] = _load_array_spec(
        spec.get("leaflets"),
        field="lipid_registration.leaflets",
    )
    return _CallablePlan(lipid_registration, kwargs)


def _build_lipid_msd(system: System, spec: Dict[str, Any]):
    from .analysis.lipid import lipid_msd
    kwargs = _lipid_kwargs(spec, ["selection", "com_removal_selection"])
    if not kwargs.get("selection"):
        raise ValueError("lipid_msd.selection is required")
    return _CallablePlan(lipid_msd, kwargs)


def _build_lipid_scc(system: System, spec: Dict[str, Any]):
    from .analysis.lipid import lipid_scc
    kwargs = _lipid_kwargs(spec, ["tail_selection"])
    if not kwargs.get("tail_selection"):
        raise ValueError("lipid_scc.tail_selection is required")
    if spec.get("normals") is not None:
        kwargs["normals"] = _load_array_spec(spec.get("normals"), field="lipid_scc.normals")
    return _CallablePlan(lipid_scc, kwargs)


PLAN_BUILDERS = {
    entry.plan_name: globals()[entry.build_fn]
    for entry in ANALYSIS_REGISTRY
    if entry.build_fn
}

CLI_TO_PLAN: dict[str, str] = {}
for entry in ANALYSIS_REGISTRY:
    CLI_TO_PLAN[entry.cli_name] = entry.plan_name
    for legacy_alias in entry.legacy_cli_aliases:
        CLI_TO_PLAN[legacy_alias] = entry.plan_name
