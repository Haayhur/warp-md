from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .cli_api import System, _select
from .cli_parse import (
    _parse_charges_arg,
    _parse_float_tuple,
    _parse_group_types_arg,
    _parse_int_list,
    _parse_int_tuple,
)
from .cli_analysis_registry import ANALYSIS_REGISTRY


def _parse_optional_ref(raw):
    if raw is None:
        return None
    value = str(raw).strip()
    if value.lower() in {"topology", "top", "topo", "frame0", "first"}:
        return value
    return int(value)


def _parse_array_arg(raw: str, label: str) -> np.ndarray:
    value = str(raw).strip()
    path_value = value[4:] if value.startswith("npy:") else value
    path = Path(path_value)
    if path.exists() or path.suffix.lower() in {".npy", ".npz"}:
        data = np.load(path)
        if isinstance(data, np.lib.npyio.NpzFile):
            keys = list(data.files)
            if len(keys) != 1:
                raise ValueError(f"{label} npz must contain exactly one array")
            return np.asarray(data[keys[0]])
        return np.asarray(data)
    try:
        return np.asarray(json.loads(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be JSON or a .npy/.npz path") from exc


def _parse_pair_list_arg(raw: str, label: str) -> Any:
    value = str(raw).strip()
    if value.lower() == "sequential":
        return "sequential"
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be 'sequential' or JSON") from exc
    arr = np.asarray(data, dtype=object)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{label} must be a JSON list of pairs")
    return data


def _parse_quad_list_arg(raw: str, label: str) -> list[Any]:
    try:
        data = json.loads(str(raw).strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be JSON") from exc
    arr = np.asarray(data, dtype=object)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"{label} must be a JSON list of atom-index quartets")
    return data


def _spec_rg(args, system: System) -> Dict[str, Any]:
    return {
        "selection": args.selection,
        "mass_weighted": args.mass_weighted,
    }


def _spec_gyrate(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "selection": args.selection,
        "mass": args.mass,
        "axes": args.axes,
        "nomax": not args.max_radius,
        "tensor": args.tensor,
        "dtype": "dict",
    }
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    return _add_frame_indices_spec(spec, args)


def _spec_rmsd(args, system: System) -> Dict[str, Any]:
    return {
        "selection": args.selection,
        "reference": args.reference,
        "align": args.align,
    }


def _spec_msd(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "selection": args.selection,
        "group_by": args.group_by,
    }
    if args.axis:
        spec["axis"] = _parse_float_tuple(args.axis, 3, "axis")
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    if args.frame_decimation:
        spec["frame_decimation"] = _parse_int_tuple(args.frame_decimation, 2, "frame_decimation")
    if args.dt_decimation:
        spec["dt_decimation"] = _parse_int_tuple(args.dt_decimation, 4, "dt_decimation")
    if args.time_binning:
        spec["time_binning"] = _parse_float_tuple(args.time_binning, 2, "time_binning")
    if args.lag_mode:
        spec["lag_mode"] = args.lag_mode
    if args.max_lag is not None:
        spec["max_lag"] = args.max_lag
    if args.memory_budget_bytes is not None:
        spec["memory_budget_bytes"] = args.memory_budget_bytes
    if args.multi_tau_m is not None:
        spec["multi_tau_m"] = args.multi_tau_m
    if args.multi_tau_levels is not None:
        spec["multi_tau_levels"] = args.multi_tau_levels
    selection = _select(system, args.selection, "msd.selection")
    group_types = _parse_group_types_arg(args.group_types, system, selection, args.group_by)
    if group_types is not None:
        spec["group_types"] = group_types
    return spec


def _spec_rotacf(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "selection": args.selection,
        "group_by": args.group_by,
        "p2_legendre": args.p2_legendre,
    }
    orient = _parse_int_list(args.orientation, "orientation")
    if len(orient) not in (2, 3):
        raise ValueError("orientation must have 2 or 3 indices")
    spec["orientation"] = orient
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    if args.frame_decimation:
        spec["frame_decimation"] = _parse_int_tuple(args.frame_decimation, 2, "frame_decimation")
    if args.dt_decimation:
        spec["dt_decimation"] = _parse_int_tuple(args.dt_decimation, 4, "dt_decimation")
    if args.time_binning:
        spec["time_binning"] = _parse_float_tuple(args.time_binning, 2, "time_binning")
    if args.lag_mode:
        spec["lag_mode"] = args.lag_mode
    if args.max_lag is not None:
        spec["max_lag"] = args.max_lag
    if args.memory_budget_bytes is not None:
        spec["memory_budget_bytes"] = args.memory_budget_bytes
    if args.multi_tau_m is not None:
        spec["multi_tau_m"] = args.multi_tau_m
    if args.multi_tau_levels is not None:
        spec["multi_tau_levels"] = args.multi_tau_levels
    selection = _select(system, args.selection, "rotacf.selection")
    group_types = _parse_group_types_arg(args.group_types, system, selection, args.group_by)
    if group_types is not None:
        spec["group_types"] = group_types
    return spec


def _spec_conductivity(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "selection": args.selection,
        "group_by": args.group_by,
        "temperature": args.temperature,
        "transference": args.transference,
        "charges": _parse_charges_arg(args.charges, system),
    }
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    if args.frame_decimation:
        spec["frame_decimation"] = _parse_int_tuple(args.frame_decimation, 2, "frame_decimation")
    if args.dt_decimation:
        spec["dt_decimation"] = _parse_int_tuple(args.dt_decimation, 4, "dt_decimation")
    if args.time_binning:
        spec["time_binning"] = _parse_float_tuple(args.time_binning, 2, "time_binning")
    if args.lag_mode:
        spec["lag_mode"] = args.lag_mode
    if args.max_lag is not None:
        spec["max_lag"] = args.max_lag
    if args.memory_budget_bytes is not None:
        spec["memory_budget_bytes"] = args.memory_budget_bytes
    if args.multi_tau_m is not None:
        spec["multi_tau_m"] = args.multi_tau_m
    if args.multi_tau_levels is not None:
        spec["multi_tau_levels"] = args.multi_tau_levels
    selection = _select(system, args.selection, "conductivity.selection")
    group_types = _parse_group_types_arg(args.group_types, system, selection, args.group_by)
    if group_types is not None:
        spec["group_types"] = group_types
    return spec


def _spec_dielectric(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "selection": args.selection,
        "group_by": args.group_by,
        "charges": _parse_charges_arg(args.charges, system),
        "temperature": args.temperature,
        "make_whole": args.make_whole,
    }
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    selection = _select(system, args.selection, "dielectric.selection")
    group_types = _parse_group_types_arg(args.group_types, system, selection, args.group_by)
    if group_types is not None:
        spec["group_types"] = group_types
    return spec


def _spec_dipole_alignment(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "selection": args.selection,
        "group_by": args.group_by,
        "charges": _parse_charges_arg(args.charges, system),
    }
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    selection = _select(system, args.selection, "dipole_alignment.selection")
    group_types = _parse_group_types_arg(args.group_types, system, selection, args.group_by)
    if group_types is not None:
        spec["group_types"] = group_types
    return spec


def _spec_dipole_moments(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "selection": args.selection,
        "group_by": args.group_by,
        "charges": _parse_charges_arg(args.charges, system),
        "dtype": "dict",
    }
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    if args.group_types:
        selection = _select(system, args.selection, "dipole_moments.selection")
        group_types = _parse_group_types_arg(args.group_types, system, selection, args.group_by)
        if group_types is not None:
            spec["group_types"] = group_types
    return _add_frame_indices_spec(spec, args)


def _spec_ion_pair(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "selection": args.selection,
        "group_by": args.group_by,
        "rclust_cat": args.rclust_cat,
        "rclust_ani": args.rclust_ani,
        "cation_type": args.cation_type,
        "anion_type": args.anion_type,
        "max_cluster": args.max_cluster,
    }
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    if args.lag_mode:
        spec["lag_mode"] = args.lag_mode
    if args.max_lag is not None:
        spec["max_lag"] = args.max_lag
    if args.memory_budget_bytes is not None:
        spec["memory_budget_bytes"] = args.memory_budget_bytes
    if args.multi_tau_m is not None:
        spec["multi_tau_m"] = args.multi_tau_m
    if args.multi_tau_levels is not None:
        spec["multi_tau_levels"] = args.multi_tau_levels
    selection = _select(system, args.selection, "ion_pair_correlation.selection")
    group_types = _parse_group_types_arg(args.group_types, system, selection, args.group_by)
    if group_types is not None:
        spec["group_types"] = group_types
    return spec


def _spec_structure_factor(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "selection": args.selection,
        "bins": args.bins,
        "r_max": args.r_max,
        "q_bins": args.q_bins,
        "q_max": args.q_max,
        "pbc": args.pbc,
    }
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    return spec


def _spec_water_count(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "water_selection": args.water_selection,
        "center_selection": args.center_selection,
        "box_unit": _parse_float_tuple(args.box_unit, 3, "box_unit"),
        "region_size": _parse_float_tuple(args.region_size, 3, "region_size"),
    }
    if args.shift:
        spec["shift"] = _parse_float_tuple(args.shift, 3, "shift")
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    return spec


def _spec_free_volume(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "selection": args.selection,
        "center_selection": args.center_selection,
    }
    if hasattr(args, 'box_unit') and args.box_unit is not None:
        spec["box_unit"] = _parse_float_tuple(args.box_unit, 3, "box_unit")
    if hasattr(args, 'region_size') and args.region_size is not None:
        spec["region_size"] = _parse_float_tuple(args.region_size, 3, "region_size")
    if args.probe_radius is not None:
        spec["probe_radius"] = args.probe_radius
    if args.shift:
        spec["shift"] = _parse_float_tuple(args.shift, 3, "shift")
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    return spec


def _spec_hydrophobic_defects(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "lipid_selection": args.lipid_selection,
        "reference_selection": args.reference_selection,
        "leaflet": args.leaflet,
        "leaflet_bins": args.leaflet_bins,
        "voxel_size": args.voxel_size,
        "grid_mode": args.grid_mode,
    }
    if args.midplane_selection:
        spec["midplane_selection"] = args.midplane_selection
    if args.z_bounds:
        spec["z_bounds"] = _parse_float_tuple(args.z_bounds, 2, "z_bounds")
    if args.probe_radius is not None:
        spec["probe_radius"] = args.probe_radius
    if args.defect_radius is not None:
        spec["defect_radius"] = args.defect_radius
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    return spec


def _spec_bondi_ffv(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "selection": args.selection,
    }
    if args.bondi_scale is not None:
        spec["bondi_scale"] = args.bondi_scale
    if args.probe_radius is not None:
        spec["probe_radius"] = args.probe_radius
    if args.seed is not None:
        spec["seed"] = args.seed
    if args.ninsert_per_nm3 is not None:
        spec["ninsert_per_nm3"] = args.ninsert_per_nm3
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    return spec


def _spec_equipartition(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "selection": args.selection,
        "group_by": args.group_by,
    }
    if args.velocity_scale is not None:
        spec["velocity_scale"] = args.velocity_scale
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    selection = _select(system, args.selection, "equipartition.selection")
    group_types = _parse_group_types_arg(args.group_types, system, selection, args.group_by)
    if group_types is not None:
        spec["group_types"] = group_types
    return spec


def _spec_hbond(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "donors": args.donors,
        "acceptors": args.acceptors,
        "dist_cutoff": args.dist_cutoff,
    }
    if args.hydrogens:
        if args.angle_cutoff is None:
            raise ValueError("angle_cutoff is required when hydrogens are provided")
        spec["hydrogens"] = args.hydrogens
        spec["angle_cutoff"] = args.angle_cutoff
    return _add_frame_indices_spec(spec, args)


def _spec_saltbr(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "selection": args.selection,
        "group_by": args.group_by,
    }
    if args.charges:
        spec["charges"] = _parse_charges_arg(args.charges, system)
    if args.truncate is not None:
        spec["truncate"] = args.truncate
    if args.contact_cutoff is not None:
        spec["contact_cutoff"] = args.contact_cutoff
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    return _add_frame_indices_spec(spec, args)


def _parse_str_list(raw: str, label: str) -> list[str]:
    values = [part.strip() for part in raw.replace(",", " ").split() if part.strip()]
    if not values:
        raise ValueError(f"{label} must have at least one value")
    return values


def _add_optional_water_resnames(spec: Dict[str, Any], args) -> Dict[str, Any]:
    if args.water_resnames:
        spec["water_resnames"] = _parse_str_list(args.water_resnames, "water_resnames")
    return spec


def _spec_h2order(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "axis": args.axis,
        "bin": args.bin,
    }
    if args.selection:
        spec["selection"] = args.selection
    if args.charges:
        spec["charges"] = _parse_charges_arg(args.charges, system)
    if args.n_slices is not None:
        spec["n_slices"] = args.n_slices
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    return _add_frame_indices_spec(_add_optional_water_resnames(spec, args), args)


def _spec_hydorder(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "selection": args.selection,
        "axis": args.axis,
        "bin": args.bin,
        "tblock": args.tblock,
    }
    if args.sgang1 is not None:
        spec["sgang1"] = args.sgang1
    if args.sgang2 is not None:
        spec["sgang2"] = args.sgang2
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    return _add_frame_indices_spec(spec, args)


def _add_solvent_triplet_specs(spec: Dict[str, Any], args) -> Dict[str, Any]:
    for arg_name in ("atom1_indices", "atom2_indices", "atom3_indices"):
        raw = getattr(args, arg_name)
        if raw:
            spec[arg_name] = _parse_int_list(raw, arg_name)
    return spec


def _spec_sorient(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "solute_selection": args.solute_selection,
        "r_min": args.r_min,
        "r_max": args.r_max,
        "cbin": args.cbin,
        "rbin": args.rbin,
        "use_com": args.use_com,
        "use_vector23": args.use_vector23,
    }
    if args.solvent_selection:
        spec["solvent_selection"] = args.solvent_selection
    if args.r_profile_max is not None:
        spec["r_profile_max"] = args.r_profile_max
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    return _add_frame_indices_spec(
        _add_optional_water_resnames(_add_solvent_triplet_specs(spec, args), args),
        args,
    )


def _spec_spol(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "solute_selection": args.solute_selection,
        "r_min": args.r_min,
        "r_max": args.r_max,
        "bin": args.bin,
        "use_com": args.use_com,
        "reference_atom": args.reference_atom,
        "direction_atom_offsets": _parse_int_tuple(
            args.direction_atom_offsets, 3, "direction_atom_offsets"
        ),
        "refdip": args.refdip,
    }
    if args.solvent_selection:
        spec["solvent_selection"] = args.solvent_selection
    if args.charges:
        spec["charges"] = _parse_charges_arg(args.charges, system)
    if args.r_hist_max is not None:
        spec["r_hist_max"] = args.r_hist_max
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    return _add_frame_indices_spec(
        _add_optional_water_resnames(_add_solvent_triplet_specs(spec, args), args),
        args,
    )


def _spec_rama(args, system: System) -> Dict[str, Any]:
    return _add_frame_indices_spec(
        {
            "selection": args.selection,
            "range360": args.range360,
        },
        args,
    )


def _add_dynamics_spec(spec: Dict[str, Any], args) -> Dict[str, Any]:
    if args.frame_decimation:
        spec["frame_decimation"] = _parse_int_tuple(args.frame_decimation, 2, "frame_decimation")
    if args.dt_decimation:
        spec["dt_decimation"] = _parse_int_tuple(args.dt_decimation, 4, "dt_decimation")
    if args.time_binning:
        spec["time_binning"] = _parse_float_tuple(args.time_binning, 2, "time_binning")
    if args.lag_mode:
        spec["lag_mode"] = args.lag_mode
    if args.max_lag is not None:
        spec["max_lag"] = args.max_lag
    if args.memory_budget_bytes is not None:
        spec["memory_budget_bytes"] = args.memory_budget_bytes
    if args.multi_tau_m is not None:
        spec["multi_tau_m"] = args.multi_tau_m
    if args.multi_tau_levels is not None:
        spec["multi_tau_levels"] = args.multi_tau_levels
    return spec


def _spec_current(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "selection": args.selection,
        "temperature": args.temperature,
        "group_by": args.group_by,
        "make_whole": args.make_whole,
    }
    if args.charges:
        spec["charges"] = _parse_charges_arg(args.charges, system)
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    if args.group_types:
        selection = _select(system, args.selection, "current.selection")
        group_types = _parse_group_types_arg(args.group_types, system, selection, args.group_by)
        if group_types is not None:
            spec["group_types"] = group_types
    return _add_frame_indices_spec(_add_dynamics_spec(spec, args), args)


def _spec_bundle(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "top_selection": args.top_selection,
        "bottom_selection": args.bottom_selection,
        "n_axes": args.n_axes,
        "use_z_reference": args.use_z_reference,
        "mass_weighted": args.mass_weighted,
    }
    if args.kink_selection:
        spec["kink_selection"] = args.kink_selection
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    return _add_frame_indices_spec(spec, args)


def _spec_helix(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "selection": args.selection,
        "fit": args.fit,
        "check_each_frame": args.check_each_frame,
    }
    if args.residue_start is not None:
        spec["residue_start"] = args.residue_start
    if args.residue_end is not None:
        spec["residue_end"] = args.residue_end
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    return _add_frame_indices_spec(spec, args)


def _spec_helixorient(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "ca_selection": args.ca_selection,
        "incremental": args.incremental,
    }
    if args.sidechain_selection:
        spec["sidechain_selection"] = args.sidechain_selection
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    return _add_frame_indices_spec(spec, args)


def _spec_mdmat(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "selection": args.selection,
        "truncate": args.truncate,
        "include_contacts": args.include_contacts,
        "include_frames": args.include_frames,
        "frames_mode": args.frames_mode,
        "memory_budget_bytes": args.memory_budget_bytes,
    }
    if args.frames_out:
        spec["frames_out"] = args.frames_out
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    return _add_frame_indices_spec(spec, args)


def _spec_docking(args, system: System) -> Dict[str, Any]:
    return _add_frame_indices_spec(
        {
            "receptor_mask": args.receptor_mask,
            "ligand_mask": args.ligand_mask,
            "close_contact_cutoff": args.close_contact_cutoff,
            "hydrophobic_cutoff": args.hydrophobic_cutoff,
            "hydrogen_bond_cutoff": args.hydrogen_bond_cutoff,
            "clash_cutoff": args.clash_cutoff,
            "salt_bridge_cutoff": args.salt_bridge_cutoff,
            "halogen_bond_cutoff": args.halogen_bond_cutoff,
            "metal_coordination_cutoff": args.metal_coordination_cutoff,
            "cation_pi_cutoff": args.cation_pi_cutoff,
            "pi_pi_cutoff": args.pi_pi_cutoff,
            "hbond_min_angle_deg": args.hbond_min_angle_deg,
            "donor_hydrogen_cutoff": args.donor_hydrogen_cutoff,
            "allow_missing_hydrogen": args.allow_missing_hydrogen,
            "length_scale": args.length_scale,
            "max_events_per_frame": args.max_events_per_frame,
        },
        args,
    )


def _spec_dssp(args, system: System) -> Dict[str, Any]:
    return _add_frame_indices_spec(
        {
            "mask": args.mask,
            "simplified": args.simplified,
            "dtype": args.dtype,
        },
        args,
    )


def _spec_diffusion(args, system: System) -> Dict[str, Any]:
    return _add_frame_indices_spec(
        {
            "mask": args.mask,
            "tstep": args.tstep,
            "individual": args.individual,
        },
        args,
    )


def _spec_rmsf(args, system: System) -> Dict[str, Any]:
    return _add_frame_indices_spec(
        {
            "mask": args.mask,
            "byres": args.byres,
            "bymask": args.bymask,
            "calcadp": args.calcadp,
            "length_scale": args.length_scale,
        },
        args,
    )


def _parse_native_contacts_ref(raw: str):
    value = str(raw).strip()
    if value.lower() in {"topology", "top", "topo", "frame0", "first"}:
        return value
    return int(value)


def _spec_native_contacts(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "mask": args.mask,
        "ref": _parse_native_contacts_ref(args.ref),
        "distance": args.distance,
        "image": args.image,
    }
    if args.mask2:
        spec["mask2"] = args.mask2
    if args.mindist is not None:
        spec["mindist"] = args.mindist
    if args.maxdist is not None:
        spec["maxdist"] = args.maxdist
    return _add_frame_indices_spec(spec, args)


def _spec_pca(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "mask": args.mask,
        "n_vecs": args.n_vecs,
        "fit": args.fit,
        "dtype": args.dtype,
    }
    if args.ref is not None:
        spec["ref"] = _parse_optional_ref(args.ref)
    if args.ref_mask:
        spec["ref_mask"] = args.ref_mask
    return spec


def _spec_projection(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "mask": args.mask,
        "eigenvectors": _parse_array_arg(args.eigenvectors, "eigenvectors"),
        "scalar_type": args.scalar_type,
        "dtype": args.dtype,
    }
    if args.eigenvalues:
        spec["eigenvalues"] = _parse_array_arg(args.eigenvalues, "eigenvalues")
    if args.average_coords:
        spec["average_coords"] = _parse_array_arg(args.average_coords, "average_coords")
    return _add_frame_indices_spec(spec, args)


def _spec_tordiff(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "mask": args.mask,
        "mass": args.mass,
        "time": args.time,
        "return_transitions": args.return_transitions,
        "transition_lag": args.transition_lag,
    }
    if args.diffout:
        spec["diffout"] = args.diffout
    return _add_frame_indices_spec(spec, args)


def _spec_nmr(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "selection": args.selection,
        "method": args.method,
        "order": args.order,
        "tstep": args.tstep,
        "tcorr": args.tcorr,
        "length_scale": args.length_scale,
        "pbc": args.pbc,
    }
    if args.vector_pairs:
        spec["vector_pairs"] = _parse_pair_list_arg(args.vector_pairs, "vector_pairs")
    return _add_frame_indices_spec(spec, args)


def _spec_jcoupling(args, system: System) -> Dict[str, Any]:
    return _add_frame_indices_spec(
        {
            "dihedrals": _parse_quad_list_arg(args.dihedrals, "dihedrals"),
            "karplus": _parse_float_tuple(args.karplus, 3, "karplus"),
            "kfile": args.kfile,
            "phase_deg": args.phase_deg,
            "length_scale": args.length_scale,
            "pbc": args.pbc,
            "return_dihedral": args.return_dihedral,
        },
        args,
    )


def _spec_gist(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "energy_method": args.energy_method,
        "grid_spacing": args.grid_spacing,
        "padding": args.padding,
        "temperature": args.temperature,
        "length_scale": args.length_scale,
        "orientation_bins": args.orientation_bins,
        "water_resnames": tuple(value.strip() for value in args.water_resnames.split(",") if value.strip()),
    }
    if args.solute_selection:
        spec["solute_selection"] = args.solute_selection
    if args.max_frames is not None:
        spec["max_frames"] = args.max_frames
    if args.bulk_density is not None:
        spec["bulk_density"] = args.bulk_density
    return _add_frame_indices_spec(spec, args)


def _spec_grid_map(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "selection": args.selection,
        "center_selection": args.center_selection,
        "box_unit": _parse_float_tuple(args.box_unit, 3, "box_unit"),
        "region_size": _parse_float_tuple(args.region_size, 3, "region_size"),
    }
    if args.shift:
        spec["shift"] = _parse_float_tuple(args.shift, 3, "shift")
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    return _add_frame_indices_spec(spec, args)


def _spec_density(args, system: System) -> Dict[str, Any]:
    return _spec_grid_map(args, system)


def _spec_volmap(args, system: System) -> Dict[str, Any]:
    return _spec_grid_map(args, system)


def _parse_radii_arg(raw: str):
    lowered = raw.strip().lower()
    if lowered in {"gb", "parse", "vdw"}:
        return lowered
    return [float(value) for value in _parse_str_list(raw, "radii")]


def _spec_surface(args, system: System, *, molsurf: bool = False) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "mask": args.mask,
        "algorithm": args.algorithm,
        "n_sphere_points": args.n_sphere_points,
        "radii_mode": args.radii_mode,
        "atom_area": args.atom_area,
        "volume": args.volume,
        "residue_area": args.residue_area,
    }
    if args.probe_radius is not None:
        spec["probe_radius"] = args.probe_radius
    if molsurf and args.probe is not None:
        spec["probe"] = args.probe
    if args.offset is not None:
        spec["offset"] = args.offset
    if not molsurf and args.nbrcut is not None:
        spec["nbrcut"] = args.nbrcut
    if args.solutemask:
        spec["solutemask"] = args.solutemask
    if args.radii:
        spec["radii"] = _parse_radii_arg(args.radii)
    return _add_frame_indices_spec(spec, args)


def _spec_surf(args, system: System) -> Dict[str, Any]:
    return _spec_surface(args, system, molsurf=False)


def _spec_molsurf(args, system: System) -> Dict[str, Any]:
    return _spec_surface(args, system, molsurf=True)


def _spec_watershell(args, system: System) -> Dict[str, Any]:
    return _add_frame_indices_spec(
        {
            "solute_mask": args.solute_mask,
            "solvent_mask": args.solvent_mask,
            "lower": args.lower,
            "upper": args.upper,
            "image": args.image,
        },
        args,
    )


def _spec_pairdist(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "mask": args.mask,
        "delta": args.delta,
        "mode": args.mode,
        "image": args.image,
    }
    if args.mask2:
        spec["mask2"] = args.mask2
    if args.maxdist is not None:
        spec["maxdist"] = args.maxdist
    if args.frame_indices:
        spec["frame_indices"] = _parse_int_list(args.frame_indices, "frame_indices")
    return spec


def _spec_dist_extrema(args, mode: str) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "mask": args.mask,
        "mode": mode,
        "image": args.image,
    }
    if args.mask2:
        spec["mask2"] = args.mask2
    if args.maxdist is not None:
        spec["maxdist"] = args.maxdist
    if args.frame_indices:
        spec["frame_indices"] = _parse_int_list(args.frame_indices, "frame_indices")
    return spec


def _spec_mindist(args, system: System) -> Dict[str, Any]:
    return _spec_dist_extrema(args, "min")


def _spec_maxdist(args, system: System) -> Dict[str, Any]:
    return _spec_dist_extrema(args, "max")


def _spec_rdf(args, system: System) -> Dict[str, Any]:
    return {
        "sel_a": args.sel_a,
        "sel_b": args.sel_b,
        "bins": args.bins,
        "r_max": args.r_max,
        "pbc": args.pbc,
    }


def _add_frame_indices_spec(spec: Dict[str, Any], args) -> Dict[str, Any]:
    if args.frame_indices:
        spec["frame_indices"] = _parse_int_list(args.frame_indices, "frame_indices")
    return spec


def _spec_drid(args, system: System) -> Dict[str, Any]:
    return _add_frame_indices_spec(
        {
            "selection": args.selection,
            "exclude_bonds": args.exclude_bonds,
        },
        args,
    )


def _spec_shape_descriptors(args, system: System) -> Dict[str, Any]:
    return _add_frame_indices_spec(
        {
            "selection": args.selection,
            "mass": args.mass,
        },
        args,
    )


def _spec_runningavg(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {"selection": args.selection}
    if args.window is not None:
        spec["window"] = args.window
    return _add_frame_indices_spec(spec, args)


def _spec_lineardensity(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "selection": args.selection,
        "axis": args.axis,
        "bin": args.bin,
        "weight": args.weight,
        "norm": args.norm,
    }
    if args.range:
        spec["range"] = _parse_float_tuple(args.range, 2, "range")
    if args.charges:
        spec["charges"] = _parse_charges_arg(args.charges, system)
    if args.cross_section_area is not None:
        spec["cross_section_area"] = args.cross_section_area
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    return _add_frame_indices_spec(spec, args)


def _spec_nematic_order(args, system: System) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "selection": args.selection,
        "pbc": args.pbc,
    }
    if args.reference_axis:
        spec["reference_axis"] = _parse_float_tuple(args.reference_axis, 3, "reference_axis")
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    return _add_frame_indices_spec(spec, args)


def _spec_kabsch_sander(args, system: System) -> Dict[str, Any]:
    return _add_frame_indices_spec(
        {
            "selection": args.selection,
            "energy_cutoff": args.energy_cutoff,
            "dtype": "dict",
        },
        args,
    )


def _spec_end_to_end(args, system: System) -> Dict[str, Any]:
    return {"selection": args.selection}


def _spec_contour_length(args, system: System) -> Dict[str, Any]:
    return {"selection": args.selection}


def _spec_chain_rg(args, system: System) -> Dict[str, Any]:
    return {"selection": args.selection}


def _spec_bond_length(args, system: System) -> Dict[str, Any]:
    return {
        "selection": args.selection,
        "bins": args.bins,
        "r_max": args.r_max,
    }


def _spec_bond_angle(args, system: System) -> Dict[str, Any]:
    return {
        "selection": args.selection,
        "bins": args.bins,
        "degrees": args.degrees,
    }


def _spec_persistence(args, system: System) -> Dict[str, Any]:
    return {"selection": args.selection}


def _add_lipid_common_spec(spec: Dict[str, Any], args) -> Dict[str, Any]:
    if args.length_scale is not None:
        spec["length_scale"] = args.length_scale
    if args.frame_indices:
        spec["frame_indices"] = _parse_int_list(args.frame_indices, "frame_indices")
    return spec


def _spec_lipid_leaflets(args, system: System) -> Dict[str, Any]:
    spec = {
        "selection": args.selection,
        "midplane_cutoff": args.midplane_cutoff,
        "bins": args.bins,
    }
    if args.midplane_selection:
        spec["midplane_selection"] = args.midplane_selection
    return _add_lipid_common_spec(spec, args)


def _spec_lipid_curved_leaflets(args, system: System) -> Dict[str, Any]:
    spec = {
        "selection": args.selection,
        "cutoff": args.cutoff,
        "midplane_cutoff": args.midplane_cutoff,
    }
    if args.midplane_selection:
        spec["midplane_selection"] = args.midplane_selection
    return _add_lipid_common_spec(spec, args)


def _spec_lipid_z_positions(args, system: System) -> Dict[str, Any]:
    return _add_lipid_common_spec(
        {
            "membrane_selection": args.membrane_selection,
            "height_selection": args.height_selection,
            "bins": args.bins,
        },
        args,
    )


def _spec_lipid_z_thickness(args, system: System) -> Dict[str, Any]:
    return _add_lipid_common_spec({"selection": args.selection}, args)


def _spec_lipid_z_angles(args, system: System) -> Dict[str, Any]:
    return _add_lipid_common_spec(
        {"atom_a": args.atom_a, "atom_b": args.atom_b, "degrees": args.degrees},
        args,
    )


def _spec_lipid_area(args, system: System) -> Dict[str, Any]:
    return _add_lipid_common_spec({"selection": args.selection, "leaflets": args.leaflets}, args)


def _spec_lipid_flip_flop(args, system: System) -> Dict[str, Any]:
    spec = {"leaflets": args.leaflets, "frame_cutoff": args.frame_cutoff}
    if args.residue_ids:
        spec["residue_ids"] = args.residue_ids
    return spec


def _spec_lipid_neighbours(args, system: System) -> Dict[str, Any]:
    return _add_lipid_common_spec(
        {"selection": args.selection, "cutoff": args.cutoff},
        args,
    )


def _spec_lipid_neighbour_matrix(args, system: System) -> Dict[str, Any]:
    return _spec_lipid_neighbours(args, system)


def _spec_lipid_largest_cluster(args, system: System) -> Dict[str, Any]:
    return _spec_lipid_neighbours(args, system)


def _spec_lipid_membrane_thickness(args, system: System) -> Dict[str, Any]:
    return _add_lipid_common_spec(
        {"selection": args.selection, "leaflets": args.leaflets, "bins": args.bins},
        args,
    )


def _spec_lipid_registration(args, system: System) -> Dict[str, Any]:
    return _add_lipid_common_spec(
        {
            "upper_selection": args.upper_selection,
            "lower_selection": args.lower_selection,
            "leaflets": args.leaflets,
            "bins": args.bins,
            "gaussian_sd": args.gaussian_sd,
        },
        args,
    )


def _spec_lipid_msd(args, system: System) -> Dict[str, Any]:
    spec = {"selection": args.selection}
    if args.com_removal_selection:
        spec["com_removal_selection"] = args.com_removal_selection
    return _add_lipid_common_spec(spec, args)


def _spec_lipid_scc(args, system: System) -> Dict[str, Any]:
    spec = {"tail_selection": args.tail_selection}
    if args.normals:
        spec["normals"] = args.normals
    return _add_lipid_common_spec(spec, args)


SPEC_BUILDERS = {
    entry.cli_name: globals()[entry.spec_fn]
    for entry in ANALYSIS_REGISTRY
    if entry.spec_fn is not None
}
