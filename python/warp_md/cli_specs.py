from __future__ import annotations

from typing import Any, Dict

from .cli_api import System, _select
from .cli_parse import (
    _parse_charges_arg,
    _parse_float_tuple,
    _parse_group_types_arg,
    _parse_int_list,
    _parse_int_tuple,
)


def _spec_rg(args, system: System) -> Dict[str, Any]:
    return {
        "selection": args.selection,
        "mass_weighted": args.mass_weighted,
    }


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
    return spec


def _spec_rdf(args, system: System) -> Dict[str, Any]:
    return {
        "sel_a": args.sel_a,
        "sel_b": args.sel_b,
        "bins": args.bins,
        "r_max": args.r_max,
        "pbc": args.pbc,
    }


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


SPEC_BUILDERS = {
    "rg": _spec_rg,
    "rmsd": _spec_rmsd,
    "msd": _spec_msd,
    "rotacf": _spec_rotacf,
    "conductivity": _spec_conductivity,
    "dielectric": _spec_dielectric,
    "dipole-alignment": _spec_dipole_alignment,
    "ion-pair-correlation": _spec_ion_pair,
    "structure-factor": _spec_structure_factor,
    "water-count": _spec_water_count,
    "equipartition": _spec_equipartition,
    "hbond": _spec_hbond,
    "rdf": _spec_rdf,
    "end-to-end": _spec_end_to_end,
    "contour-length": _spec_contour_length,
    "chain-rg": _spec_chain_rg,
    "bond-length-distribution": _spec_bond_length,
    "bond-angle-distribution": _spec_bond_angle,
    "persistence-length": _spec_persistence,
}
