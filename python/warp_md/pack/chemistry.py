from __future__ import annotations

import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import Box, OutputSpec, PackConfig, Structure
from .data import ion_metadata, ion_pdb, salt_recipe, water_pdb

try:
    from warp_md import traj_py  # type: ignore
except Exception:
    traj_py = None

AVOGADRO = 6.022_140_76e23
ANGSTROM3_TO_LITER = 1.0e-27
WATER_MOLARITY = 55.5
DEFAULT_PACKING_FRACTION = 0.80
DEFAULT_NEUTRALIZE_CATION = "Na+"
DEFAULT_NEUTRALIZE_ANION = "Cl-"


@dataclass(frozen=True)
class ResolvedSalt:
    name: str | None
    formula: str | None
    species: dict[str, int]


def _normalize_box_size(
    box_size: float | Sequence[float],
) -> tuple[float, float, float]:
    if isinstance(box_size, (int, float)):
        side = float(box_size)
        return (side, side, side)
    values = tuple(float(value) for value in box_size)
    if len(values) != 3:
        raise ValueError("box_size must be a scalar or three floats")
    return values


def _normalize_catalog_entries(
    entries: Sequence[Mapping[str, Any]] | None,
    key: str,
    aliases_key: str = "aliases",
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for raw in entries or []:
        entry = dict(raw)
        name = str(entry.get(key, "")).strip()
        if not name:
            raise ValueError(f"custom catalog entries require '{key}'")
        aliases = [str(alias).strip() for alias in entry.get(aliases_key, [])]
        entry[key] = name
        entry[aliases_key] = aliases
        for alias in [name, *aliases]:
            lookup[alias.strip().lower()] = entry
    return lookup


def _ion_entry(
    species: str,
    custom_ions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    key = species.strip().lower()
    if key in custom_ions:
        return custom_ions[key]
    return ion_metadata(species)


def _canonicalize_salt_species(
    species: Mapping[str, Any],
    custom_ions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, int]:
    if not species:
        raise ValueError("salt mapping requires non-empty species")
    custom_lookup = _normalize_catalog_entries(custom_ions, "species")
    canonical: dict[str, int] = {}
    net_charge = 0
    has_positive = False
    has_negative = False
    for raw_name, raw_count in species.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("salt mapping contains an empty ion species name")
        count = int(raw_count)
        if count <= 0:
            raise ValueError("salt species counts must be > 0")
        entry = _ion_entry(name, custom_lookup)
        charge = int(entry["charge_e"])
        net_charge += charge * count
        has_positive |= charge > 0
        has_negative |= charge < 0
        canonical_name = str(entry["species"])
        canonical[canonical_name] = canonical.get(canonical_name, 0) + count
    if not has_positive or not has_negative:
        raise ValueError("salt must include at least one cation and one anion")
    if net_charge != 0:
        raise ValueError("salt formula/species must be charge neutral")
    return canonical


def _resolve_salt(
    salt: str | Mapping[str, Any] | None,
    custom_salts: Sequence[Mapping[str, Any]] | None = None,
    custom_ions: Sequence[Mapping[str, Any]] | None = None,
) -> ResolvedSalt | None:
    if salt is None:
        return None
    custom_lookup = _normalize_catalog_entries(custom_salts, "name")
    if isinstance(salt, Mapping):
        species = _canonicalize_salt_species(dict(salt.get("species", {})), custom_ions)
        return ResolvedSalt(
            name=str(salt["name"]).strip() if salt.get("name") else None,
            formula=str(salt["formula"]).strip() if salt.get("formula") else None,
            species=species,
        )
    key = salt.strip().lower()
    if key in custom_lookup:
        entry = custom_lookup[key]
        return ResolvedSalt(
            name=entry.get("name"),
            formula=entry.get("formula"),
            species=_canonicalize_salt_species(dict(entry["species"]), custom_ions),
        )
    entry = salt_recipe(salt)
    return ResolvedSalt(
        name=entry.get("name"),
        formula=entry.get("formula"),
        species=_canonicalize_salt_species(dict(entry["species"]), custom_ions),
    )


def _resolve_ion_templates(
    custom_ions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    lookup = _normalize_catalog_entries(custom_ions, "species")
    templates: dict[str, str] = {}
    for entry in lookup.values():
        species = str(entry["species"])
        template = str(entry.get("template", "")).strip()
        if not template:
            raise ValueError(f"custom ion '{species}' requires template")
        if not Path(template).exists():
            raise ValueError(f"custom ion '{species}' template does not exist: {template}")
        templates[species] = template
    return templates


def estimate_salt_formula_units(
    box_size: float | Sequence[float],
    molar: float,
) -> int:
    if molar < 0:
        raise ValueError("molar must be >= 0")
    lx, ly, lz = _normalize_box_size(box_size)
    volume_l = lx * ly * lz * ANGSTROM3_TO_LITER
    return int(round(molar * volume_l * AVOGADRO))


def estimate_water_count(
    box_size: float | Sequence[float],
    occupied_volume_angstrom3: float = 0.0,
    packing_fraction: float = DEFAULT_PACKING_FRACTION,
) -> int:
    lx, ly, lz = _normalize_box_size(box_size)
    free_volume = max(lx * ly * lz - occupied_volume_angstrom3, 0.0)
    estimate = WATER_MOLARITY * AVOGADRO * ANGSTROM3_TO_LITER * free_volume * packing_fraction
    return int(round(max(estimate, 0.0)))


def _box_volume_angstrom3(box_size: tuple[float, float, float]) -> float:
    return box_size[0] * box_size[1] * box_size[2]


def _box_volume_liter(box_size: tuple[float, float, float]) -> float:
    return _box_volume_angstrom3(box_size) * ANGSTROM3_TO_LITER


def _achieved_molarity(
    box_size: tuple[float, float, float],
    formula_units: int,
) -> float:
    volume_liter = _box_volume_liter(box_size)
    if volume_liter <= 0:
        return 0.0
    return float(formula_units / AVOGADRO / volume_liter)


def _counterion_count(solute_net_charge_e: float, ion_charge_e: int) -> int:
    if ion_charge_e == 0:
        raise ValueError("neutralization ion must have non-zero charge")
    return int(ceil(abs(float(solute_net_charge_e)) / abs(int(ion_charge_e))))


def _pick_counterion(
    *,
    need_positive: bool,
    resolved_salt: ResolvedSalt | None,
    neutralize_with: str | None,
    custom_ions: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    custom_lookup = _normalize_catalog_entries(custom_ions, "species")
    if neutralize_with:
        entry = _ion_entry(neutralize_with, custom_lookup)
        charge = int(entry["charge_e"])
        if (need_positive and charge <= 0) or (not need_positive and charge >= 0):
            direction = "positive" if need_positive else "negative"
            raise ValueError(f"neutralize_with must resolve to a {direction} ion")
        return str(entry["species"]), entry
    if resolved_salt is not None:
        for species in resolved_salt.species:
            entry = _ion_entry(species, custom_lookup)
            charge = int(entry["charge_e"])
            if (need_positive and charge > 0) or (not need_positive and charge < 0):
                return str(entry["species"]), entry
    fallback = DEFAULT_NEUTRALIZE_CATION if need_positive else DEFAULT_NEUTRALIZE_ANION
    entry = _ion_entry(fallback, custom_lookup)
    return str(entry["species"]), entry


def _resolve_chemistry_fallback(
    box_size: float | Sequence[float],
    *,
    solvent_model: str = "tip3p",
    salt: str | Mapping[str, Any] | None = None,
    salt_molar: float | None = None,
    water_count: int | None = None,
    occupied_volume_angstrom3: float = 0.0,
    custom_ions: Sequence[Mapping[str, Any]] | None = None,
    custom_salts: Sequence[Mapping[str, Any]] | None = None,
    neutralize: bool = False,
    solute_net_charge_e: float | None = None,
    neutralize_with: str | None = None,
) -> dict[str, Any]:
    resolved_box = _normalize_box_size(box_size)
    resolved_salt = _resolve_salt(salt, custom_salts, custom_ions)
    custom_templates = _resolve_ion_templates(custom_ions)
    formula_units = 0
    salt_ion_counts: dict[str, int] = {}
    ion_counts: dict[str, int] = {}
    warnings: list[str] = []
    if resolved_salt is not None and salt_molar is not None:
        formula_units = estimate_salt_formula_units(resolved_box, salt_molar)
        salt_ion_counts = {
            species: int(count) * formula_units
            for species, count in resolved_salt.species.items()
        }
        ion_counts.update(salt_ion_counts)
    resolved_water_count = (
        int(water_count)
        if water_count is not None
        else estimate_water_count(resolved_box, occupied_volume_angstrom3)
    )
    ion_templates = {
        species: custom_templates.get(species) or ion_pdb(species) for species in ion_counts
    }
    ion_templates.update(custom_templates)

    neutralization: dict[str, Any] = {
        "enabled": bool(neutralize),
        "solute_net_charge_e": solute_net_charge_e,
        "counterion": None,
        "counterion_count": 0,
        "counterion_charge_e": None,
        "applied_charge_e": 0.0,
        "residual_charge_e": solute_net_charge_e,
    }
    if neutralize:
        if solute_net_charge_e is None:
            raise ValueError("solute_net_charge_e is required when neutralize=True")
        if abs(float(solute_net_charge_e)) < 1.0e-8:
            neutralization["residual_charge_e"] = 0.0
        else:
            need_positive = float(solute_net_charge_e) < 0.0
            species, counterion_entry = _pick_counterion(
                need_positive=need_positive,
                resolved_salt=resolved_salt,
                neutralize_with=neutralize_with,
                custom_ions=custom_ions,
            )
            charge_e = int(counterion_entry["charge_e"])
            count = _counterion_count(float(solute_net_charge_e), charge_e)
            applied_charge = float(count * charge_e)
            residual_charge = float(solute_net_charge_e) + applied_charge
            ion_counts[species] = ion_counts.get(species, 0) + count
            ion_templates[species] = custom_templates.get(species) or ion_pdb(species)
            neutralization.update(
                {
                    "counterion": species,
                    "counterion_count": count,
                    "counterion_charge_e": charge_e,
                    "applied_charge_e": applied_charge,
                    "residual_charge_e": residual_charge,
                }
            )
            if abs(residual_charge) > 1.0e-8:
                warnings.append(
                    f"neutralization with {species} leaves residual charge {residual_charge:+.3f} e"
                )

    achieved_salt_molar = (
        _achieved_molarity(resolved_box, formula_units)
        if resolved_salt is not None and formula_units > 0
        else None
    )
    return {
        "box_size": list(resolved_box),
        "box_volume_angstrom3": _box_volume_angstrom3(resolved_box),
        "box_volume_liter": _box_volume_liter(resolved_box),
        "solvent_model": solvent_model,
        "salt": None
        if resolved_salt is None
        else {
            "name": resolved_salt.name,
            "formula": resolved_salt.formula,
            "species": dict(resolved_salt.species),
            "molar": salt_molar,
            "formula_units": formula_units,
            "achieved_molar": achieved_salt_molar,
        },
        "ion_counts": ion_counts,
        "salt_ion_counts": salt_ion_counts,
        "water_count": resolved_water_count,
        "neutralization": neutralization,
        "templates": {
            "solvent": water_pdb(solvent_model),
            "ions": ion_templates,
        },
        "warnings": warnings,
        "custom_ions": custom_templates,
    }


def _native_resolve_payload(
    box_size: float | Sequence[float],
    *,
    solvent_model: str,
    salt: str | Mapping[str, Any] | None,
    salt_molar: float | None,
    water_count: int | None,
    occupied_volume_angstrom3: float,
    custom_ions: Sequence[Mapping[str, Any]] | None,
    custom_salts: Sequence[Mapping[str, Any]] | None,
    neutralize: bool,
    solute_net_charge_e: float | None,
    neutralize_with: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "box_size_angstrom": (
            float(box_size)
            if isinstance(box_size, (int, float))
            else list(_normalize_box_size(box_size))
        ),
        "solvent_model": solvent_model,
        "occupied_volume_angstrom3": occupied_volume_angstrom3,
        "catalog": {
            "ions": [dict(item) for item in custom_ions or []],
            "salts": [dict(item) for item in custom_salts or []],
        },
    }
    if water_count is not None:
        payload["water_count"] = int(water_count)
    if salt is not None:
        if isinstance(salt, Mapping):
            species = dict(salt.get("species", {})) if salt.get("species") else None
            formula = salt.get("formula")
            name = salt.get("name")
            if species:
                salt_payload = {"species": species}
            elif formula is not None:
                salt_payload = {"formula": formula}
            elif name is not None:
                salt_payload = {"name": name}
            else:
                salt_payload = {}
        else:
            salt_payload = {"name": str(salt)}
        if salt_molar is not None:
            salt_payload["molar"] = float(salt_molar)
        payload["salt"] = salt_payload
    if neutralize_with is not None:
        payload["neutralize"] = {"enabled": bool(neutralize), "with": neutralize_with}
    else:
        payload["neutralize"] = bool(neutralize)
    if solute_net_charge_e is not None:
        payload["solute_net_charge_e"] = float(solute_net_charge_e)
    return payload


def resolve_chemistry(
    box_size: float | Sequence[float],
    *,
    solvent_model: str = "tip3p",
    salt: str | Mapping[str, Any] | None = None,
    salt_molar: float | None = None,
    water_count: int | None = None,
    occupied_volume_angstrom3: float = 0.0,
    custom_ions: Sequence[Mapping[str, Any]] | None = None,
    custom_salts: Sequence[Mapping[str, Any]] | None = None,
    neutralize: bool = False,
    solute_net_charge_e: float | None = None,
    neutralize_with: str | None = None,
) -> dict[str, Any]:
    if traj_py is not None and hasattr(traj_py, "pack_resolve_chemistry"):
        payload = _native_resolve_payload(
            box_size,
            solvent_model=solvent_model,
            salt=salt,
            salt_molar=salt_molar,
            water_count=water_count,
            occupied_volume_angstrom3=occupied_volume_angstrom3,
            custom_ions=custom_ions,
            custom_salts=custom_salts,
            neutralize=neutralize,
            solute_net_charge_e=solute_net_charge_e,
            neutralize_with=neutralize_with,
        )
        return dict(traj_py.pack_resolve_chemistry(json.dumps(payload)))
    return _resolve_chemistry_fallback(
        box_size,
        solvent_model=solvent_model,
        salt=salt,
        salt_molar=salt_molar,
        water_count=water_count,
        occupied_volume_angstrom3=occupied_volume_angstrom3,
        custom_ions=custom_ions,
        custom_salts=custom_salts,
        neutralize=neutralize,
        solute_net_charge_e=solute_net_charge_e,
        neutralize_with=neutralize_with,
    )


def solution_recipe(
    box_size: float | Sequence[float],
    *,
    solvent_model: str = "tip3p",
    salt: str | Mapping[str, Any] | None = None,
    salt_molar: float | None = None,
    water_count: int | None = None,
    occupied_volume_angstrom3: float = 0.0,
    custom_ions: Sequence[Mapping[str, Any]] | None = None,
    custom_salts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return resolve_chemistry(
        box_size,
        solvent_model=solvent_model,
        salt=salt,
        salt_molar=salt_molar,
        water_count=water_count,
        occupied_volume_angstrom3=occupied_volume_angstrom3,
        custom_ions=custom_ions,
        custom_salts=custom_salts,
    )


def solution_pack_config(
    *,
    solute_path: str,
    box_size: float | Sequence[float],
    output_path: str | None = None,
    solvent_model: str = "tip3p",
    salt: str | Mapping[str, Any] | None = None,
    salt_molar: float | None = None,
    water_count: int | None = None,
    occupied_volume_angstrom3: float = 0.0,
    custom_ions: Sequence[Mapping[str, Any]] | None = None,
    custom_salts: Sequence[Mapping[str, Any]] | None = None,
    min_distance: float = 2.0,
    neutralize: bool = False,
    solute_net_charge_e: float | None = None,
    neutralize_with: str | None = None,
) -> PackConfig:
    recipe = resolve_chemistry(
        box_size,
        solvent_model=solvent_model,
        salt=salt,
        salt_molar=salt_molar,
        water_count=water_count,
        occupied_volume_angstrom3=occupied_volume_angstrom3,
        custom_ions=custom_ions,
        custom_salts=custom_salts,
        neutralize=neutralize,
        solute_net_charge_e=solute_net_charge_e,
        neutralize_with=neutralize_with,
    )
    resolved_box = tuple(recipe["box_size"])
    center = tuple(side / 2.0 for side in resolved_box)
    ion_templates = recipe["templates"]["ions"]
    structures = [
        Structure(
            solute_path,
            count=1,
            fixed=True,
            rotate=False,
            center=True,
            positions=[center],
            resnumbers=3,
        )
    ]
    if recipe["water_count"] > 0:
        structures.append(
            Structure(water_pdb(solvent_model), count=int(recipe["water_count"]), resnumbers=3)
        )
    for species, count in recipe["ion_counts"].items():
        template = ion_templates.get(species) or ion_pdb(species)
        structures.append(
            Structure(
                template,
                count=int(count),
                rotate=False,
                resnumbers=3,
                name=species,
            )
        )
    output = None
    if output_path is not None:
        suffix = Path(output_path).suffix.lower().lstrip(".") or "pdb"
        output = OutputSpec(output_path, suffix)
    return PackConfig(
        structures=structures,
        box=Box(resolved_box, shape="orthorhombic"),
        min_distance=min_distance,
        pbc=True,
        output=output,
    )


def ion_parameterization(species: str) -> dict[str, Any]:
    metadata = ion_metadata(species)
    return dict(metadata.get("parameterization", {}))
