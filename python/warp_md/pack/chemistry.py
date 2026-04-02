from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import Box, OutputSpec, PackConfig, Structure
from .data import ion_metadata, ion_pdb, salt_recipe, water_pdb

AVOGADRO = 6.022_140_76e23
ANGSTROM3_TO_LITER = 1.0e-27
WATER_MOLARITY = 55.5
DEFAULT_PACKING_FRACTION = 0.80


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


def _resolve_salt(
    salt: str | Mapping[str, Any] | None,
    custom_salts: Sequence[Mapping[str, Any]] | None = None,
) -> ResolvedSalt | None:
    if salt is None:
        return None
    custom_lookup = _normalize_catalog_entries(custom_salts, "name")
    if isinstance(salt, Mapping):
        species = {str(k): int(v) for k, v in dict(salt.get("species", {})).items()}
        if not species:
            raise ValueError("salt mapping requires non-empty species")
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
            species={str(k): int(v) for k, v in dict(entry["species"]).items()},
        )
    entry = salt_recipe(salt)
    return ResolvedSalt(
        name=entry.get("name"),
        formula=entry.get("formula"),
        species={str(k): int(v) for k, v in dict(entry["species"]).items()},
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
    resolved_box = _normalize_box_size(box_size)
    resolved_salt = _resolve_salt(salt, custom_salts)
    formula_units = 0
    ion_counts: dict[str, int] = {}
    if resolved_salt is not None and salt_molar is not None:
        formula_units = estimate_salt_formula_units(resolved_box, salt_molar)
        ion_counts = {
            species: int(count) * formula_units
            for species, count in resolved_salt.species.items()
        }
    resolved_water_count = (
        int(water_count)
        if water_count is not None
        else estimate_water_count(resolved_box, occupied_volume_angstrom3)
    )
    return {
        "box_size": list(resolved_box),
        "solvent_model": solvent_model,
        "salt": None
        if resolved_salt is None
        else {
            "name": resolved_salt.name,
            "formula": resolved_salt.formula,
            "species": dict(resolved_salt.species),
            "molar": salt_molar,
            "formula_units": formula_units,
        },
        "ion_counts": ion_counts,
        "water_count": resolved_water_count,
        "custom_ions": _resolve_ion_templates(custom_ions),
    }


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
) -> PackConfig:
    recipe = solution_recipe(
        box_size,
        solvent_model=solvent_model,
        salt=salt,
        salt_molar=salt_molar,
        water_count=water_count,
        occupied_volume_angstrom3=occupied_volume_angstrom3,
        custom_ions=custom_ions,
        custom_salts=custom_salts,
    )
    resolved_box = tuple(recipe["box_size"])
    center = tuple(side / 2.0 for side in resolved_box)
    ion_templates = recipe["custom_ions"]
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
