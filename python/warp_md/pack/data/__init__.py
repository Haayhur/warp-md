from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from importlib import resources

_WATER_MODELS = {
    "spce": "spce.pdb",
    "spc/e": "spce.pdb",  # alias
    "tip3p": "tip3p.pdb",
    "tip4pew": "tip4pew.pdb",
    "tip4p-ew": "tip4pew.pdb",  # alias
    "tip5p": "tip5p.pdb",
}

_ION_REGISTRY_ENV = "WARP_MD_ION_REGISTRY"
_SALT_REGISTRY_ENV = "WARP_MD_SALT_REGISTRY"


def _get_data_path(filename: str) -> Path:
    """
    Get the path to a data file, with fallback for different installation methods.

    Tries multiple strategies:
    1. importlib.resources (for proper wheel installations)
    2. __file__-based path (for development installs and some wheel installations)
    """
    # Try importlib.resources first (standard way)
    try:
        with resources.as_file(resources.files(__package__).joinpath(filename)) as path:
            if path.exists():
                return path
    except (TypeError, FileNotFoundError, AttributeError):
        pass

    # Fallback to __file__-based path (works with some maturin builds)
    try:
        here = Path(__file__).resolve().parent
        path = here / filename
        if path.exists():
            return path
    except (OSError, AttributeError):
        pass

    # If nothing works, raise an error with helpful information
    raise FileNotFoundError(
        f"Could not locate data file '{filename}'. "
        f"This usually means the package was not built correctly. "
        f"Please reinstall using: pip install --force-reinstall warp-md"
    )


def _normalize_species_key(species: str) -> str:
    return species.strip().lower()


def _default_ion_registry_path() -> Path:
    return _get_data_path("ions.json")


def _overlay_ion_registry_path() -> Path | None:
    value = os.environ.get(_ION_REGISTRY_ENV)
    return Path(value) if value else None


def _default_salt_registry_path() -> Path:
    return _get_data_path("salts.json")


def _overlay_salt_registry_path() -> Path | None:
    value = os.environ.get(_SALT_REGISTRY_ENV)
    return Path(value) if value else None


def _load_ion_registry_entries(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ions = payload.get("ions")
    if not isinstance(ions, list):
        raise ValueError(f"invalid ion registry at {path}: missing 'ions' list")
    resolved: list[dict] = []
    for entry in ions:
        item = dict(entry)
        template_value = str(item.get("template", "")).strip()
        template = Path(template_value) if template_value else None
        if template is not None and not template.is_absolute():
            item["template"] = str((path.parent / template).resolve())
        resolved.append(item)
    return resolved


def _load_salt_registry_entries(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    salts = payload.get("salts")
    if not isinstance(salts, list):
        raise ValueError(f"invalid salt registry at {path}: missing 'salts' list")
    return [dict(entry) for entry in salts]


def _normalize_salt_name_key(name: str) -> str:
    return name.strip().lower()


def _validate_salt_species_map(
    species: dict,
    ion_lookup: dict[str, dict],
    *,
    label: str,
) -> dict[str, int]:
    if not isinstance(species, dict) or not species:
        raise ValueError(f"{label} must define a non-empty species map")
    canonical: dict[str, int] = {}
    net_charge = 0
    has_positive = False
    has_negative = False
    for raw_name, raw_count in species.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"{label} contains an empty ion species name")
        count = int(raw_count)
        if count <= 0:
            raise ValueError(f"{label} species counts must be > 0")
        key = _normalize_species_key(name)
        if key not in ion_lookup:
            raise ValueError(f"{label} references unknown ion species '{name}'")
        entry = ion_lookup[key]
        charge = int(entry["charge_e"])
        net_charge += charge * count
        has_positive |= charge > 0
        has_negative |= charge < 0
        canonical_name = str(entry["species"])
        canonical[canonical_name] = canonical.get(canonical_name, 0) + count
    if not has_positive or not has_negative:
        raise ValueError(f"{label} must include at least one cation and one anion")
    if net_charge != 0:
        raise ValueError(f"{label} must be charge neutral")
    return canonical


@lru_cache(maxsize=1)
def _salt_registry() -> tuple[dict[str, dict], list[str]]:
    entries = _load_salt_registry_entries(_default_salt_registry_path())
    overlay_path = _overlay_salt_registry_path()
    if overlay_path is not None:
        entries.extend(_load_salt_registry_entries(overlay_path))
    ion_lookup, _ = _ion_registry()

    by_lookup: dict[str, dict] = {}
    canonical_names: list[str] = []
    for raw_entry in entries:
        name = str(raw_entry.get("name", "")).strip()
        if not name:
            raise ValueError("salt registry names cannot be empty")
        entry = {
            **raw_entry,
            "name": name,
            "aliases": [str(alias).strip() for alias in raw_entry.get("aliases", [])],
            "formula": str(raw_entry.get("formula", "")).strip(),
            "species": _validate_salt_species_map(
                dict(raw_entry.get("species", {})),
                ion_lookup,
                label=f"salt registry entry '{name}'",
            ),
        }
        if not entry["formula"]:
            raise ValueError(f"salt registry entry '{name}' is missing formula")
        canonical_names.append(name)
        for alias in [*entry["aliases"], name]:
            key = _normalize_salt_name_key(alias)
            if not key:
                raise ValueError(f"salt registry entry '{name}' has an empty alias")
            existing = by_lookup.get(key)
            if existing and existing["name"] != name:
                raise ValueError(
                    f"salt registry alias '{alias}' conflicts between "
                    f"'{existing['name']}' and '{name}'"
                )
            by_lookup[key] = entry
    return by_lookup, canonical_names


@lru_cache(maxsize=1)
def _ion_registry() -> tuple[dict[str, dict], list[str]]:
    entries = _load_ion_registry_entries(_default_ion_registry_path())
    overlay_path = _overlay_ion_registry_path()
    if overlay_path is not None:
        entries.extend(_load_ion_registry_entries(overlay_path))

    canonical_entries: dict[str, dict] = {}
    for raw_entry in entries:
        species = str(raw_entry.get("species", "")).strip()
        if not species:
            raise ValueError("ion registry species names cannot be empty")
        canonical_entries[_normalize_species_key(species)] = {
            **raw_entry,
            "species": species,
            "aliases": [str(alias).strip() for alias in raw_entry.get("aliases", [])],
            "template": str(raw_entry.get("template", "")).strip(),
            "formula_symbol": str(raw_entry.get("formula_symbol", "")).strip(),
            "charge_e": int(raw_entry.get("charge_e", 0)),
            "mass_amu": float(raw_entry.get("mass_amu", 0.0)),
        }

    by_lookup: dict[str, dict] = {}
    canonical_species: list[str] = []
    for entry in canonical_entries.values():
        if not entry["template"]:
            raise ValueError(f"ion registry entry '{entry['species']}' is missing template")
        if not Path(entry["template"]).exists():
            raise ValueError(
                f"ion registry entry '{entry['species']}' template does not exist: {entry['template']}"
            )
        if not entry["formula_symbol"]:
            raise ValueError(f"ion registry entry '{entry['species']}' is missing formula_symbol")
        if entry["charge_e"] == 0:
            raise ValueError(f"ion registry entry '{entry['species']}' must define non-zero charge_e")
        if entry["mass_amu"] <= 0:
            raise ValueError(f"ion registry entry '{entry['species']}' must define positive mass_amu")
        topology_kind = str(entry.get("topology_kind", "")).strip()
        if topology_kind and topology_kind not in {"single_atom", "polyatomic"}:
            raise ValueError(
                f"ion registry entry '{entry['species']}' has unsupported topology_kind '{topology_kind}'"
            )
        atom_count = entry.get("atom_count")
        if atom_count is not None and int(atom_count) <= 0:
            raise ValueError(f"ion registry entry '{entry['species']}' atom_count must be > 0")
        canonical_species.append(entry["species"])
        for alias in [*entry["aliases"], entry["species"]]:
            key = _normalize_species_key(alias)
            if not key:
                raise ValueError(f"ion registry entry '{entry['species']}' has an empty alias")
            existing = by_lookup.get(key)
            if existing and existing["species"] != entry["species"]:
                raise ValueError(
                    f"ion registry alias '{alias}' conflicts between "
                    f"'{existing['species']}' and '{entry['species']}'"
                )
            by_lookup[key] = entry
    return by_lookup, canonical_species


def water_pdb(model: str) -> str:
    """Return the path to a bundled single-molecule water PDB (spce, tip3p, tip4pew, tip5p, spc/e, tip4p-ew)."""
    key = "".join(ch for ch in model.lower() if ch.isalnum())
    if key not in _WATER_MODELS:
        raise ValueError(f"unknown water model: {model}")
    filename = _WATER_MODELS[key]
    return str(_get_data_path(filename))


def available_water_models() -> list[str]:
    return sorted(_WATER_MODELS.keys())


def ion_pdb(species: str) -> str:
    """Return the path to a bundled or registry-defined single-ion PDB."""
    registry, _ = _ion_registry()
    key = _normalize_species_key(species)
    if key not in registry:
        raise ValueError(f"unknown ion species: {species}")
    return str(Path(registry[key]["template"]))


def available_ion_species() -> list[str]:
    """Return canonical ion species names from the active registry."""
    _, canonical_species = _ion_registry()
    return canonical_species.copy()


def ion_metadata(species: str) -> dict:
    """Return bundled or registry-defined ion metadata by species or alias."""
    registry, _ = _ion_registry()
    key = _normalize_species_key(species)
    if key not in registry:
        raise ValueError(f"unknown ion species: {species}")
    return dict(registry[key])


def salt_recipe(name: str) -> dict:
    """Return bundled or registry-defined salt metadata by name or alias."""
    registry, _ = _salt_registry()
    key = _normalize_salt_name_key(name)
    if key not in registry:
        raise ValueError(f"unknown salt name: {name}")
    return dict(registry[key])


def available_salt_names() -> list[str]:
    """Return canonical salt names from the active registry."""
    _, canonical_names = _salt_registry()
    return canonical_names.copy()
