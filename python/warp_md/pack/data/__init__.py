from __future__ import annotations

import os
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

_ION_MODELS = {
    "na+": "na.pdb",
    "cl-": "cl.pdb",
    "k+": "k.pdb",
}


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
    """Return the path to a bundled single-ion PDB (Na+, Cl-, K+)."""
    key = species.strip().lower()
    if key not in _ION_MODELS:
        raise ValueError(f"unknown ion species: {species}")
    filename = _ION_MODELS[key]
    return str(_get_data_path(filename))


def available_ion_species() -> list[str]:
    return sorted(_ION_MODELS.keys())
