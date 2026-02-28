from __future__ import annotations

from importlib import resources

_WATER_MODELS = {
    "spce": "spce.pdb",
    "tip3p": "tip3p.pdb",
    "tip4pew": "tip4pew.pdb",
    "tip5p": "tip5p.pdb",
}


def water_pdb(model: str) -> str:
    """Return the path to a bundled single-molecule water PDB (spce, tip3p, tip4pew, tip5p)."""
    key = "".join(ch for ch in model.lower() if ch.isalnum())
    if key not in _WATER_MODELS:
        raise ValueError(f"unknown water model: {model}")
    return str(resources.files(__package__).joinpath(_WATER_MODELS[key]))


def available_water_models() -> list[str]:
    return sorted(_WATER_MODELS.keys())
