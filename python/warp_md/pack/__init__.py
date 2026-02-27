# Usage:
# from warp_md.pack import PackConfig, Structure, Box, run, export
# from warp_md.pack import PackConfigBuilder  # fluent API

from .config import (
    AtomOverride,
    Box,
    Constraint,
    OutputSpec,
    PackConfig,
    PackResult,
    Structure,
    ValidationError,
)
from .data import available_water_models, water_pdb
from .runner import parse_inp, run, run_inp
from .export import export

# Lazy imports for builder classes to avoid circular import with parent warp_md package
_builder_classes = {}


def __getattr__(name: str):
    """Lazy load builder classes on demand."""
    if name in ("PackConfigBuilder", "StructureBuilder"):
        if not _builder_classes:
            from .builder import PackConfigBuilder, StructureBuilder

            _builder_classes["PackConfigBuilder"] = PackConfigBuilder
            _builder_classes["StructureBuilder"] = StructureBuilder
        return _builder_classes[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Data classes
    "AtomOverride",
    "Box",
    "Constraint",
    "OutputSpec",
    "PackConfig",
    "PackResult",
    "Structure",
    # Fluent builders (lazy loaded)
    "PackConfigBuilder",
    "StructureBuilder",
    # Exception
    "ValidationError",
    # Functions
    "run",
    "run_inp",
    "parse_inp",
    "export",
    # Utilities
    "water_pdb",
    "available_water_models",
]
