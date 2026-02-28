from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .io import open_trajectory_auto

_API_IMPORT_ERROR: Optional[Exception]
try:
    from . import (
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
        Trajectory,
        WaterCountPlan,
    )
    _API_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - import guard for help/metadata usage
    BondAngleDistributionPlan = None  # type: ignore[assignment]
    BondLengthDistributionPlan = None  # type: ignore[assignment]
    ChainRgPlan = None  # type: ignore[assignment]
    ConductivityPlan = None  # type: ignore[assignment]
    ContourLengthPlan = None  # type: ignore[assignment]
    DielectricPlan = None  # type: ignore[assignment]
    DipoleAlignmentPlan = None  # type: ignore[assignment]
    EndToEndPlan = None  # type: ignore[assignment]
    EquipartitionPlan = None  # type: ignore[assignment]
    HbondPlan = None  # type: ignore[assignment]
    IonPairCorrelationPlan = None  # type: ignore[assignment]
    MsdPlan = None  # type: ignore[assignment]
    PersistenceLengthPlan = None  # type: ignore[assignment]
    RdfPlan = None  # type: ignore[assignment]
    RgPlan = None  # type: ignore[assignment]
    RmsdPlan = None  # type: ignore[assignment]
    RotAcfPlan = None  # type: ignore[assignment]
    StructureFactorPlan = None  # type: ignore[assignment]
    System = None  # type: ignore[assignment]
    Trajectory = None  # type: ignore[assignment]
    WaterCountPlan = None  # type: ignore[assignment]
    _API_IMPORT_ERROR = exc


def _require_api() -> None:
    if _API_IMPORT_ERROR is not None:
        raise RuntimeError(
            "warp-md Python bindings are unavailable. Run `maturin develop` or install warp-md."
        ) from _API_IMPORT_ERROR


def _load_system(spec: Dict[str, Any]) -> System:
    _require_api()
    path = spec.get("path")
    if not path:
        raise ValueError("system.path is required")
    fmt = spec.get("format")
    if fmt is None:
        fmt = Path(path).suffix.lower().lstrip(".")
    if fmt == "pdb":
        return System.from_pdb(path)
    if fmt == "gro":
        return System.from_gro(path)
    raise ValueError("system.format must be pdb or gro")


def _load_trajectory(spec: Dict[str, Any], system: System) -> Trajectory:
    _require_api()
    path = spec.get("path")
    if not path:
        raise ValueError("trajectory.path is required")
    return open_trajectory_auto(
        path,
        system,
        format=spec.get("format"),
        length_scale=spec.get("length_scale"),
        trajectory_cls=Trajectory,
    )


def _select(system: System, expr: str, label: str):
    if not expr:
        raise ValueError(f"{label} selection is required")
    return system.select(expr)
