"""Pack configuration entrypoint.

Re-exports the core pack data structures for backwards compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .errors import ValidationError
from .result import PackResult
from .specs import AtomOverride, Box, Constraint, OutputSpec, Structure


@dataclass
class PackConfig:
    structures: List[Structure]
    box: Box
    seed: int = 0
    max_attempts: int = 10000
    min_distance: float = 2.0
    filetype: Optional[str] = None
    add_box_sides: bool = False
    add_box_sides_fix: Optional[float] = None
    add_amber_ter: bool = False
    amber_ter_preserve: bool = False
    hexadecimal_indices: bool = False
    ignore_conect: bool = False
    non_standard_conect: bool = True
    pbc: bool = False
    pbc_min: Optional[Tuple[float, float, float]] = None
    pbc_max: Optional[Tuple[float, float, float]] = None
    maxit: Optional[int] = None
    nloop: Optional[int] = None
    nloop0: Optional[int] = None
    avoid_overlap: bool = True
    packall: bool = False
    check: bool = False
    sidemax: Optional[float] = None
    discale: Optional[float] = None
    precision: Optional[float] = None
    chkgrad: bool = False
    iprint1: Optional[int] = None
    iprint2: Optional[int] = None
    use_short_tol: bool = False
    short_tol_dist: Optional[float] = None
    short_tol_scale: Optional[float] = None
    movefrac: Optional[float] = None
    movebadrandom: bool = False
    disable_movebad: bool = False
    maxmove: Optional[int] = None
    randominitialpoint: bool = False
    fbins: Optional[float] = None
    writeout: Optional[float] = None
    writebad: bool = False
    restart_from: Optional[str] = None
    restart_to: Optional[str] = None
    relax_steps: Optional[int] = None
    relax_step: Optional[float] = None
    gencan_maxit: Optional[int] = None
    gencan_step: Optional[float] = None
    write_crd: Optional[str] = None
    output: Optional[OutputSpec] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "box": self.box.to_dict(),
            "structures": [s.to_dict() for s in self.structures],
            "seed": int(self.seed),
            "max_attempts": int(self.max_attempts),
            "min_distance": float(self.min_distance),
            "pbc": bool(self.pbc),
        }
        if self.filetype:
            out["filetype"] = self.filetype
        if self.add_box_sides:
            out["add_box_sides"] = True
        if self.add_box_sides_fix is not None:
            out["add_box_sides_fix"] = float(self.add_box_sides_fix)
        if self.add_amber_ter:
            out["add_amber_ter"] = True
        if self.amber_ter_preserve:
            out["amber_ter_preserve"] = True
        if self.hexadecimal_indices:
            out["hexadecimal_indices"] = True
        if self.ignore_conect:
            out["ignore_conect"] = True
        if not self.non_standard_conect:
            out["non_standard_conect"] = False
        if self.pbc_min is not None:
            out["pbc_min"] = list(self.pbc_min)
        if self.pbc_max is not None:
            out["pbc_max"] = list(self.pbc_max)
        if self.maxit is not None:
            out["maxit"] = int(self.maxit)
        if self.nloop is not None:
            out["nloop"] = int(self.nloop)
        if self.nloop0 is not None:
            out["nloop0"] = int(self.nloop0)
        if not self.avoid_overlap:
            out["avoid_overlap"] = False
        if self.packall:
            out["packall"] = True
        if self.check:
            out["check"] = True
        if self.sidemax is not None:
            out["sidemax"] = float(self.sidemax)
        if self.discale is not None:
            out["discale"] = float(self.discale)
        if self.precision is not None:
            out["precision"] = float(self.precision)
        if self.chkgrad:
            out["chkgrad"] = True
        if self.iprint1 is not None:
            out["iprint1"] = int(self.iprint1)
        if self.iprint2 is not None:
            out["iprint2"] = int(self.iprint2)
        if self.use_short_tol:
            out["use_short_tol"] = True
        if self.short_tol_dist is not None:
            out["short_tol_dist"] = float(self.short_tol_dist)
        if self.short_tol_scale is not None:
            out["short_tol_scale"] = float(self.short_tol_scale)
        if self.movefrac is not None:
            out["movefrac"] = float(self.movefrac)
        if self.movebadrandom:
            out["movebadrandom"] = True
        if self.disable_movebad:
            out["disable_movebad"] = True
        if self.maxmove is not None:
            out["maxmove"] = int(self.maxmove)
        if self.randominitialpoint:
            out["randominitialpoint"] = True
        if self.fbins is not None:
            out["fbins"] = float(self.fbins)
        if self.writeout is not None:
            out["writeout"] = float(self.writeout)
        if self.writebad:
            out["writebad"] = True
        if self.restart_from:
            out["restart_from"] = self.restart_from
        if self.restart_to:
            out["restart_to"] = self.restart_to
        if self.relax_steps is not None:
            out["relax_steps"] = int(self.relax_steps)
        if self.relax_step is not None:
            out["relax_step"] = float(self.relax_step)
        if self.gencan_maxit is not None:
            out["gencan_maxit"] = int(self.gencan_maxit)
        if self.gencan_step is not None:
            out["gencan_step"] = float(self.gencan_step)
        if self.write_crd:
            out["write_crd"] = self.write_crd
        if self.output is not None:
            out["output"] = self.output.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PackConfig":
        structures = [Structure.from_dict(s) for s in data["structures"]]
        box = Box.from_dict(data["box"])
        output = OutputSpec.from_dict(data["output"]) if "output" in data else None
        return cls(
            structures=structures,
            box=box,
            seed=data.get("seed", 0),
            max_attempts=data.get("max_attempts", 10000),
            min_distance=data.get("min_distance", 2.0),
            filetype=data.get("filetype"),
            add_box_sides=data.get("add_box_sides", False),
            add_box_sides_fix=data.get("add_box_sides_fix"),
            add_amber_ter=data.get("add_amber_ter", False),
            amber_ter_preserve=data.get("amber_ter_preserve", False),
            hexadecimal_indices=data.get("hexadecimal_indices", False),
            ignore_conect=data.get("ignore_conect", False),
            non_standard_conect=data.get("non_standard_conect", True),
            pbc=data.get("pbc", False),
            pbc_min=tuple(data["pbc_min"]) if "pbc_min" in data else None,
            pbc_max=tuple(data["pbc_max"]) if "pbc_max" in data else None,
            maxit=data.get("maxit"),
            nloop=data.get("nloop"),
            nloop0=data.get("nloop0"),
            avoid_overlap=data.get("avoid_overlap", True),
            packall=data.get("packall", False),
            check=data.get("check", False),
            sidemax=data.get("sidemax"),
            discale=data.get("discale"),
            precision=data.get("precision"),
            chkgrad=data.get("chkgrad", False),
            iprint1=data.get("iprint1"),
            iprint2=data.get("iprint2"),
            use_short_tol=data.get("use_short_tol", False),
            short_tol_dist=data.get("short_tol_dist"),
            short_tol_scale=data.get("short_tol_scale"),
            movefrac=data.get("movefrac"),
            movebadrandom=data.get("movebadrandom", False),
            disable_movebad=data.get("disable_movebad", False),
            maxmove=data.get("maxmove"),
            randominitialpoint=data.get("randominitialpoint", False),
            fbins=data.get("fbins"),
            writeout=data.get("writeout"),
            writebad=data.get("writebad", False),
            restart_from=data.get("restart_from"),
            restart_to=data.get("restart_to"),
            relax_steps=data.get("relax_steps"),
            relax_step=data.get("relax_step"),
            gencan_maxit=data.get("gencan_maxit"),
            gencan_step=data.get("gencan_step"),
            write_crd=data.get("write_crd"),
            output=output,
        )

    def validate(self) -> None:
        if not self.structures:
            raise ValidationError("PackConfig requires at least one structure")
        if self.min_distance <= 0:
            raise ValidationError("min_distance must be positive")
        if self.max_attempts < 1:
            raise ValidationError("max_attempts must be >= 1")

        try:
            self.box.validate()
        except ValidationError as e:
            raise ValidationError(f"Box: {e}") from e

        for i, s in enumerate(self.structures):
            try:
                s.validate()
            except ValidationError as e:
                raise ValidationError(f"Structure {i} ('{s.path}'): {e}") from e

        if self.output:
            try:
                self.output.validate()
            except ValidationError as e:
                raise ValidationError(f"Output: {e}") from e


__all__ = [
    "AtomOverride",
    "Box",
    "Constraint",
    "OutputSpec",
    "PackConfig",
    "PackResult",
    "Structure",
    "ValidationError",
]
