"""Pack configuration data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .errors import ValidationError


@dataclass
class Constraint:
    mode: str
    shape: str
    min: Optional[Tuple[float, float, float]] = None
    max: Optional[Tuple[float, float, float]] = None
    center: Optional[Tuple[float, float, float]] = None
    radius: Optional[float] = None
    base: Optional[Tuple[float, float, float]] = None
    axis: Optional[Tuple[float, float, float]] = None
    height: Optional[float] = None
    side: Optional[float] = None
    radii: Optional[Tuple[float, float, float]] = None
    sigma: Optional[Tuple[float, float]] = None
    z0: Optional[float] = None
    amplitude: Optional[float] = None
    point: Optional[Tuple[float, float, float]] = None
    normal: Optional[Tuple[float, float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"mode": self.mode, "shape": self.shape}
        if self.min is not None:
            out["min"] = list(self.min)
        if self.max is not None:
            out["max"] = list(self.max)
        if self.center is not None:
            out["center"] = list(self.center)
        if self.radius is not None:
            out["radius"] = float(self.radius)
        if self.base is not None:
            out["base"] = list(self.base)
        if self.axis is not None:
            out["axis"] = list(self.axis)
        if self.height is not None:
            out["height"] = float(self.height)
        if self.side is not None:
            out["side"] = float(self.side)
        if self.radii is not None:
            out["radii"] = list(self.radii)
        if self.sigma is not None:
            out["sigma"] = list(self.sigma)
        if self.z0 is not None:
            out["z0"] = float(self.z0)
        if self.amplitude is not None:
            out["amplitude"] = float(self.amplitude)
        if self.point is not None:
            out["point"] = list(self.point)
        if self.normal is not None:
            out["normal"] = list(self.normal)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Constraint":
        return cls(
            mode=data["mode"],
            shape=data["shape"],
            min=tuple(data["min"]) if "min" in data else None,
            max=tuple(data["max"]) if "max" in data else None,
            center=tuple(data["center"]) if "center" in data else None,
            radius=data.get("radius"),
            base=tuple(data["base"]) if "base" in data else None,
            axis=tuple(data["axis"]) if "axis" in data else None,
            height=data.get("height"),
            side=data.get("side"),
            radii=tuple(data["radii"]) if "radii" in data else None,
            sigma=tuple(data["sigma"]) if "sigma" in data else None,
            z0=data.get("z0"),
            amplitude=data.get("amplitude"),
            point=tuple(data["point"]) if "point" in data else None,
            normal=tuple(data["normal"]) if "normal" in data else None,
        )

    def validate(self) -> None:
        valid_modes = ("inside", "outside", "fixed", "over", "below")
        if self.mode not in valid_modes:
            raise ValidationError(f"Invalid mode '{self.mode}'. Must be one of: {valid_modes}")

        valid_shapes = (
            "box",
            "sphere",
            "cylinder",
            "cube",
            "ellipsoid",
            "gaussian",
            "plane",
        )
        if self.shape not in valid_shapes:
            raise ValidationError(
                f"Invalid shape '{self.shape}'. Must be one of: {valid_shapes}"
            )

        if self.shape == "box":
            if self.min is None or self.max is None:
                raise ValidationError("Box constraint requires 'min' and 'max' coordinates")
        elif self.shape == "sphere":
            if self.center is None or self.radius is None:
                raise ValidationError("Sphere constraint requires 'center' and 'radius'")
            if self.radius <= 0:
                raise ValidationError("Sphere radius must be positive")
        elif self.shape == "cylinder":
            if self.base is None or self.axis is None or self.radius is None or self.height is None:
                raise ValidationError(
                    "Cylinder constraint requires 'base', 'axis', 'radius', and 'height'"
                )
        elif self.shape == "cube":
            if self.center is None or self.side is None:
                raise ValidationError("Cube constraint requires 'center' and 'side'")
        elif self.shape == "ellipsoid":
            if self.center is None or self.radii is None:
                raise ValidationError("Ellipsoid constraint requires 'center' and 'radii'")
        elif self.shape == "plane":
            if self.point is None or self.normal is None:
                raise ValidationError("Plane constraint requires 'point' and 'normal'")


@dataclass
class AtomOverride:
    indices: List[int]
    radius: Optional[float] = None
    fscale: Optional[float] = None
    short_radius: Optional[float] = None
    short_radius_scale: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"indices": [int(i) for i in self.indices]}
        if self.radius is not None:
            out["radius"] = float(self.radius)
        if self.fscale is not None:
            out["fscale"] = float(self.fscale)
        if self.short_radius is not None:
            out["short_radius"] = float(self.short_radius)
        if self.short_radius_scale is not None:
            out["short_radius_scale"] = float(self.short_radius_scale)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtomOverride":
        return cls(
            indices=list(data["indices"]),
            radius=data.get("radius"),
            fscale=data.get("fscale"),
            short_radius=data.get("short_radius"),
            short_radius_scale=data.get("short_radius_scale"),
        )

    def validate(self) -> None:
        if not self.indices:
            raise ValidationError("AtomOverride requires at least one index")
        if any(i < 0 for i in self.indices):
            raise ValidationError("Atom indices must be non-negative")
        if self.radius is not None and self.radius <= 0:
            raise ValidationError("Atom radius must be positive")


@dataclass
class Structure:
    path: str
    count: int = 1
    name: Optional[str] = None
    topology: Optional[str] = None
    restart_from: Optional[str] = None
    restart_to: Optional[str] = None
    fixed_eulers: Optional[List[Tuple[float, float, float]]] = None
    chain: Optional[str] = None
    changechains: bool = False
    segid: Optional[str] = None
    filetype: Optional[str] = None
    connect: bool = True
    rotate: bool = True
    fixed: bool = False
    positions: Optional[List[Tuple[float, float, float]]] = None
    translate: Optional[Tuple[float, float, float]] = None
    center: bool = True
    min_distance: Optional[float] = None
    resnumbers: Optional[int] = None
    maxmove: Optional[int] = None
    nloop: Optional[int] = None
    nloop0: Optional[int] = None
    constraints: Optional[List[Constraint]] = None
    radius: Optional[float] = None
    fscale: Optional[float] = None
    short_radius: Optional[float] = None
    short_radius_scale: Optional[float] = None
    atom_overrides: Optional[List[AtomOverride]] = None
    rot_bounds: Optional[
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    ] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "path": self.path,
            "count": int(self.count),
            "rotate": bool(self.rotate),
            "fixed": bool(self.fixed),
            "center": bool(self.center),
        }
        if self.name:
            out["name"] = self.name
        if self.topology:
            out["topology"] = self.topology
        if self.restart_from:
            out["restart_from"] = self.restart_from
        if self.restart_to:
            out["restart_to"] = self.restart_to
        if self.fixed_eulers is not None:
            out["fixed_eulers"] = [list(e) for e in self.fixed_eulers]
        if self.chain:
            out["chain"] = self.chain
        if self.changechains:
            out["changechains"] = True
        if self.segid:
            out["segid"] = self.segid
        if self.filetype:
            out["filetype"] = self.filetype
        if not self.connect:
            out["connect"] = False
        if self.positions is not None:
            out["positions"] = [list(p) for p in self.positions]
        if self.translate is not None:
            out["translate"] = list(self.translate)
        if self.min_distance is not None:
            out["min_distance"] = float(self.min_distance)
        if self.resnumbers is not None:
            out["resnumbers"] = int(self.resnumbers)
        if self.maxmove is not None:
            out["maxmove"] = int(self.maxmove)
        if self.nloop is not None:
            out["nloop"] = int(self.nloop)
        if self.nloop0 is not None:
            out["nloop0"] = int(self.nloop0)
        if self.constraints:
            out["constraints"] = [c.to_dict() for c in self.constraints]
        if self.radius is not None:
            out["radius"] = float(self.radius)
        if self.fscale is not None:
            out["fscale"] = float(self.fscale)
        if self.short_radius is not None:
            out["short_radius"] = float(self.short_radius)
        if self.short_radius_scale is not None:
            out["short_radius_scale"] = float(self.short_radius_scale)
        if self.atom_overrides:
            out["atom_overrides"] = [o.to_dict() for o in self.atom_overrides]
        if self.rot_bounds is not None:
            out["rot_bounds"] = [list(b) for b in self.rot_bounds]
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Structure":
        constraints = None
        if "constraints" in data:
            constraints = [Constraint.from_dict(c) for c in data["constraints"]]
        atom_overrides = None
        if "atom_overrides" in data:
            atom_overrides = [AtomOverride.from_dict(o) for o in data["atom_overrides"]]
        return cls(
            path=data["path"],
            count=data.get("count", 1),
            name=data.get("name"),
            topology=data.get("topology"),
            restart_from=data.get("restart_from"),
            restart_to=data.get("restart_to"),
            fixed_eulers=[tuple(e) for e in data["fixed_eulers"]]
            if "fixed_eulers" in data
            else None,
            chain=data.get("chain"),
            changechains=data.get("changechains", False),
            segid=data.get("segid"),
            filetype=data.get("filetype"),
            connect=data.get("connect", True),
            rotate=data.get("rotate", True),
            fixed=data.get("fixed", False),
            positions=[tuple(p) for p in data["positions"]]
            if "positions" in data
            else None,
            translate=tuple(data["translate"]) if "translate" in data else None,
            center=data.get("center", True),
            min_distance=data.get("min_distance"),
            resnumbers=data.get("resnumbers"),
            maxmove=data.get("maxmove"),
            nloop=data.get("nloop"),
            nloop0=data.get("nloop0"),
            constraints=constraints,
            radius=data.get("radius"),
            fscale=data.get("fscale"),
            short_radius=data.get("short_radius"),
            short_radius_scale=data.get("short_radius_scale"),
            atom_overrides=atom_overrides,
            rot_bounds=[tuple(b) for b in data["rot_bounds"]]
            if "rot_bounds" in data
            else None,
        )

    def validate(self) -> None:
        if not self.path:
            raise ValidationError("Structure path cannot be empty")
        if self.count < 1:
            raise ValidationError(f"Structure count must be >= 1, got {self.count}")
        if self.min_distance is not None and self.min_distance <= 0:
            raise ValidationError("min_distance must be positive")
        if self.constraints:
            for i, c in enumerate(self.constraints):
                try:
                    c.validate()
                except ValidationError as e:
                    raise ValidationError(f"Constraint {i}: {e}") from e
        if self.atom_overrides:
            for i, o in enumerate(self.atom_overrides):
                try:
                    o.validate()
                except ValidationError as e:
                    raise ValidationError(f"AtomOverride {i}: {e}") from e


@dataclass
class Box:
    size: Tuple[float, float, float]
    shape: str = "orthorhombic"

    def to_dict(self) -> Dict[str, Any]:
        return {"size": list(self.size), "shape": self.shape}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Box":
        return cls(
            size=tuple(data["size"]),
            shape=data.get("shape", "orthorhombic"),
        )

    def validate(self) -> None:
        if len(self.size) != 3:
            raise ValidationError(f"Box size must have 3 dimensions, got {len(self.size)}")
        if any(s <= 0 for s in self.size):
            raise ValidationError("All box dimensions must be positive")
        valid_shapes = ("orthorhombic", "cubic", "triclinic")
        if self.shape not in valid_shapes:
            raise ValidationError(
                f"Invalid box shape '{self.shape}'. Must be one of: {valid_shapes}"
            )


@dataclass
class OutputSpec:
    path: str
    format: str
    scale: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"path": self.path, "format": self.format}
        if self.scale is not None:
            out["scale"] = float(self.scale)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputSpec":
        return cls(
            path=data["path"],
            format=data["format"],
            scale=data.get("scale"),
        )

    def validate(self) -> None:
        if not self.path:
            raise ValidationError("Output path cannot be empty")
        valid_formats = (
            "pdb",
            "xyz",
            "pdbx",
            "cif",
            "mmcif",
            "gro",
            "lammps",
            "lammps-data",
            "lmp",
            "mol2",
            "crd",
        )
        if self.format.lower() not in valid_formats:
            raise ValidationError(
                f"Invalid format '{self.format}'. Must be one of: {valid_formats}"
            )
        if self.scale is not None and self.scale <= 0:
            raise ValidationError("Output scale must be positive")


__all__ = [
    "AtomOverride",
    "Box",
    "Constraint",
    "OutputSpec",
    "Structure",
]
