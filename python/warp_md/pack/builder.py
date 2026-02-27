# Usage:
# from warp_md.pack import PackConfigBuilder
# cfg = (PackConfigBuilder()
#     .box(100, 100, 100)
#     .add("protein.pdb", count=10)
#         .inside_sphere((50, 50, 50), radius=40)
#         .done()
#     .add(water_pdb("tip3p"), count=1000)
#         .done()
#     .min_distance(2.0)
#     .build())

from __future__ import annotations

from typing import List, Optional, Tuple, Union

from .config import (
    AtomOverride,
    Box,
    Constraint,
    OutputSpec,
    PackConfig,
    Structure,
    ValidationError,
)


class StructureBuilder:
    """Fluent builder for Structure configuration."""

    def __init__(self, parent: "PackConfigBuilder", path: str, count: int = 1):
        self._parent = parent
        self._path = path
        self._count = count
        self._name: Optional[str] = None
        self._topology: Optional[str] = None
        self._chain: Optional[str] = None
        self._segid: Optional[str] = None
        self._rotate: bool = True
        self._fixed: bool = False
        self._center: bool = True
        self._connect: bool = True
        self._min_distance: Optional[float] = None
        self._positions: Optional[List[Tuple[float, float, float]]] = None
        self._translate: Optional[Tuple[float, float, float]] = None
        self._fixed_eulers: Optional[List[Tuple[float, float, float]]] = None
        self._constraints: List[Constraint] = []
        self._atom_overrides: List[AtomOverride] = []
        self._radius: Optional[float] = None
        self._fscale: Optional[float] = None

    # --- Constraint builders ---

    def inside_sphere(
        self,
        center: Tuple[float, float, float],
        radius: float,
    ) -> "StructureBuilder":
        """Add a spherical constraint to keep molecules inside."""
        self._constraints.append(
            Constraint(mode="inside", shape="sphere", center=center, radius=radius)
        )
        return self

    def outside_sphere(
        self,
        center: Tuple[float, float, float],
        radius: float,
    ) -> "StructureBuilder":
        """Add a spherical constraint to keep molecules outside."""
        self._constraints.append(
            Constraint(mode="outside", shape="sphere", center=center, radius=radius)
        )
        return self

    def inside_box(
        self,
        min: Tuple[float, float, float],
        max: Tuple[float, float, float],
    ) -> "StructureBuilder":
        """Add a box constraint to keep molecules inside."""
        self._constraints.append(
            Constraint(mode="inside", shape="box", min=min, max=max)
        )
        return self

    def outside_box(
        self,
        min: Tuple[float, float, float],
        max: Tuple[float, float, float],
    ) -> "StructureBuilder":
        """Add a box constraint to keep molecules outside."""
        self._constraints.append(
            Constraint(mode="outside", shape="box", min=min, max=max)
        )
        return self

    def inside_cylinder(
        self,
        base: Tuple[float, float, float],
        axis: Tuple[float, float, float],
        radius: float,
        height: float,
    ) -> "StructureBuilder":
        """Add a cylindrical constraint to keep molecules inside."""
        self._constraints.append(
            Constraint(
                mode="inside",
                shape="cylinder",
                base=base,
                axis=axis,
                radius=radius,
                height=height,
            )
        )
        return self

    def inside_ellipsoid(
        self,
        center: Tuple[float, float, float],
        radii: Tuple[float, float, float],
    ) -> "StructureBuilder":
        """Add an ellipsoid constraint to keep molecules inside."""
        self._constraints.append(
            Constraint(mode="inside", shape="ellipsoid", center=center, radii=radii)
        )
        return self

    def constraint(self, constraint: Constraint) -> "StructureBuilder":
        """Add a custom constraint."""
        self._constraints.append(constraint)
        return self

    # --- Structure properties ---

    def name(self, name: str) -> "StructureBuilder":
        """Set structure name."""
        self._name = name
        return self

    def topology(self, path: str) -> "StructureBuilder":
        """Set topology file path."""
        self._topology = path
        return self

    def chain(self, chain_id: str) -> "StructureBuilder":
        """Set chain ID."""
        self._chain = chain_id
        return self

    def segid(self, segid: str) -> "StructureBuilder":
        """Set segment ID."""
        self._segid = segid
        return self

    def rotate(self, enabled: bool = True) -> "StructureBuilder":
        """Enable or disable random rotation."""
        self._rotate = enabled
        return self

    def no_rotate(self) -> "StructureBuilder":
        """Disable random rotation (alias for rotate(False))."""
        self._rotate = False
        return self

    def fixed(self) -> "StructureBuilder":
        """Mark structure as fixed (no movement during packing)."""
        self._fixed = True
        return self

    def center(self, enabled: bool = True) -> "StructureBuilder":
        """Enable or disable centering."""
        self._center = enabled
        return self

    def no_center(self) -> "StructureBuilder":
        """Disable centering (alias for center(False))."""
        self._center = False
        return self

    def connect(self, enabled: bool = True) -> "StructureBuilder":
        """Enable or disable bond connectivity."""
        self._connect = enabled
        return self

    def no_connect(self) -> "StructureBuilder":
        """Disable bond connectivity (alias for connect(False))."""
        self._connect = False
        return self

    def min_distance(self, distance: float) -> "StructureBuilder":
        """Set minimum distance for this structure."""
        self._min_distance = distance
        return self

    def fixed_at(
        self, positions: List[Tuple[float, float, float]]
    ) -> "StructureBuilder":
        """Fix molecules at specific positions."""
        self._positions = positions
        self._fixed = True
        return self

    def translate(self, offset: Tuple[float, float, float]) -> "StructureBuilder":
        """Apply translation offset."""
        self._translate = offset
        return self

    def fixed_orientations(
        self, eulers: List[Tuple[float, float, float]]
    ) -> "StructureBuilder":
        """Fix Euler angles for molecules."""
        self._fixed_eulers = eulers
        return self

    def radius(self, r: float) -> "StructureBuilder":
        """Set atomic radius for packing."""
        self._radius = r
        return self

    def fscale(self, scale: float) -> "StructureBuilder":
        """Set force scale factor."""
        self._fscale = scale
        return self

    # --- Build ---

    def done(self) -> "PackConfigBuilder":
        """Finish building this structure and return to pack config builder."""
        structure = Structure(
            path=self._path,
            count=self._count,
            name=self._name,
            topology=self._topology,
            chain=self._chain,
            segid=self._segid,
            rotate=self._rotate,
            fixed=self._fixed,
            center=self._center,
            connect=self._connect,
            min_distance=self._min_distance,
            positions=self._positions,
            translate=self._translate,
            fixed_eulers=self._fixed_eulers,
            constraints=self._constraints if self._constraints else None,
            atom_overrides=self._atom_overrides if self._atom_overrides else None,
            radius=self._radius,
            fscale=self._fscale,
        )
        self._parent._structures.append(structure)
        return self._parent

    def build(self, validate: bool = True) -> PackConfig:
        """Convenience method: finish this structure and build the config."""
        return self.done().build(validate=validate)


class PackConfigBuilder:
    """Fluent builder for PackConfig."""

    def __init__(self):
        self._structures: List[Structure] = []
        self._box: Optional[Box] = None
        self._seed: int = 0
        self._max_attempts: int = 10000
        self._min_distance: float = 2.0
        self._pbc: bool = False
        self._output: Optional[OutputSpec] = None
        self._nloop: Optional[int] = None
        self._maxit: Optional[int] = None
        self._avoid_overlap: bool = True
        self._add_box_sides: bool = False

    # --- Box configuration ---

    def box(
        self,
        x: float,
        y: Optional[float] = None,
        z: Optional[float] = None,
        shape: str = "orthorhombic",
    ) -> "PackConfigBuilder":
        """Set simulation box dimensions.
        
        Can be called as:
        - box(size) for cubic box
        - box(x, y, z) for orthorhombic box
        """
        if y is None and z is None:
            # Cubic box
            self._box = Box(size=(x, x, x), shape=shape)
        elif y is not None and z is not None:
            self._box = Box(size=(x, y, z), shape=shape)
        else:
            raise ValidationError("box() requires 1 or 3 dimensions")
        return self

    def cubic_box(self, side: float) -> "PackConfigBuilder":
        """Set a cubic simulation box."""
        self._box = Box(size=(side, side, side), shape="cubic")
        return self

    # --- Structure builders ---

    def add(self, path: str, count: int = 1) -> StructureBuilder:
        """Add a structure to pack.
        
        Returns a StructureBuilder for fluent constraint/property configuration.
        Call .done() to return to PackConfigBuilder.
        """
        return StructureBuilder(self, path, count)

    def add_structure(self, structure: Structure) -> "PackConfigBuilder":
        """Add a pre-built Structure object."""
        self._structures.append(structure)
        return self

    # --- Pack configuration ---

    def seed(self, seed: int) -> "PackConfigBuilder":
        """Set random seed for reproducibility."""
        self._seed = seed
        return self

    def min_distance(self, distance: float) -> "PackConfigBuilder":
        """Set global minimum distance between molecules."""
        self._min_distance = distance
        return self

    def max_attempts(self, attempts: int) -> "PackConfigBuilder":
        """Set maximum packing attempts."""
        self._max_attempts = attempts
        return self

    def nloop(self, n: int) -> "PackConfigBuilder":
        """Set number of optimization loops."""
        self._nloop = n
        return self

    def maxit(self, n: int) -> "PackConfigBuilder":
        """Set maximum iterations."""
        self._maxit = n
        return self

    def pbc(self, enabled: bool = True) -> "PackConfigBuilder":
        """Enable/disable periodic boundary conditions."""
        self._pbc = enabled
        return self

    def no_overlap_check(self) -> "PackConfigBuilder":
        """Disable overlap checking."""
        self._avoid_overlap = False
        return self

    def add_box_sides(self, enabled: bool = True) -> "PackConfigBuilder":
        """Add box information to output."""
        self._add_box_sides = enabled
        return self

    # --- Output configuration ---

    def output(
        self, path: str, format: str, scale: Optional[float] = None
    ) -> "PackConfigBuilder":
        """Configure output file."""
        self._output = OutputSpec(path=path, format=format, scale=scale)
        return self

    # --- Build ---

    def build(self, validate: bool = True) -> PackConfig:
        """Build the PackConfig.
        
        Args:
            validate: If True, validate configuration before returning.
        
        Returns:
            Configured PackConfig ready for packing.
        
        Raises:
            ValidationError: If validate=True and configuration is invalid.
        """
        if self._box is None:
            raise ValidationError("Box dimensions must be specified. Call .box() first.")
        if validate and not self._structures:
            raise ValidationError("At least one structure must be added. Call .add() first.")

        config = PackConfig(
            structures=self._structures,
            box=self._box,
            seed=self._seed,
            max_attempts=self._max_attempts,
            min_distance=self._min_distance,
            pbc=self._pbc,
            nloop=self._nloop,
            maxit=self._maxit,
            avoid_overlap=self._avoid_overlap,
            add_box_sides=self._add_box_sides,
            output=self._output,
        )

        if validate:
            config.validate()

        return config
