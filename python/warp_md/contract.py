"""Analysis contract metadata and validation for agent consumers.

This module provides:
- Complete analysis metadata registry
- Request validation with structured errors
- Plan schema discovery
- Request normalization
- Selection linting
- Capabilities fingerprint
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .agent_schema import (
    AGENT_REQUEST_SCHEMA_VERSION,
    AnalysisName,
    RunRequest,
    validate_run_request,
)


# Semantic field types for machine-readable contracts
SemanticType = Literal[
    "selection",    # Atom selection string
    "mask",         # Atom mask string (same as selection but semantic distinction)
    "path",         # File path
    "integer",      # Integer value
    "float",        # Floating point value
    "boolean",      # True/False flag
    "charges",      # Charge specification (by_atom, by_resname, by_name)
    "vector",       # Vector/tuple of numbers
    "string",       # Generic string
]


# Artifact semantic kinds
ArtifactKind = Literal[
    "timeseries",   # Time-indexed data
    "histogram",    # Binned distribution
    "grid",         # 3D grid data
    "profile",      # 1D profile (density, etc.)
    "table",        # Tabular data
    "artifact",     # Generic artifact
]


@dataclass
class FieldSpec:
    """Metadata for a single analysis field."""
    type: str  # "string", "integer", "float", "boolean", "array"
    semantic_type: SemanticType
    description: str = ""
    default: Any = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    unit: Optional[str] = None
    choices: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "type": self.type,
            "semantic_type": self.semantic_type,
        }
        if self.description:
            d["description"] = self.description
        if self.default is not None:
            d["default"] = self.default
        if self.minimum is not None:
            d["minimum"] = self.minimum
        if self.maximum is not None:
            d["maximum"] = self.maximum
        if self.unit:
            d["unit"] = self.unit
        if self.choices:
            d["choices"] = self.choices
        return d


@dataclass
class ArtifactSpec:
    """Output artifact metadata."""
    kind: ArtifactKind
    format: str  # npz, json, csv, etc.
    fields: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {"kind": self.kind, "format": self.format}
        if self.fields:
            d["fields"] = self.fields
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class AnalysisContract:
    """Complete contract metadata for a single analysis."""
    name: str
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, FieldSpec] = field(default_factory=dict)
    outputs: List[ArtifactSpec] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "aliases": self.aliases,
            "description": self.description,
            "required_fields": self.required_fields,
            "optional_fields": self.optional_fields,
            "fields": {k: v.to_dict() for k, v in self.field_types.items()},
            "outputs": [o.to_dict() for o in self.outputs],
            "tags": self.tags,
            "examples": self.examples,
        }


# Analysis metadata registry
# This is the source of truth for all analysis contracts
ANALYSIS_METADATA: Dict[str, AnalysisContract] = {
    "rg": AnalysisContract(
        name="rg",
        aliases=["radius-of-gyration"],
        description="Radius of gyration - measure of polymer compactness",
        required_fields=["selection"],
        optional_fields=["mass_weighted"],
        field_types={
            "selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Atom selection for Rg calculation",
            ),
            "mass_weighted": FieldSpec(
                type="boolean",
                semantic_type="boolean",
                description="Use mass-weighted coordinates",
                default=False,
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="timeseries",
                format="npz",
                fields=["time_ps", "rg_nm"],
                description="Time series of radius of gyration values",
            ),
        ],
        tags=["structural", "polymer", "protein"],
        examples=[
            {
                "name": "rg",
                "selection": "protein",
                "mass_weighted": True,
            },
        ],
    ),

    "rmsd": AnalysisContract(
        name="rmsd",
        aliases=["root-mean-square-deviation"],
        description="Root mean square deviation from reference structure",
        required_fields=["selection"],
        optional_fields=["reference", "align"],
        field_types={
            "selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Atom selection for RMSD calculation",
            ),
            "reference": FieldSpec(
                type="integer",
                semantic_type="integer",
                description="Reference frame index (0-based)",
                minimum=0,
                default=0,
            ),
            "align": FieldSpec(
                type="boolean",
                semantic_type="boolean",
                description="Align to reference before RMSD calculation",
                default=True,
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="timeseries",
                format="npz",
                fields=["time_ps", "rmsd_nm"],
                description="Time series of RMSD values",
            ),
        ],
        tags=["structural", "protein"],
        examples=[
            {
                "name": "rmsd",
                "selection": "protein and name CA",
                "reference": 0,
                "align": True,
            },
        ],
    ),

    "msd": AnalysisContract(
        name="msd",
        aliases=["mean-square-displacement"],
        description="Mean square displacement - diffusion analysis",
        required_fields=["selection"],
        optional_fields=[
            "group_by", "axis", "length_scale", "frame_decimation",
            "dt_decimation", "time_binning", "lag_mode", "max_lag",
            "memory_budget_bytes", "multi_tau_m", "multi_tau_levels",
        ],
        field_types={
            "selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Atom selection for MSD calculation",
            ),
            "group_by": FieldSpec(
                type="string",
                semantic_type="string",
                description="How to group atoms (resid, molecule, etc.)",
                default="resid",
            ),
            "axis": FieldSpec(
                type="array",
                semantic_type="vector",
                description="3D vector for directional MSD (x, y, z)",
            ),
            "length_scale": FieldSpec(
                type="float",
                semantic_type="float",
                description="Length unit conversion factor (nm per unit)",
                default=1.0,
            ),
            "max_lag": FieldSpec(
                type="integer",
                semantic_type="integer",
                description="Maximum lag time in frames",
                minimum=1,
            ),
            "lag_mode": FieldSpec(
                type="string",
                semantic_type="string",
                description="Lag computation mode",
                choices=["linear", "log", "ring"],
                default="linear",
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="timeseries",
                format="npz",
                fields=["lag_time_ps", "msd_nm2"],
                description="MSD vs lag time",
            ),
        ],
        tags=["dynamic", "diffusion", "transport"],
        examples=[
            {
                "name": "msd",
                "selection": "resname CL and name NA",
            },
        ],
    ),

    "rotacf": AnalysisContract(
        name="rotacf",
        aliases=["rotational-autocorrelation", "rotational-acf"],
        description="Rotational autocorrelation function",
        required_fields=["selection", "orientation"],
        optional_fields=[
            "group_by", "p2_legendre", "length_scale", "frame_decimation",
            "dt_decimation", "time_binning", "lag_mode", "max_lag",
        ],
        field_types={
            "selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Atom selection for rotational ACF",
            ),
            "orientation": FieldSpec(
                type="array",
                semantic_type="vector",
                description="Vector indices for orientation (2 or 3 atoms)",
            ),
            "p2_legendre": FieldSpec(
                type="boolean",
                semantic_type="boolean",
                description="Use second Legendre polynomial",
                default=True,
            ),
            "max_lag": FieldSpec(
                type="integer",
                semantic_type="integer",
                description="Maximum lag time in frames",
                minimum=1,
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="timeseries",
                format="npz",
                fields=["lag_time_ps", "acf"],
                description="Rotational ACF vs lag time",
            ),
        ],
        tags=["dynamic", "rotational"],
        examples=[
            {
                "name": "rotacf",
                "selection": "resname MEOH and name OH",
                "orientation": [0, 1, 2],
            },
        ],
    ),

    "conductivity": AnalysisContract(
        name="conductivity",
        aliases=["electrical-conductivity"],
        description="Electrical conductivity via Einstein relation",
        required_fields=["selection", "charges", "temperature"],
        optional_fields=[
            "group_by", "transference", "length_scale", "frame_decimation",
            "dt_decimation", "time_binning", "lag_mode", "max_lag",
        ],
        field_types={
            "selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Ion selection for conductivity",
            ),
            "charges": FieldSpec(
                type="string",
                semantic_type="charges",
                description="Charge specification method",
                choices=["by_atom", "by_resname", "by_name"],
            ),
            "temperature": FieldSpec(
                type="float",
                semantic_type="float",
                description="Temperature in Kelvin",
                minimum=0,
                unit="K",
            ),
            "transference": FieldSpec(
                type="boolean",
                semantic_type="boolean",
                description="Compute transference numbers",
                default=False,
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="timeseries",
                format="npz",
                fields=["lag_time_ps", "conductivity_S_per_cm"],
                description="Conductivity vs lag time",
            ),
        ],
        tags=["dynamic", "electrostatics", "transport"],
        examples=[
            {
                "name": "conductivity",
                "selection": "resname Na or resname CL",
                "charges": "by_resname",
                "temperature": 300.0,
            },
        ],
    ),

    "dielectric": AnalysisContract(
        name="dielectric",
        aliases=["dielectric-constant"],
        description="Dielectric constant from dipole fluctuations",
        required_fields=["selection", "charges"],
        optional_fields=["group_by", "length_scale", "temperature", "make_whole"],
        field_types={
            "selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Molecule selection for dielectric",
            ),
            "charges": FieldSpec(
                type="string",
                semantic_type="charges",
                description="Charge specification method",
            ),
            "temperature": FieldSpec(
                type="float",
                semantic_type="float",
                description="Simulation temperature in Kelvin",
                default=300.0,
                minimum=0.0,
                unit="K",
            ),
            "make_whole": FieldSpec(
                type="boolean",
                semantic_type="boolean",
                description="Reconstruct grouped molecules across periodic boundaries",
                default=True,
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="artifact",
                format="npz",
                fields=["dielectric_constant", "dipole_moment_debye"],
                description="Dielectric constant and dipole moment",
            ),
        ],
        tags=["electrostatics"],
        examples=[
            {
                "name": "dielectric",
                "selection": "resname SOL",
                "charges": "by_resname",
            },
        ],
    ),

    "dipole_alignment": AnalysisContract(
        name="dipole_alignment",
        aliases=["dipole-alignment"],
        description="Dipole-dipole alignment correlation",
        required_fields=["selection", "charges"],
        optional_fields=["group_by", "length_scale"],
        field_types={
            "selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Molecule selection for dipole alignment",
            ),
            "charges": FieldSpec(
                type="string",
                semantic_type="charges",
                description="Charge specification method",
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="histogram",
                format="npz",
                fields=["cos_theta", "probability"],
                description="Dipole alignment distribution",
            ),
        ],
        tags=["electrostatics"],
        examples=[
            {
                "name": "dipole_alignment",
                "selection": "resname SOL",
                "charges": "by_resname",
            },
        ],
    ),

    "ion_pair_correlation": AnalysisContract(
        name="ion_pair_correlation",
        aliases=["ion-pair-correlation", "ion-pair"],
        description="Ion pair lifetime and correlation analysis",
        required_fields=["selection", "rclust_cat", "rclust_ani"],
        optional_fields=[
            "group_by", "cation_type", "anion_type", "max_cluster",
            "length_scale", "lag_mode", "max_lag",
        ],
        field_types={
            "selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Ion selection",
            ),
            "rclust_cat": FieldSpec(
                type="float",
                semantic_type="float",
                description="Cation clustering cutoff distance",
                minimum=0,
                unit="nm",
            ),
            "rclust_ani": FieldSpec(
                type="float",
                semantic_type="float",
                description="Anion clustering cutoff distance",
                minimum=0,
                unit="nm",
            ),
            "cation_type": FieldSpec(
                type="string",
                semantic_type="string",
                description="Cation residue name",
            ),
            "anion_type": FieldSpec(
                type="string",
                semantic_type="string",
                description="Anion residue name",
            ),
            "max_cluster": FieldSpec(
                type="integer",
                semantic_type="integer",
                description="Maximum cluster size",
                minimum=1,
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="timeseries",
                format="npz",
                fields=["lag_time_ps", "correlation"],
                description="Ion pair correlation function",
            ),
        ],
        tags=["electrostatics", "dynamic"],
        examples=[
            {
                "name": "ion_pair_correlation",
                "selection": "resname Na or resname CL",
                "rclust_cat": 0.35,
                "rclust_ani": 0.35,
            },
        ],
    ),

    "structure_factor": AnalysisContract(
        name="structure_factor",
        aliases=["structure-factor", "sk"],
        description="Static structure factor S(q)",
        required_fields=["selection", "bins", "r_max", "q_bins", "q_max"],
        optional_fields=["pbc", "length_scale"],
        field_types={
            "selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Atom selection for structure factor",
            ),
            "bins": FieldSpec(
                type="integer",
                semantic_type="integer",
                description="Number of real-space bins",
                minimum=1,
            ),
            "r_max": FieldSpec(
                type="float",
                semantic_type="float",
                description="Maximum real-space distance",
                minimum=0,
                unit="nm",
            ),
            "q_bins": FieldSpec(
                type="integer",
                semantic_type="integer",
                description="Number of q-space bins",
                minimum=1,
            ),
            "q_max": FieldSpec(
                type="float",
                semantic_type="float",
                description="Maximum q value",
                minimum=0,
            ),
            "pbc": FieldSpec(
                type="boolean",
                semantic_type="boolean",
                description="Apply periodic boundary conditions",
                default=True,
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="histogram",
                format="npz",
                fields=["q_inv_nm", "structure_factor"],
                description="Structure factor vs q",
            ),
        ],
        tags=["structural"],
        examples=[
            {
                "name": "structure_factor",
                "selection": "resname SOL",
                "bins": 200,
                "r_max": 1.0,
                "q_bins": 200,
                "q_max": 30.0,
            },
        ],
    ),

    "water_count": AnalysisContract(
        name="water_count",
        aliases=["water-count"],
        description="Water molecule count in spatial regions",
        required_fields=["water_selection", "center_selection", "box_unit", "region_size"],
        optional_fields=["shift", "length_scale"],
        field_types={
            "water_selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Water molecule selection",
            ),
            "center_selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Center point selection",
            ),
            "box_unit": FieldSpec(
                type="array",
                semantic_type="vector",
                description="Box dimensions (x, y, z)",
            ),
            "region_size": FieldSpec(
                type="array",
                semantic_type="vector",
                description="Region dimensions (x, y, z)",
            ),
            "shift": FieldSpec(
                type="array",
                semantic_type="vector",
                description="Origin shift (x, y, z)",
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="table",
                format="npz",
                fields=["counts"],
                description="Water counts per region",
            ),
        ],
        tags=["solvent", "spatial"],
        examples=[
            {
                "name": "water_count",
                "water_selection": "resname SOL",
                "center_selection": "resname LIG",
                "box_unit": [0.5, 0.5, 0.5],
                "region_size": [0.5, 0.5, 0.5],
            },
        ],
    ),

    "equipartition": AnalysisContract(
        name="equipartition",
        aliases=["kinetic-energy", "ke-distribution"],
        description="Kinetic energy distribution by group",
        required_fields=["selection"],
        optional_fields=["group_by", "velocity_scale", "length_scale"],
        field_types={
            "selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Atom selection for kinetic energy",
            ),
            "group_by": FieldSpec(
                type="string",
                semantic_type="string",
                description="How to group atoms",
            ),
            "velocity_scale": FieldSpec(
                type="float",
                semantic_type="float",
                description="Velocity scaling factor",
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="table",
                format="npz",
                fields=["group", "kinetic_energy_kJ_per_mol"],
                description="Kinetic energy per group",
            ),
        ],
        tags=["thermodynamic"],
        examples=[
            {
                "name": "equipartition",
                "selection": "protein",
            },
        ],
    ),

    "hbond": AnalysisContract(
        name="hbond",
        aliases=["hydrogen-bond"],
        description="Hydrogen bond analysis",
        required_fields=["donors", "acceptors", "dist_cutoff"],
        optional_fields=["hydrogens", "angle_cutoff"],
        field_types={
            "donors": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Hydrogen bond donor selection",
            ),
            "acceptors": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Hydrogen bond acceptor selection",
            ),
            "dist_cutoff": FieldSpec(
                type="float",
                semantic_type="float",
                description="Distance cutoff for H-bond",
                minimum=0,
                unit="nm",
            ),
            "hydrogens": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Hydrogen atom selection (for angle criteria)",
            ),
            "angle_cutoff": FieldSpec(
                type="float",
                semantic_type="float",
                description="Angle cutoff in degrees",
                minimum=0,
                maximum=180,
                unit="degrees",
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="timeseries",
                format="npz",
                fields=["time_ps", "hbond_count"],
                description="Hydrogen bond count vs time",
            ),
        ],
        tags=["structural", "solvent"],
        examples=[
            {
                "name": "hbond",
                "donors": "resname SOL and name OH",
                "acceptors": "resname SOL and name O",
                "dist_cutoff": 0.35,
            },
        ],
    ),

    "rdf": AnalysisContract(
        name="rdf",
        aliases=["radial-distribution-function", "pair-distribution"],
        description="Radial distribution function g(r)",
        required_fields=["sel_a", "sel_b", "bins", "r_max"],
        optional_fields=["pbc"],
        field_types={
            "sel_a": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Selection for group A",
            ),
            "sel_b": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Selection for group B",
            ),
            "bins": FieldSpec(
                type="integer",
                semantic_type="integer",
                description="Number of r bins",
                minimum=1,
                default=200,
            ),
            "r_max": FieldSpec(
                type="float",
                semantic_type="float",
                description="Maximum r distance",
                minimum=0,
                unit="nm",
            ),
            "pbc": FieldSpec(
                type="boolean",
                semantic_type="boolean",
                description="Apply periodic boundary conditions",
                default=True,
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="histogram",
                format="npz",
                fields=["r_nm", "gr"],
                description="RDF g(r) vs distance",
            ),
        ],
        tags=["structural", "solvent", "distribution"],
        examples=[
            {
                "name": "rdf",
                "sel_a": "resname SOL and name OW",
                "sel_b": "resname SOL and name OW",
                "bins": 200,
                "r_max": 1.0,
            },
        ],
    ),

    "end_to_end": AnalysisContract(
        name="end_to_end",
        aliases=["end-to-end-distance"],
        description="End-to-end distance for polymers",
        required_fields=["selection"],
        optional_fields=[],
        field_types={
            "selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Polymer atom selection",
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="timeseries",
                format="npz",
                fields=["time_ps", "end_to_end_nm"],
                description="End-to-end distance vs time",
            ),
        ],
        tags=["structural", "polymer"],
        examples=[
            {
                "name": "end_to_end",
                "selection": "polymer",
            },
        ],
    ),

    "contour_length": AnalysisContract(
        name="contour_length",
        aliases=["contour-length"],
        description="Contour length (bond path length) for polymers",
        required_fields=["selection"],
        optional_fields=[],
        field_types={
            "selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Polymer atom selection",
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="timeseries",
                format="npz",
                fields=["time_ps", "contour_nm"],
                description="Contour length vs time",
            ),
        ],
        tags=["structural", "polymer"],
        examples=[
            {
                "name": "contour_length",
                "selection": "polymer",
            },
        ],
    ),

    "chain_rg": AnalysisContract(
        name="chain_rg",
        aliases=["chain-rg"],
        description="Radius of gyration per chain/molecule",
        required_fields=["selection"],
        optional_fields=[],
        field_types={
            "selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Polymer atom selection",
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="table",
                format="npz",
                fields=["chain_id", "rg_nm"],
                description="Rg per chain",
            ),
        ],
        tags=["structural", "polymer"],
        examples=[
            {
                "name": "chain_rg",
                "selection": "polymer",
            },
        ],
    ),

    "bond_length_distribution": AnalysisContract(
        name="bond_length_distribution",
        aliases=["bond-length"],
        description="Bond length distribution",
        required_fields=["selection", "bins", "r_max"],
        optional_fields=[],
        field_types={
            "selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Atom selection for bond analysis",
            ),
            "bins": FieldSpec(
                type="integer",
                semantic_type="integer",
                description="Number of distance bins",
                minimum=1,
            ),
            "r_max": FieldSpec(
                type="float",
                semantic_type="float",
                description="Maximum bond distance",
                minimum=0,
                unit="nm",
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="histogram",
                format="npz",
                fields=["distance_nm", "probability"],
                description="Bond length distribution",
            ),
        ],
        tags=["structural", "bond"],
        examples=[
            {
                "name": "bond_length_distribution",
                "selection": "polymer",
                "bins": 100,
                "r_max": 0.2,
            },
        ],
    ),

    "bond_angle_distribution": AnalysisContract(
        name="bond_angle_distribution",
        aliases=["bond-angle"],
        description="Bond angle distribution",
        required_fields=["selection", "bins"],
        optional_fields=["degrees"],
        field_types={
            "selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Atom selection for angle analysis",
            ),
            "bins": FieldSpec(
                type="integer",
                semantic_type="integer",
                description="Number of angle bins",
                minimum=1,
            ),
            "degrees": FieldSpec(
                type="boolean",
                semantic_type="boolean",
                description="Use degrees (vs radians)",
                default=True,
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="histogram",
                format="npz",
                fields=["angle", "probability"],
                description="Bond angle distribution",
            ),
        ],
        tags=["structural", "bond"],
        examples=[
            {
                "name": "bond_angle_distribution",
                "selection": "polymer",
                "bins": 180,
            },
        ],
    ),

    "persistence_length": AnalysisContract(
        name="persistence_length",
        aliases=["persistence-length"],
        description="Polymer persistence length from bond vectors",
        required_fields=["selection"],
        optional_fields=[],
        field_types={
            "selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Polymer backbone selection",
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="artifact",
                format="npz",
                fields=["persistence_length_nm"],
                description="Persistence length",
            ),
        ],
        tags=["structural", "polymer"],
        examples=[
            {
                "name": "persistence_length",
                "selection": "polymer",
            },
        ],
    ),

    "docking": AnalysisContract(
        name="docking",
        aliases=["docking-analysis"],
        description="Docking pose analysis (protein-ligand interactions)",
        required_fields=["receptor_mask", "ligand_mask"],
        optional_fields=[
            "close_contact_cutoff", "hydrophobic_cutoff", "hydrogen_bond_cutoff",
            "clash_cutoff", "salt_bridge_cutoff", "halogen_bond_cutoff",
            "metal_coordination_cutoff", "cation_pi_cutoff", "pi_pi_cutoff",
            "hbond_min_angle_deg", "donor_hydrogen_cutoff", "allow_missing_hydrogen",
            "length_scale", "frame_indices", "max_events_per_frame",
        ],
        field_types={
            "receptor_mask": FieldSpec(
                type="string",
                semantic_type="mask",
                description="Receptor atom mask",
            ),
            "ligand_mask": FieldSpec(
                type="string",
                semantic_type="mask",
                description="Ligand atom mask",
            ),
            "close_contact_cutoff": FieldSpec(
                type="float",
                semantic_type="float",
                description="Close contact distance cutoff",
                minimum=0,
                unit="nm",
            ),
            "hydrophobic_cutoff": FieldSpec(
                type="float",
                semantic_type="float",
                description="Hydrophobic interaction cutoff",
                minimum=0,
                unit="nm",
            ),
            "hydrogen_bond_cutoff": FieldSpec(
                type="float",
                semantic_type="float",
                description="Hydrogen bond distance cutoff",
                minimum=0,
                unit="nm",
            ),
            "clash_cutoff": FieldSpec(
                type="float",
                semantic_type="float",
                description="Clash detection cutoff",
                minimum=0,
                unit="nm",
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="table",
                format="json",
                fields=["interaction_type", "count", "details"],
                description="Docking interaction summary",
            ),
        ],
        tags=["docking", "protein", "structural"],
        examples=[
            {
                "name": "docking",
                "receptor_mask": "protein",
                "ligand_mask": "resname LIG",
            },
        ],
    ),

    "dssp": AnalysisContract(
        name="dssp",
        aliases=["secondary-structure"],
        description="Protein secondary structure assignment (DSSP)",
        required_fields=[],
        optional_fields=["mask", "simplified"],
        field_types={
            "mask": FieldSpec(
                type="string",
                semantic_type="mask",
                description="Protein atom mask",
            ),
            "simplified": FieldSpec(
                type="boolean",
                semantic_type="boolean",
                description="Use simplified 3-state classification",
                default=False,
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="table",
                format="npz",
                fields=["residue", "structure"],
                description="Secondary structure per residue",
            ),
        ],
        tags=["protein", "structural"],
        examples=[
            {
                "name": "dssp",
                "mask": "protein",
            },
        ],
    ),

    "diffusion": AnalysisContract(
        name="diffusion",
        aliases=["diffusion-coefficient"],
        description="Diffusion coefficient from MSD slope",
        required_fields=[],
        optional_fields=["mask", "tstep", "individual"],
        field_types={
            "mask": FieldSpec(
                type="string",
                semantic_type="mask",
                description="Atom mask",
            ),
            "tstep": FieldSpec(
                type="float",
                semantic_type="float",
                description="Time step in picoseconds",
            ),
            "individual": FieldSpec(
                type="boolean",
                semantic_type="boolean",
                description="Compute per-species diffusion",
                default=False,
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="artifact",
                format="npz",
                fields=["diffusion_coefficient", "msd_data"],
                description="Diffusion coefficient and MSD data",
            ),
        ],
        tags=["dynamic", "transport", "diffusion"],
        examples=[
            {
                "name": "diffusion",
                "mask": "resname SOL",
            },
        ],
    ),

    "pca": AnalysisContract(
        name="pca",
        aliases=["principal-component-analysis", "pca-analysis"],
        description="Principal component analysis of atomic fluctuations",
        required_fields=["mask"],
        optional_fields=["n_vecs", "fit", "ref", "ref_mask"],
        field_types={
            "mask": FieldSpec(
                type="string",
                semantic_type="mask",
                description="Atom mask for PCA",
            ),
            "n_vecs": FieldSpec(
                type="integer",
                semantic_type="integer",
                description="Number of eigenvectors to compute",
                minimum=1,
            ),
            "fit": FieldSpec(
                type="boolean",
                semantic_type="boolean",
                description="Fit to average structure",
                default=True,
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="artifact",
                format="npz",
                fields=["eigenvalues", "eigenvectors", "projections"],
                description="PCA eigendecomposition results",
            ),
        ],
        tags=["structural", "protein"],
        examples=[
            {
                "name": "pca",
                "mask": "protein and name CA",
                "n_vecs": 10,
            },
        ],
    ),

    "rmsf": AnalysisContract(
        name="rmsf",
        aliases=["root-mean-square-fluctuation"],
        description="Root mean square fluctuation (per-atom mobility)",
        required_fields=[],
        optional_fields=["mask", "byres", "bymask", "calcadp", "length_scale"],
        field_types={
            "mask": FieldSpec(
                type="string",
                semantic_type="mask",
                description="Atom mask",
            ),
            "byres": FieldSpec(
                type="boolean",
                semantic_type="boolean",
                description="Aggregate by residue",
                default=False,
            ),
            "bymask": FieldSpec(
                type="string",
                semantic_type="mask",
                description="Mask for aggregation",
            ),
            "calcadp": FieldSpec(
                type="boolean",
                semantic_type="boolean",
                description="Calculate ADP (B-factors)",
                default=False,
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="table",
                format="npz",
                fields=["atom", "rmsf_nm"],
                description="RMSF per atom/residue",
            ),
        ],
        tags=["structural", "protein"],
        examples=[
            {
                "name": "rmsf",
                "mask": "protein and name CA",
                "byres": True,
            },
        ],
    ),

    "density": AnalysisContract(
        name="density",
        aliases=["number-density", "mass-density"],
        description="Density profile along a direction",
        required_fields=[],
        optional_fields=["mask", "density_type", "delta", "direction", "cutoff", "center"],
        field_types={
            "mask": FieldSpec(
                type="string",
                semantic_type="mask",
                description="Atom mask",
            ),
            "density_type": FieldSpec(
                type="string",
                semantic_type="string",
                description="Type of density (number, mass, charge)",
                choices=["number", "mass", "charge"],
            ),
            "delta": FieldSpec(
                type="float",
                semantic_type="float",
                description="Bin width",
                minimum=0,
            ),
            "direction": FieldSpec(
                type="string",
                semantic_type="string",
                description="Direction for profile (x, y, z, or normal)",
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="profile",
                format="npz",
                fields=["position", "density"],
                description="Density vs position",
            ),
        ],
        tags=["structural", "spatial"],
        examples=[
            {
                "name": "density",
                "mask": "resname SOL",
                "direction": "z",
            },
        ],
    ),

    "native_contacts": AnalysisContract(
        name="native_contacts",
        aliases=["native-contacts", "q-value"],
        description="Native contact analysis (folding)",
        required_fields=[],
        optional_fields=["mask", "mask2", "ref", "distance", "mindist", "maxdist", "image"],
        field_types={
            "mask": FieldSpec(
                type="string",
                semantic_type="mask",
                description="First atom mask",
            ),
            "mask2": FieldSpec(
                type="string",
                semantic_type="mask",
                description="Second atom mask",
            ),
            "distance": FieldSpec(
                type="float",
                semantic_type="float",
                description="Contact distance cutoff",
                minimum=0,
                unit="nm",
            ),
            "mindist": FieldSpec(
                type="float",
                semantic_type="float",
                description="Minimum contact distance",
                minimum=0,
                unit="nm",
            ),
            "maxdist": FieldSpec(
                type="float",
                semantic_type="float",
                description="Maximum contact distance",
                minimum=0,
                unit="nm",
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="timeseries",
                format="npz",
                fields=["time_ps", "native_contacts", "q_value"],
                description="Native contacts vs time",
            ),
        ],
        tags=["structural", "protein", "folding"],
        examples=[
            {
                "name": "native_contacts",
                "mask": "protein",
            },
        ],
    ),

    "volmap": AnalysisContract(
        name="volmap",
        aliases=["volumetric-map", "density-map"],
        description="Volumetric density map",
        required_fields=[],
        optional_fields=[
            "mask", "grid_spacing", "size", "center", "buffer",
            "centermask", "radscale", "peakcut", "dtype",
        ],
        field_types={
            "mask": FieldSpec(
                type="string",
                semantic_type="mask",
                description="Atom mask",
            ),
            "grid_spacing": FieldSpec(
                type="float",
                semantic_type="float",
                description="Grid spacing",
                minimum=0,
                unit="nm",
            ),
            "size": FieldSpec(
                type="array",
                semantic_type="vector",
                description="Grid dimensions (nx, ny, nz)",
            ),
            "center": FieldSpec(
                type="array",
                semantic_type="vector",
                description="Grid center (x, y, z)",
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="grid",
                format="dx",
                fields=["density"],
                description="3D volumetric density",
            ),
        ],
        tags=["spatial", "grid", "solvent"],
        examples=[
            {
                "name": "volmap",
                "mask": "resname SOL and name OW",
                "grid_spacing": 0.1,
            },
        ],
    ),

    "free_volume": AnalysisContract(
        name="free_volume",
        aliases=["free-volume", "free-volume-grid", "voxel-free-volume"],
        description="Voxel-grid free-volume fraction. Useful for spatial void maps, not Bondi-style polymer FFV.",
        required_fields=["selection", "center_selection"],
        optional_fields=["box_unit", "region_size", "probe_radius", "shift", "length_scale"],
        field_types={
            "selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Atoms treated as occupied volume",
            ),
            "center_selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Selection used to define grid origin and auto-detect region_size",
            ),
            "box_unit": FieldSpec(
                type="array",
                semantic_type="vector",
                description="Voxel size (x, y, z) in Angstroms. Defaults to [1.0, 1.0, 1.0] if not specified.",
                default=[1.0, 1.0, 1.0],
            ),
            "region_size": FieldSpec(
                type="array",
                semantic_type="vector",
                description="Region extents (x, y, z) in Angstroms. Auto-detected from center_selection bounding box with 10%% padding if not specified.",
            ),
            "probe_radius": FieldSpec(
                type="float",
                semantic_type="float",
                description="Probe radius that expands occupied volume",
                minimum=0,
            ),
            "shift": FieldSpec(
                type="array",
                semantic_type="vector",
                description="Shift for centered coordinates",
            ),
            "length_scale": FieldSpec(
                type="float",
                semantic_type="float",
                description="Coordinate length scale",
                default=1.0,
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="grid",
                format="npz",
                fields=["dims", "mean", "std", "first", "last", "min", "max"],
                description="Per-voxel free-volume fraction statistics",
            ),
        ],
        tags=["spatial", "grid", "void", "solvation"],
        examples=[
            {
                "name": "free_volume",
                "selection": "protein",
                "center_selection": "protein",
                "box_unit": [1.0, 1.0, 1.0],
                "region_size": [30.0, 30.0, 30.0],
                "probe_radius": 0.5,
                "note": "Explicit parameters",
            },
            {
                "name": "free_volume_auto",
                "selection": "protein",
                "center_selection": "protein",
                "note": "Auto-detects region_size from bounding box, defaults box_unit to 1.0 Å",
            },
        ],
    ),

    "bondi_ffv": AnalysisContract(
        name="bondi_ffv",
        aliases=["bondi-ffv", "ffv", "fractional-free-volume"],
        description="GROMACS-style free-volume Monte Carlo with Bondi radii. Reports raw free-volume fraction and the Lourenco/GROMACS FFV convention FFV = 1 - scale * (1 - free_volume_fraction).",
        required_fields=["selection"],
        optional_fields=["bondi_scale", "probe_radius", "seed", "ninsert_per_nm3", "length_scale"],
        field_types={
            "selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Atoms treated as excluded volume during probe insertion",
            ),
            "bondi_scale": FieldSpec(
                type="float",
                semantic_type="float",
                description="Homogeneity scale factor in FFV = 1 - scale * (1 - free_volume_fraction)",
                default=1.3,
                minimum=0,
            ),
            "probe_radius": FieldSpec(
                type="float",
                semantic_type="float",
                description="Probe radius in Angstroms. 0.0 yields true free volume.",
                default=0.0,
                minimum=0,
            ),
            "seed": FieldSpec(
                type="integer",
                semantic_type="integer",
                description="Deterministic Monte Carlo seed",
                default=0,
            ),
            "ninsert_per_nm3": FieldSpec(
                type="integer",
                semantic_type="integer",
                description="Probe insertions per cubic nanometer",
                default=1000,
                minimum=1,
            ),
            "length_scale": FieldSpec(
                type="float",
                semantic_type="float",
                description="Coordinate/box length scale",
                default=1.0,
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="table",
                format="json",
                fields=[
                    "time",
                    "total_volume_a3",
                    "vdw_volume_a3",
                    "raw_free_volume_a3",
                    "raw_free_volume_fraction",
                    "fractional_free_volume",
                    "density_g_cm3",
                    "molar_mass_dalton",
                    "bondi_scale",
                ],
                description="Per-frame Bondi/Lourenco free-volume metrics",
            ),
        ],
        tags=["polymer", "free-volume", "bondi", "gromacs-parity"],
        examples=[
            {
                "name": "bondi_ffv",
                "selection": "not name QQQQ",
                "bondi_scale": 1.3,
                "probe_radius": 0.0,
                "seed": -1107428613,
                "ninsert_per_nm3": 1000,
            },
        ],
    ),

    "surf": AnalysisContract(
        name="surf",
        aliases=["surface-area", "sas"],
        description="Solvent accessible surface area",
        required_fields=[],
        optional_fields=["mask", "algorithm", "probe_radius", "n_sphere_points", "radii"],
        field_types={
            "mask": FieldSpec(
                type="string",
                semantic_type="mask",
                description="Atom mask",
            ),
            "algorithm": FieldSpec(
                type="string",
                semantic_type="string",
                description="Surface area algorithm",
            ),
            "probe_radius": FieldSpec(
                type="float",
                semantic_type="float",
                description="Probe radius",
                minimum=0,
                unit="nm",
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="timeseries",
                format="npz",
                fields=["time_ps", "surface_area_nm2"],
                description="Surface area vs time",
            ),
        ],
        tags=["structural", "surface"],
        examples=[
            {
                "name": "surf",
                "mask": "protein",
            },
        ],
    ),

    "molsurf": AnalysisContract(
        name="molsurf",
        aliases=["molecular-surface"],
        description="Molecular surface area (Connolly)",
        required_fields=[],
        optional_fields=["mask", "algorithm", "probe_radius", "radii"],
        field_types={
            "mask": FieldSpec(
                type="string",
                semantic_type="mask",
                description="Atom mask",
            ),
            "probe_radius": FieldSpec(
                type="float",
                semantic_type="float",
                description="Probe radius",
                minimum=0,
                unit="nm",
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="timeseries",
                format="npz",
                fields=["time_ps", "surface_area_nm2"],
                description="Molecular surface area vs time",
            ),
        ],
        tags=["structural", "surface"],
        examples=[
            {
                "name": "molsurf",
                "mask": "protein",
            },
        ],
    ),

    "watershell": AnalysisContract(
        name="watershell",
        aliases=["water-shell", "solvation-shell"],
        description="Water shell analysis around solute",
        required_fields=["solute_mask"],
        optional_fields=["solvent_mask", "lower", "upper", "image"],
        field_types={
            "solute_mask": FieldSpec(
                type="string",
                semantic_type="mask",
                description="Solute atom mask",
            ),
            "solvent_mask": FieldSpec(
                type="string",
                semantic_type="mask",
                description="Solvent atom mask",
            ),
            "lower": FieldSpec(
                type="float",
                semantic_type="float",
                description="Lower distance cutoff",
                minimum=0,
                unit="nm",
            ),
            "upper": FieldSpec(
                type="float",
                semantic_type="float",
                description="Upper distance cutoff",
                minimum=0,
                unit="nm",
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="timeseries",
                format="npz",
                fields=["time_ps", "water_count"],
                description="Water count in shell vs time",
            ),
        ],
        tags=["solvent", "solvation"],
        examples=[
            {
                "name": "watershell",
                "solute_mask": "protein",
                "lower": 0.3,
                "upper": 0.5,
            },
        ],
    ),

    "tordiff": AnalysisContract(
        name="tordiff",
        aliases=["torsional-diffusion"],
        description="Torsional diffusion coefficient",
        required_fields=["mask"],
        optional_fields=["tstep"],
        field_types={
            "mask": FieldSpec(
                type="string",
                semantic_type="mask",
                description="Atom mask defining torsion",
            ),
            "tstep": FieldSpec(
                type="float",
                semantic_type="float",
                description="Time step in picoseconds",
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="artifact",
                format="npz",
                fields=["torsional_diffusion"],
                description="Torsional diffusion coefficient",
            ),
        ],
        tags=["dynamic", "diffusion"],
        examples=[
            {
                "name": "tordiff",
                "mask": "dihedral selection",
            },
        ],
    ),

    "projection": AnalysisContract(
        name="projection",
        aliases=["pca-projection"],
        description="Project trajectory onto PCA eigenvectors",
        required_fields=["mask"],
        optional_fields=["eigenvec", "n_vecs", "fit", "ref", "ref_mask"],
        field_types={
            "mask": FieldSpec(
                type="string",
                semantic_type="mask",
                description="Atom mask",
            ),
            "eigenvec": FieldSpec(
                type="path",
                semantic_type="path",
                description="Path to eigenvector file",
            ),
            "n_vecs": FieldSpec(
                type="integer",
                semantic_type="integer",
                description="Number of eigenvectors",
                minimum=1,
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="timeseries",
                format="npz",
                fields=["time_ps", "pc1", "pc2", "..."],
                description="PCA projections vs time",
            ),
        ],
        tags=["structural", "protein"],
        examples=[
            {
                "name": "projection",
                "mask": "protein and name CA",
                "n_vecs": 3,
            },
        ],
    ),

    "nmr": AnalysisContract(
        name="nmr",
        aliases=["nmr-order-parameters"],
        description="NMR NH order parameters",
        required_fields=["selection"],
        optional_fields=["vector_pairs", "method", "order", "tstep", "tcorr", "length_scale", "pbc"],
        field_types={
            "selection": FieldSpec(
                type="string",
                semantic_type="selection",
                description="NH vector selection",
            ),
            "vector_pairs": FieldSpec(
                type="string",
                semantic_type="string",
                description="Vector pair specification",
            ),
            "method": FieldSpec(
                type="string",
                semantic_type="string",
                description="Computation method",
            ),
            "order": FieldSpec(
                type="integer",
                semantic_type="integer",
                description="Order parameter (2 for S^2)",
                default=2,
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="table",
                format="npz",
                fields=["residue", "order_parameter"],
                description="Order parameters per residue",
            ),
        ],
        tags=["nmr", "protein"],
        examples=[
            {
                "name": "nmr",
                "selection": "protein",
            },
        ],
    ),

    "jcoupling": AnalysisContract(
        name="jcoupling",
        aliases=["j-coupling", "scalar-coupling"],
        description="J-coupling constants from dihedral angles",
        required_fields=["dihedrals"],
        optional_fields=["karplus", "kfile", "phase_deg", "length_scale", "pbc"],
        field_types={
            "dihedrals": FieldSpec(
                type="array",
                semantic_type="vector",
                description="Dihedral atom indices (quadruples)",
            ),
            "karplus": FieldSpec(
                type="string",
                semantic_type="string",
                description="Karplus parameterization",
            ),
            "kfile": FieldSpec(
                type="path",
                semantic_type="path",
                description="Path to Karplus parameter file",
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="table",
                format="npz",
                fields=["dihedral", "j_coupling_hz"],
                description="J-coupling per dihedral",
            ),
        ],
        tags=["nmr"],
        examples=[
            {
                "name": "jcoupling",
                "dihedrals": [[1, 2, 3, 4], [5, 6, 7, 8]],
            },
        ],
    ),

    "gist": AnalysisContract(
        name="gist",
        aliases=["grid-inhomogeneous-solvation-theory"],
        description="Grid inhomogeneous solvation theory (water thermodynamics)",
        required_fields=["solute", "solvent"],
        optional_fields=[],
        field_types={
            "solute": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Solute selection",
            ),
            "solvent": FieldSpec(
                type="string",
                semantic_type="selection",
                description="Solvent selection",
            ),
        },
        outputs=[
            ArtifactSpec(
                kind="grid",
                format="npz",
                fields=["energy", "entropy"],
                description="3D thermodynamic maps",
            ),
        ],
        tags=["thermodynamic", "solvent", "grid"],
        examples=[
            {
                "name": "gist",
                "solute": "protein",
                "solvent": "resname SOL",
            },
        ],
    ),
}


# CLI to plan name mappings (preserving hyphen/underscore variants)
CLI_TO_ANALYSIS: Dict[str, str] = {
    "rg": "rg",
    "rmsd": "rmsd",
    "msd": "msd",
    "rotacf": "rotacf",
    "conductivity": "conductivity",
    "dielectric": "dielectric",
    "dipole-alignment": "dipole_alignment",
    "dipole_alignment": "dipole_alignment",
    "ion-pair-correlation": "ion_pair_correlation",
    "ion_pair_correlation": "ion_pair_correlation",
    "structure-factor": "structure_factor",
    "structure_factor": "structure_factor",
    "water-count": "water_count",
    "water_count": "water_count",
    "free-volume": "free_volume",
    "free_volume": "free_volume",
    "free-volume-grid": "free_volume",
    "voxel-free-volume": "free_volume",
    "bondi-ffv": "bondi_ffv",
    "bondi_ffv": "bondi_ffv",
    "ffv": "bondi_ffv",
    "fractional-free-volume": "bondi_ffv",
    "equipartition": "equipartition",
    "hbond": "hbond",
    "rdf": "rdf",
    "end-to-end": "end_to_end",
    "end_to_end": "end_to_end",
    "contour-length": "contour_length",
    "contour_length": "contour_length",
    "chain-rg": "chain_rg",
    "chain_rg": "chain_rg",
    "bond-length-distribution": "bond_length_distribution",
    "bond_length_distribution": "bond_length_distribution",
    "bond-angle-distribution": "bond_angle_distribution",
    "bond_angle_distribution": "bond_angle_distribution",
    "persistence-length": "persistence_length",
    "persistence_length": "persistence_length",
    "docking": "docking",
    "dssp": "dssp",
    "diffusion": "diffusion",
    "pca": "pca",
    "rmsf": "rmsf",
    "density": "density",
    "native-contacts": "native_contacts",
    "native_contacts": "native_contacts",
    "volmap": "volmap",
    "surf": "surf",
    "molsurf": "molsurf",
    "watershell": "watershell",
    "tordiff": "tordiff",
    "projection": "projection",
    "nmr": "nmr",
    "jcoupling": "jcoupling",
    "gist": "gist",
}

_ANALYSIS_SHARED_FIELDS = frozenset({"out", "device", "chunk_frames"})
_RUN_REQUEST_TOP_LEVEL_FIELDS = frozenset(RunRequest.model_fields.keys())


def _resolve_analysis_name(name: str) -> str:
    """Resolve CLI name to canonical analysis name."""
    # Direct lookup
    if name in ANALYSIS_METADATA:
        return name
    # Try hyphen/underscore variants
    alt = name.replace("-", "_")
    if alt in ANALYSIS_METADATA:
        return alt
    alt = name.replace("_", "-")
    if alt in ANALYSIS_METADATA:
        return alt.replace("_", "-")
    # Try aliases
    for contract in ANALYSIS_METADATA.values():
        if name in contract.aliases or name.replace("_", "-") in contract.aliases:
            return contract.name
    raise ValueError(f"unknown analysis: {name}")


def get_plan_schema(plan_name: str) -> Dict[str, Any]:
    """Get full contract metadata for a single analysis plan.

    Args:
        plan_name: Analysis name (CLI or canonical form)

    Returns:
        Dictionary with complete contract metadata

    Raises:
        ValueError: If analysis name is unknown
    """
    try:
        canonical = _resolve_analysis_name(plan_name)
    except ValueError:
        raise ValueError(f"unknown plan: {plan_name}")
    contract = ANALYSIS_METADATA.get(canonical)
    if not contract:
        raise ValueError(f"unknown plan: {plan_name}")
    return contract.to_dict()


def list_all_plans(details: bool = False) -> Dict[str, Any]:
    """List all available analysis plans.

    Args:
        details: Include full metadata for each plan

    Returns:
        Dictionary with plan names or detailed contracts
    """
    if not details:
        return {"plans": sorted(ANALYSIS_METADATA.keys())}
    return {
        "plans": [contract.to_dict() for contract in ANALYSIS_METADATA.values()],
    }


# Validation result structures
class ValidationErrorDetail(BaseModel):
    """Single validation error detail."""
    code: str
    path: str
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Result of request validation."""
    schema_version: str = AGENT_REQUEST_SCHEMA_VERSION
    status: Literal["ok", "error"]
    valid: bool
    normalized_request: Optional[Dict[str, Any]] = None
    errors: List[ValidationErrorDetail] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


def validate_request(
    payload: Dict[str, Any],
    *,
    strict: bool = False,
    check_selections: bool = False,
) -> ValidationResult:
    """Validate a RunRequest against full contract.

    Args:
        payload: Raw request dictionary
        strict: If True, reject unknown fields
        check_selections: If True, validate selection syntax (requires system load)

    Returns:
        ValidationResult with status, errors, warnings, and normalized request
    """
    errors: List[ValidationErrorDetail] = []
    warnings: List[str] = []
    normalized: Optional[Dict[str, Any]] = None

    # Step 1: Pydantic schema validation
    try:
        request = validate_run_request(payload)
        normalized = request.model_dump(mode="python")
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) if err["loc"] else "root"
            errors.append(ValidationErrorDetail(
                code="E_SCHEMA_VALIDATION",
                path=loc,
                message=err.get("msg", "validation error"),
            ))
        return ValidationResult(
            status="error",
            valid=False,
            errors=errors,
            warnings=warnings,
        )

    # Step 2: Analysis-specific validation
    for idx, analysis in enumerate(normalized.get("analyses", [])):
        analysis_name = analysis.get("name", "")
        path_prefix = f"analyses[{idx}]"

        try:
            canonical = _resolve_analysis_name(analysis_name)
        except ValueError as e:
            errors.append(ValidationErrorDetail(
                code="E_UNKNOWN_ANALYSIS",
                path=f"{path_prefix}.name",
                message=str(e),
            ))
            continue

        contract = ANALYSIS_METADATA.get(canonical)
        if not contract:
            errors.append(ValidationErrorDetail(
                code="E_MISSING_CONTRACT",
                path=f"{path_prefix}",
                message=f"No contract found for analysis: {canonical}",
            ))
            continue

        # Check required fields
        provided_fields = set(k for k in analysis.keys() if k != "name")
        required = set(contract.required_fields)
        missing = required - provided_fields

        for field in missing:
            errors.append(ValidationErrorDetail(
                code="E_REQUIRED_FIELD",
                path=f"{path_prefix}.{field}",
                message=f"{canonical} requires field '{field}'",
            ))

        # Check field types where we can
        for field_name, field_value in analysis.items():
            if field_name == "name":
                continue
            if field_name in _ANALYSIS_SHARED_FIELDS:
                continue
            field_spec = contract.field_types.get(field_name)
            if not field_spec:
                if strict:
                    errors.append(ValidationErrorDetail(
                        code="E_UNKNOWN_FIELD",
                        path=f"{path_prefix}.{field_name}",
                        message=f"Unknown field for {canonical}: {field_name}",
                    ))
                continue

            # Type checks for known field types
            if field_spec.type == "boolean" and not isinstance(field_value, bool):
                errors.append(ValidationErrorDetail(
                    code="E_FIELD_TYPE",
                    path=f"{path_prefix}.{field_name}",
                    message=f"Expected boolean, got {type(field_value).__name__}",
                ))
            elif field_spec.type == "integer" and not isinstance(field_value, int):
                errors.append(ValidationErrorDetail(
                    code="E_FIELD_TYPE",
                    path=f"{path_prefix}.{field_name}",
                    message=f"Expected integer, got {type(field_value).__name__}",
                ))
            elif field_spec.type == "float" and not isinstance(field_value, (int, float)):
                errors.append(ValidationErrorDetail(
                    code="E_FIELD_TYPE",
                    path=f"{path_prefix}.{field_name}",
                    message=f"Expected float, got {type(field_value).__name__}",
                ))
            elif field_spec.type == "array" and not isinstance(field_value, (list, tuple)):
                errors.append(ValidationErrorDetail(
                    code="E_FIELD_TYPE",
                    path=f"{path_prefix}.{field_name}",
                    message=f"Expected array, got {type(field_value).__name__}",
                ))

            # Range checks
            if isinstance(field_value, (int, float)):
                if field_spec.minimum is not None and field_value < field_spec.minimum:
                    errors.append(ValidationErrorDetail(
                        code="E_VALUE_RANGE",
                        path=f"{path_prefix}.{field_name}",
                        message=f"Value {field_value} below minimum {field_spec.minimum}",
                    ))
                if field_spec.maximum is not None and field_value > field_spec.maximum:
                    errors.append(ValidationErrorDetail(
                        code="E_VALUE_RANGE",
                        path=f"{path_prefix}.{field_name}",
                        message=f"Value {field_value} above maximum {field_spec.maximum}",
                    ))

            # Choice checks
            if field_spec.choices and field_value not in field_spec.choices:
                errors.append(ValidationErrorDetail(
                    code="E_INVALID_CHOICE",
                    path=f"{path_prefix}.{field_name}",
                    message=f"Invalid choice: {field_value}. Must be one of {field_spec.choices}",
                ))

    # Step 3: Selection syntax validation (if requested)
    # This would require loading the system - defer to lint-selection command

    return ValidationResult(
        status="ok" if not errors else "error",
        valid=len(errors) == 0,
        normalized_request=normalized if not errors else None,
        errors=errors,
        warnings=warnings,
    )


def normalize_request(
    payload: Dict[str, Any],
    *,
    strip_unknown: bool = False,
) -> Dict[str, Any]:
    """Canonicalize a request by resolving aliases and filling defaults.

    Args:
        payload: Raw request dictionary
        strip_unknown: Remove fields not in contract definitions

    Returns:
        Canonicalized request dictionary
    """
    # Start with a copy of the payload
    import copy
    normalized = copy.deepcopy(payload)

    # Normalize field aliases (topology -> system, traj -> trajectory)
    if "topology" in normalized:
        if "system" not in normalized:
            normalized["system"] = normalized.pop("topology")
        else:
            normalized.pop("topology")
    if "traj" in normalized:
        if "trajectory" not in normalized:
            normalized["trajectory"] = normalized.pop("traj")
        else:
            normalized.pop("traj")

    if strip_unknown:
        normalized = {
            key: value
            for key, value in normalized.items()
            if key in _RUN_REQUEST_TOP_LEVEL_FIELDS
        }

    # Normalize analysis names and fill defaults
    normalized_analyses = []
    for analysis in normalized.get("analyses", []):
        if not isinstance(analysis, dict):
            normalized_analyses.append(analysis)
            continue

        analysis_payload = dict(analysis)
        name = analysis_payload.get("name", "")
        try:
            canonical = _resolve_analysis_name(name)
            analysis_payload["name"] = canonical
        except ValueError:
            normalized_analyses.append(analysis_payload)
            continue  # Keep original name if unknown

        # Fill default values for optional fields
        contract = ANALYSIS_METADATA.get(canonical)
        if contract:
            if strip_unknown:
                allowed_fields = {"name", *contract.field_types.keys(), *_ANALYSIS_SHARED_FIELDS}
                analysis_payload = {
                    key: value
                    for key, value in analysis_payload.items()
                    if key in allowed_fields
                }
            for field_name, field_spec in contract.field_types.items():
                if field_name not in analysis_payload and field_spec.default is not None:
                    analysis_payload[field_name] = field_spec.default

        normalized_analyses.append(analysis_payload)

    if "analyses" in normalized:
        normalized["analyses"] = normalized_analyses

    return normalized


def generate_template(
    analysis_name: str,
    *,
    fill_defaults: bool = False,
) -> Dict[str, Any]:
    """Generate a request template for a single analysis.

    Args:
        analysis_name: Analysis name
        fill_defaults: Include default values for optional fields

    Returns:
        Template request dictionary
    """
    canonical = _resolve_analysis_name(analysis_name)
    contract = ANALYSIS_METADATA.get(canonical)
    if not contract:
        raise ValueError(f"unknown analysis: {analysis_name}")

    analysis_spec: Dict[str, Any] = {"name": canonical}

    # Add required fields with placeholder values
    for field in contract.required_fields:
        field_spec = contract.field_types.get(field)
        if field_spec:
            if field_spec.semantic_type in ("selection", "mask"):
                analysis_spec[field] = f"<{field}_expression>"
            elif field_spec.type == "array":
                analysis_spec[field] = []
            elif field_spec.type == "integer":
                analysis_spec[field] = field_spec.default or 0
            elif field_spec.type == "float":
                analysis_spec[field] = field_spec.default or 0.0
            elif field_spec.type == "boolean":
                analysis_spec[field] = field_spec.default or False
            elif field_spec.semantic_type == "charges":
                analysis_spec[field] = "by_resname"
            else:
                analysis_spec[field] = f"<{field}>"

    # Add optional fields with defaults if requested
    if fill_defaults:
        for field, field_spec in contract.field_types.items():
            if field in contract.optional_fields and field not in analysis_spec:
                if field_spec.default is not None:
                    analysis_spec[field] = field_spec.default

    return {
        "version": AGENT_REQUEST_SCHEMA_VERSION,
        "system": {"path": "<topology-path>"},
        "trajectory": {"path": "<trajectory-path>"},
        "analyses": [analysis_spec],
    }


def _compute_catalog_hash() -> str:
    """Compute hash of analysis catalog for versioning."""
    # Sort analysis names and metadata for stable hash
    items = []
    for name in sorted(ANALYSIS_METADATA.keys()):
        contract = ANALYSIS_METADATA[name]
        items.append(f"{name}:{','.join(sorted(contract.required_fields))}")
    content = "|".join(items)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def capabilities() -> Dict[str, Any]:
    """Return capabilities/version fingerprint."""
    return {
        "schema_version": AGENT_REQUEST_SCHEMA_VERSION,
        "cli_version": _resolve_cli_version(),
        "available_plans": sorted(ANALYSIS_METADATA.keys()),
        "plan_catalog_hash": _compute_catalog_hash(),
        "supports_streaming": True,
        "supports_selection_linting": True,
    }


def _resolve_cli_version() -> str:
    """Get CLI version string."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        for name in ("warp-md", "warp_md"):
            try:
                return version(name)
            except PackageNotFoundError:
                continue
    except Exception:
        pass
    return "dev"


# Selection linting

class SelectionLintResult(BaseModel):
    """Result of selection/mask linting."""
    model_config = ConfigDict(extra="forbid")

    valid: bool
    expression: str
    field_type: Literal["selection", "mask"]
    matched_atoms: Optional[int] = None
    total_atoms: Optional[int] = None
    error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


def lint_selection(
    expr: str,
    field_type: Literal["selection", "mask"] = "selection",
    system_path: Optional[str] = None,
) -> SelectionLintResult:
    """Validate a selection expression without running analysis.

    Args:
        expr: Selection expression to validate
        field_type: Type of field (selection or mask)
        system_path: Optional path to topology file for atom count

    Returns:
        SelectionLintResult with validation status
    """
    # Basic syntax validation
    if not expr or not expr.strip():
        return SelectionLintResult(
            valid=False,
            expression=expr,
            field_type=field_type,
            error="Selection expression cannot be empty",
        )

    # Check for obviously malformed expressions
    expr_stripped = expr.strip()

    # Check for unbalanced quotes
    single_quotes = expr_stripped.count("'")
    double_quotes = expr_stripped.count('"')
    if single_quotes % 2 != 0:
        return SelectionLintResult(
            valid=False,
            expression=expr,
            field_type=field_type,
            error="Unbalanced single quotes in selection",
        )
    if double_quotes % 2 != 0:
        return SelectionLintResult(
            valid=False,
            expression=expr,
            field_type=field_type,
            error="Unbalanced double quotes in selection",
        )

    # Check for unbalanced parentheses
    paren_depth = 0
    for i, char in enumerate(expr_stripped):
        if char == '(':
            paren_depth += 1
        elif char == ')':
            paren_depth -= 1
        if paren_depth < 0:
            return SelectionLintResult(
                valid=False,
                expression=expr,
                field_type=field_type,
                error=f"Unbalanced parentheses at position {i}",
            )
    if paren_depth != 0:
        return SelectionLintResult(
            valid=False,
            expression=expr,
            field_type=field_type,
            error="Unbalanced parentheses in selection",
        )

    # If system path provided, try to load and count atoms
    matched_atoms = None
    total_atoms = None
    warnings_list = []

    if system_path:
        try:
            from .cli_api import _load_system
            system = _load_system({"path": system_path})
            total_atoms = len(system.atoms)

            try:
                selection = system.select(expr_stripped)
                matched_atoms = len(selection.indices)

                if matched_atoms == 0:
                    warnings_list.append("Selection matched zero atoms")

            except Exception as sel_exc:
                # Selection compilation failed
                return SelectionLintResult(
                    valid=False,
                    expression=expr,
                    field_type=field_type,
                    error=f"Selection syntax error: {sel_exc}",
                )
        except Exception as load_exc:
            warnings_list.append(f"Could not load topology for atom count: {load_exc}")

    return SelectionLintResult(
        valid=True,
        expression=expr,
        field_type=field_type,
        matched_atoms=matched_atoms,
        total_atoms=total_atoms,
        warnings=warnings_list,
    )


# Keyword mapping for goal-to-analysis suggestions
# Maps goal keywords to analysis tags, descriptions, or direct analysis names
_GOAL_KEYWORDS: Dict[str, List[str]] = {
    # Structural analysis
    "radius": ["rg", "radius-of-gyration"],
    "gyration": ["rg", "radius-of-gyration"],
    "size": ["rg"],
    "compactness": ["rg"],
    "rmsd": ["rmsd"],
    "alignment": ["rmsd", "dipole_alignment"],
    "structure": ["rmsd", "dssp", "ramachandran"],
    "secondary": ["dssp"],
    "backbone": ["dssp", "ramachandran"],
    "dihedral": ["ramachandran", "dihedral"],
    "torsion": ["dihedral"],

    # Dynamics
    "motion": ["msd", "displacement"],
    "diffusion": ["msd"],
    "transport": ["msd", "conductivity", "mobility"],
    "velocity": ["velocity"],
    "flow": ["velocity", "current_density"],

    # Electrostatics
    "charge": ["dipole_alignment", "conductivity"],
    "dipole": ["dipole_alignment"],
    "polarization": ["dipole_alignment"],
    "conductivity": ["conductivity"],
    "permittivity": ["dielectric"],
    "dielectric": ["dielectric"],
    "electrostatic": ["dielectric", "dipole_alignment", "conductivity"],
    "potential": ["electrokinetic"],

    # Distribution/Correlation
    "distribution": ["rdf", "coordination", "volume"],
    "pair": ["rdf"],
    "correlation": ["rdf", "velocity_autocorr", "dipole_autocorr"],
    "coordination": ["coordination"],
    "neighbor": ["coordination", "rdf"],
    "contact": ["coordination", "hbond"],
    "hydrogen": ["hbond"],
    "bond": ["hbond"],

    # Density/Spatial
    "density": ["density_map", "conductivity"],
    "profile": ["density_profile"],
    "concentration": ["density_map", "local_concentration"],
    "spatial": ["density_map", "spatial_distribution"],
    "map": ["density_map"],
    "grid": ["density_map"],

    # Energy
    "energy": ["energy"],
    "kinetic": ["energy"],
    "potential_energy": ["energy"],

    # Time series/properties
    "temperature": ["temperature", "energy"],
    "pressure": ["energy"],
    "volume": ["volume"],

    # Docking/Binding
    "docking": ["docking"],
    "binding": ["docking"],
    "pose": ["docking"],
    "ligand": ["docking"],

    # Misc
    "cluster": ["clustering"],
    "cluster_analysis": ["clustering"],
    "path": ["path_cv"],
    "collective": ["path_cv"],
    "reaction": ["path_cv"],
}


@dataclass
class SuggestionCandidate:
    """A single analysis suggestion."""
    name: str
    reason: str  # Why this analysis was suggested
    missing_fields: List[str]  # Required fields not yet provided
    score: float = 0.0  # Match score for ranking


@dataclass
class SuggestionResult:
    """Result of an analysis suggestion query."""
    candidates: List[SuggestionCandidate]
    goal: str
    total_analyses: int


def suggest_analyses(
    goal: str,
    *,
    provided_fields: Optional[List[str]] = None,
    top_n: int = 5,
) -> SuggestionResult:
    """Suggest analyses based on a natural language goal.

    Uses deterministic keyword matching against analysis tags, descriptions,
    and field names. No LLM or fuzzy matching - simple scoring rules.

    Args:
        goal: Natural language description of what the user wants to compute
        provided_fields: Optional list of field names already provided
        top_n: Maximum number of suggestions to return

    Returns:
        SuggestionResult with ranked candidates
    """
    provided_fields = provided_fields or []
    goal_lower = goal.lower()

    # Tokenize goal into words (remove punctuation)
    import re
    words = set(re.findall(r'\b\w+\b', goal_lower))

    scored: List[tuple[str, float, str]] = []

    for name, contract in ANALYSIS_METADATA.items():
        score = 0.0
        reasons = []

        # Check direct name/alias matches
        if name in goal_lower:
            score += 10.0
            reasons.append(f"name match: {name}")
        for alias in contract.aliases:
            if alias in goal_lower or alias.replace("-", "_") in goal_lower:
                score += 8.0
                reasons.append(f"alias match: {alias}")

        # Check tag matches
        for tag in contract.tags:
            if tag in goal_lower:
                score += 5.0
                reasons.append(f"tag match: {tag}")

        # Check description keyword matches
        desc_lower = contract.description.lower()
        for word in words:
            if word in desc_lower and len(word) > 3:
                score += 1.0
        if contract.description and any(w in desc_lower for w in words if len(w) > 3):
            reasons.append("description match")

        # Check keyword mapping
        for word in words:
            if word in _GOAL_KEYWORDS:
                for candidate in _GOAL_KEYWORDS[word]:
                    if candidate == name or candidate in contract.aliases:
                        score += 3.0
                        reasons.append(f"keyword match: {word}")

        # Check field name matches
        for field_name in contract.field_types.keys():
            if field_name in goal_lower:
                score += 2.0
                reasons.append(f"field match: {field_name}")

        # Check output kind matches
        for output in contract.outputs:
            if output.kind in goal_lower:
                score += 2.0
                reasons.append(f"output kind: {output.kind}")

        if score > 0:
            # Deduplicate reasons
            unique_reasons = list(dict.fromkeys(reasons))
            scored.append((name, score, ", ".join(unique_reasons)))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Build candidates
    candidates = []
    for name, score, reason in scored[:top_n]:
        contract = ANALYSIS_METADATA[name]

        # Determine missing required fields
        missing = []
        for field in contract.required_fields:
            if field not in provided_fields:
                missing.append(field)

        candidates.append(SuggestionCandidate(
            name=name,
            reason=reason,
            missing_fields=missing,
            score=score,
        ))

    return SuggestionResult(
        candidates=candidates,
        goal=goal,
        total_analyses=len(ANALYSIS_METADATA),
    )


__all__ = [
    "ANALYSIS_METADATA",
    "get_plan_schema",
    "list_all_plans",
    "validate_request",
    "normalize_request",
    "generate_template",
    "capabilities",
    "lint_selection",
    "suggest_analyses",
    "ValidationResult",
    "ValidationErrorDetail",
    "SelectionLintResult",
    "SuggestionResult",
    "SuggestionCandidate",
    "FieldSpec",
    "ArtifactSpec",
    "AnalysisContract",
]
