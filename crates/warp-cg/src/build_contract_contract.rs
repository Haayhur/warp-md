use anyhow::{anyhow, Result};
use schemars::schema_for;
use serde_json::{json, Value};

use crate::build_lipids::known_lipids;
use crate::build_solutes::known_solute_names;

use super::{
    default_solvent_mapping_ratio, default_solvent_molarity, known_anion_library_names,
    known_cation_library_names, known_solvent_library_names, BuildEvent, BuildRequest, BuildResult,
    BUILD_SCHEMA_VERSION,
};

pub fn schema_json(kind: &str) -> Result<String> {
    let schema = match kind {
        "request" => serde_json::to_value(schema_for!(BuildRequest))?,
        "result" => serde_json::to_value(schema_for!(BuildResult))?,
        "event" => serde_json::to_value(schema_for!(BuildEvent))?,
        other => return Err(anyhow!("unknown warp-cg build schema kind: {other}")),
    };
    Ok(serde_json::to_string_pretty(&schema)?)
}

pub fn example_request() -> Value {
    json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "run_id": "membrane-001",
        "mode": "membrane",
        "system": {
            "force_field": "martini3",
            "box_size_angstrom": [120.0, 120.0, 140.0],
            "placement": {
                "relaxation": true,
                "max_steps": 100,
                "push_tolerance_angstrom": 0.01
            }
        },
        "membranes": [{
            "name": "bilayer",
            "center_z_angstrom": 0.0,
            "leaflets": [
                {
                    "name": "upper",
                    "side": "upper",
                    "apl_angstrom2": 64.0,
                    "exclusions": [{"name": "protein-footprint", "center_angstrom": [0.0, 0.0], "radius_angstrom": 10.0}],
                "regions": [{
                    "name": "inspection-hole",
                    "role": "hole",
                        "geometry": {
                            "shape": "circle",
                            "center_angstrom": [24.0, 0.0],
                            "radius_angstrom": 8.0
                        }
                    }],
                    "composition": [
                        {"lipid": "POPC", "count": 64},
                        {"lipid": "POPG", "count": 16}
                    ]
                },
                {
                    "name": "lower",
                    "side": "lower",
                    "apl_angstrom2": 64.0,
                    "composition": [
                        {"lipid": "POPC", "count": 80}
                    ]
                }
            ]
        }],
        "proteins": [],
        "environment": {
            "ions": {"neutralize": true, "cation": "Na+", "anion": "Cl-"},
            "solvent": {"enabled": true}
        },
        "outputs": {
            "coordinates": "outputs/membrane.gro",
            "topology": "outputs/topol.top",
            "manifest": "outputs/membrane_manifest.json"
        }
    })
}

pub fn capabilities() -> Value {
    json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "tool": "warp-cg build",
        "schema_targets": ["request", "result", "event"],
        "modes": ["membrane"],
        "membrane": {
            "status": "basic_membrane_coordinates_and_charge",
            "reference_implementation": "external membrane builder",
            "implemented_now": [
                "agent-first request/result/event schemas",
                "protein/solute-only inserted-component builds without membranes",
                "APL-derived leaflet lipid counts",
                "asymmetric leaflet composition inputs",
                "deterministic rectangular leaflet placement",
                "optional seeded random membrane leaflet candidate placement",
                "deterministic pair and edge point relaxation",
                "per-leaflet placement quality metrics",
                "circular protein/hole exclusion masks",
                "typed leaflet regions for holes and patches: circle, ellipse, rectangle, polygon",
                "region-constrained lipid placement with in-loop relaxation projection and post-relaxation confinement",
                "exact axis-aligned rectangle region union area planning",
                "exact circle region union area planning, including clipped circle unions and unclipped circle/rotated-rectangle pairs",
                "exact ellipse region union area planning for disjoint clipped ellipses, ellipse/axis-aligned rectangle pairs, unclipped oriented-ellipse/rotated-rectangle pairs, clipped circle/axis-aligned ellipse pairs, and unclipped circle/oriented-ellipse pairs",
                "exact multiple-circle plus axis-aligned rectangle region union area planning",
                "exact multiple-circle plus multiple axis-aligned rectangles region union area planning",
                "exact circle or ellipse plus multiple axis-aligned rectangles region union area planning",
                "exact multiple mutually disjoint ellipses plus multiple axis-aligned rectangles region union area planning",
                "exact circle plus multiple mutually disjoint ellipses region union area planning",
                "exact ellipse plus multiple mutually disjoint circles region union area planning",
                "exact multiple mutually disjoint circles plus multiple convex polygons region union area planning",
                "exact circle plus mutually disjoint mixed secondary shapes region union area planning",
                "exact ellipse plus mutually disjoint mixed secondary shapes region union area planning",
                "exact rectangle plus mutually disjoint mixed secondary shapes region union area planning",
                "exact convex polygon plus mutually disjoint mixed secondary shapes region union area planning",
                "exact circle or ellipse plus multiple convex polygons region union area planning",
                "exact circle plus overlapping convex polygons and polygon-disjoint ellipses region union area planning",
                "exact ellipse plus overlapping convex polygons and polygon-disjoint circles region union area planning",
                "exact circle plus overlapping convex polygons and polygon-disjoint mixed secondary shapes region union area planning",
                "exact ellipse plus overlapping convex polygons and polygon-disjoint mixed secondary shapes region union area planning",
                "exact circle or ellipse plus multiple actual-disjoint simple non-convex polygons using exact pair-overlap proofs",
                "exact convex polygon/rotated-rectangle region union area planning",
                "exact simple polygon region union area planning",
                "exact rectangle plus simple non-convex polygon region union area planning",
                "exact mixed-shape region component partitioning using exact pair-disjoint proofs",
                "geo-backed polygonized boolean region union area planning for overlapping mixed curve/polygon regions before grid fallback",
                "adaptive mixed-shape region union area planning with reported error estimates",
                "conservative boundary-band error bounds for remaining grid-estimated mixed region unions",
                "leaflet area method and error metadata in placement metrics",
                "polygon scale_xy and rotate_degrees transforms",
                "membrane-local xy domains via center_xy_angstrom and size_xy_angstrom",
                "APL count planning from region-adjusted leaflet area",
                "membrane void solvation policy via membranes[].solvate_voids",
                "mixed solvent and mixed ion ratios with coarse mapping counts",
                "automatic circular exclusion masks from protein coordinates",
                "library-backed multi-bead Martini diacyl with LTF release tail tables, named LTF single-chain/monoglyceride/diglyceride/triglyceride, generated sphingomyelin, and sterol templates",
                "atomistic SOL water aliases for TIP3/TIP4/TIP5 residue-name output",
                "case-sensitive atomistic Na/Cl ion aliases while preserving Martini NA/CL names",
                "library-backed standard Martini 3 solvents including DMSO, ACN, alkanes, alkenes, alkynes, dienes, haloalkanes, alcohols, ethers, sulfides, ketones, aldehydes, esters, amines, carboxylic acids, and amides, reusable as coordinate-less inserted solutes",
                "format-aware coordinate emission via outputs.coordinates plus explicit outputs.gro/outputs.pdb/outputs.cif",
                "output overwrite and backup policy via outputs.overwrite and outputs.backup_existing",
                "orthorhombic unit-cell metadata in result box_meta",
                "resolved unit-cell and box-vector coordinate output from system box metadata",
                "explicit PBC metadata via system.pbc",
                "orthorhombic minimum-image placement exclusion checks for enabled system.pbc axes",
                "box-vector minimum-image placement exclusion checks for triclinic unit cells",
                "scaled triclinic XY membrane subdomain basis for explicit size_xy_angstrom deterministic leaflet placement",
                "scaled triclinic XY membrane subdomain basis for seeded random leaflet candidate placement",
                "scaled triclinic XY constrained leaflet patch and hole clipping for placement, projection, and diagnostics",
                "adaptive far-image triclinic wrapped-region queries for heavily clipped rotated polygon constraints",
                "Gromacs molecule-count topology emission",
                "structured build log emission via outputs.log",
                "machine-readable request/result snapshot emission via outputs.snapshot",
                "template/component charge accounting",
                "bead charge derivation from explicit bead charges or total charge spread",
                "Martini amino-acid and tutorial small-molecule solute library templates for coordinate-less flooding",
                "tail-code generated hydrocarbon and fatty-acid solvent/flood molecules",
                "library-backed Martini water, atomistic multi-site water, and single-bead ion defaults",
                "bead-charge balance deltas in result contract",
                "free-volume solvent count planning and deterministic W/ion emission",
                "solvent_per_lipid solvent count planning for stacked membrane solvent slabs",
                "multi-zone phase-separated solvation via environment.solvent.zones[]",
                "stacked membrane expansion via stacked_membranes[].layers[] with derived z-centers, solvent slabs, and box height",
                "nanodisc interior clipping via membranes[].protein_boundary derived from inserted protein coordinates",
                "convex-hull protein boundary clipping via membranes[].protein_boundary.geometry",
                "concave-hull protein boundary clipping via membranes[].protein_boundary.geometry",
                "alpha-shape protein boundary clipping, including disconnected component footprints, via membranes[].protein_boundary.geometry and alpha_radius_angstrom",
                "buffered non-circular protein boundary placement constraints via protein_boundary.buffer_angstrom",
                "exact buffered convex-polygon protein boundary area planning",
                "exact unbuffered overlapping simple multipolygon protein boundary area planning",
                "geo-backed overlapping multipolygon, nested-hole, and buffered polygon protein-boundary area planning",
                "multi-moleculetype inserted components via molecule_types[] and charge_topologies[]",
                "local Gromacs #include expansion for inserted-component charge topology parsing",
                "deterministic non-overlapping flood placement for repeated inserted solutes/components",
                "optional seeded placement mode for solvent and inserted-component candidate queues",
                "optional seeded random candidate source for solvent and inserted-component placement",
                "optional seeded_random orientation for inserted components",
                "bounded seeded local kick/retry expansion for dense solvent candidate queues",
                "bounded staggered grid-squeeze passes for dense solvent/ion candidate queues",
                "bounded seeded local kick/retry expansion for inserted-component flood centers",
                "bounded staggered grid-squeeze passes for dense inserted-component flood placement",
                "neutralization count planning",
                "salt neutralization methods: add, remove, mean"
            ],
            "planned": [
                "broader reference lipid/library coverage"
            ]
        },
        "known_solutes": known_solute_names(),
        "charge": {
            "core": "warp-common::charge",
            "sources": [
                "lipid_template.net_charge_e",
                "request_lipid.charge_e",
                "request_lipid.beads.charge_e",
                "single_bead_fallback.charge_e",
                "inserted_component.net_charge_e",
                "inserted_component.charge_topology.gromacs_atoms",
                "solute_template.net_charge_e"
            ],
            "charge_resolution_rule": "explicit bead charges define molecule charge; total charge overrides are spread over generated/template beads; inserted components can derive charge from GROMACS [ atoms ] or known solute templates; explicit bead lists or inserted charge totals with mismatched derived charge are rejected",
            "neutralization": "salt_method add inserts counterions, remove removes coions, mean splits the correction between insertion and removal"
        },
        "solvation": {
            "core_rule": "free volume = box volume - emitted bead sphere volumes; solvent count = round(Avogadro * free volume * molarity / mapping_ratio)",
            "defaults": {
                "solvent": "W",
                "solvent_molarity_mol_l": default_solvent_molarity(),
                "solvent_mapping_ratio": default_solvent_mapping_ratio(),
                "salt_molarity_mol_l": "environment.ions.salt_molarity_mol_l"
            },
            "mixed_species": "environment.solvent.species[] and environment.ions.cations[]/anions[] accept explicit ratios, mapping ratios, and charges",
            "known_solvents": known_solvent_library_names(),
            "generated_solvents": [
                "hydrocarbon:<tailcode>",
                "fattyacid:<tailcode>",
                "monoglyceride:<tailcode>",
                "diglyceride:<tailcode>,<tailcode>",
                "triglyceride:<tailcode>,<tailcode>,<tailcode>",
                "bmp2:<tailcode>,<tailcode>",
                "bmp3:<tailcode>,<tailcode>",
                "cardiolipin:<tailcode>,<tailcode>,<tailcode>,<tailcode>",
                "sphingolipid:<head>,<tailcode>,<tailcode>"
            ],
            "known_cations": known_cation_library_names(),
            "known_anions": known_anion_library_names(),
            "zones": "environment.solvent.zones[] inherit top-level solvent and ion defaults and can override center_angstrom, box_size_angstrom, molarity_mol_l, salt_molarity_mol_l, and species[] for phase-separated solvation"
        },
        "lipid_templates": {
            "force_fields": ["martini3", "default"],
            "known_lipids": known_lipids()
        }
    })
}
