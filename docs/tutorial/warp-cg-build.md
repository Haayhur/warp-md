---
description: Assemble coarse-grained membranes, solvent, ions, and inserted components
icon: layers
---

# CG System Building (`warp-cg build`)

`warp-cg build` assembles coarse-grained systems from typed membrane, solvent,
ion, protein, and solute requests.

## Generate, Validate, Run

```bash
warp-cg build example > build_request.json
warp-cg build validate build_request.json
warp-cg build run build_request.json --stream ndjson
```

`validate` checks the same request contract used by `run`. Exit code `0` means
the request is valid; exit code `2` means contract or preflight validation
failed.

## Minimal Membrane Request

This is the shape emitted by `warp-cg build example`:

```json
{
  "schema_version": "warp-cg.build.v1",
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
  "membranes": [
    {
      "name": "bilayer",
      "center_z_angstrom": 0.0,
      "leaflets": [
        {
          "name": "upper",
          "side": "upper",
          "apl_angstrom2": 64.0,
          "composition": [
            {"lipid": "POPC", "count": 64},
            {"lipid": "POPG", "count": 16}
          ],
          "exclusions": [
            {
              "name": "protein-footprint",
              "center_angstrom": [0.0, 0.0],
              "radius_angstrom": 10.0
            }
          ],
          "regions": [
            {
              "name": "inspection-hole",
              "role": "hole",
              "geometry": {
                "shape": "circle",
                "center_angstrom": [24.0, 0.0],
                "radius_angstrom": 8.0
              }
            }
          ]
        },
        {
          "name": "lower",
          "side": "lower",
          "apl_angstrom2": 64.0,
          "composition": [{"lipid": "POPC", "count": 80}]
        }
      ]
    }
  ],
  "proteins": [],
  "environment": {
    "ions": {
      "neutralize": true,
      "cation": "Na+",
      "anion": "Cl-"
    },
    "solvent": {"enabled": true}
  },
  "outputs": {
    "coordinates": "outputs/membrane.gro",
    "topology": "outputs/topol.top",
    "manifest": "outputs/membrane_manifest.json"
  }
}
```

## Contract Map

| Field | Purpose |
|-------|---------|
| `system.box_size_angstrom` | Required `[x, y, z]` box dimensions |
| `system.pbc` | Periodicity mode string; omit for the schema default |
| `membranes[]` | Bilayers or monolayers at explicit Z positions |
| `leaflets[].composition` | Lipid counts or fractions for one leaflet |
| `leaflets[].exclusions` | Circular lipid exclusion zones |
| `leaflets[].regions` | Geometric `hole` or `patch` placement constraints |
| `proteins[]` / `solutes[]` | Inserted CG components |
| `environment.ions` | Neutralization and bulk salt policy |
| `environment.solvent` | Solvent species, density, and optional zones |
| `outputs` | Coordinates, topology, log, snapshot, and manifest paths |

`patch` constrains the whole leaflet placement area; it does not carry an
independent lipid composition. Lipid composition is defined on the containing
leaflet.

## Geometry Forms

```json
{
  "regions": [
    {
      "name": "circle-patch",
      "role": "patch",
      "geometry": {
        "shape": "circle",
        "center_angstrom": [0.0, 0.0],
        "radius_angstrom": 35.0
      }
    },
    {
      "name": "elliptical-hole",
      "role": "hole",
      "geometry": {
        "shape": "ellipse",
        "center_angstrom": [10.0, 0.0],
        "radius_angstrom": [8.0, 14.0],
        "rotate_degrees": 20.0
      }
    }
  ]
}
```

Rectangle geometry uses `size_angstrom`; polygon geometry uses
`points_angstrom`. See the live schema for every field:

```bash
warp-cg build schema --kind request
```

## Forcefield Management

```bash
warp-cg forcefield inspect --kind martini3
warp-cg forcefield install --kind martini3 --dest forcefields/martini3
```

The build contract selects the forcefield with
`system.force_field: "martini3"`. Forcefield installation materializes the
bundled snapshot for downstream simulation tools.

## Output Artifacts

| Field | Typical Artifact |
|-------|------------------|
| `outputs.coordinates` | `.gro`, `.pdb`, or `.cif` coordinates |
| `outputs.topology` | Top-level GROMACS `.top` |
| `outputs.manifest` | Machine-readable build manifest |
| `outputs.log` | Optional text build log |
| `outputs.snapshot` | Optional normalized request snapshot |

## Next

- [Advanced CG Building](warp-cg-build-advanced.md)
- [Complex CG Systems](warp-cg-build-complex.md)
- [CG Mapping](warp-cg.md)
