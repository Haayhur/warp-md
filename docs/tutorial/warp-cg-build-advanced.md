---
description: Advanced CG building with asymmetric leaflets, holes, and constrained patches
icon: layers
---

# Advanced CG System Building

These examples use only fields accepted by `warp-cg.build.v1`.

## Asymmetric Bilayer

Different compositions are defined per leaflet:

```json
{
  "schema_version": "warp-cg.build.v1",
  "run_id": "asymmetric-bilayer",
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
      "name": "asymmetric",
      "center_z_angstrom": 0.0,
      "leaflets": [
        {
          "name": "upper",
          "side": "upper",
          "apl_angstrom2": 64.0,
          "composition": [
            {"lipid": "POPC", "count": 80},
            {"lipid": "POPE", "count": 40},
            {"lipid": "CHOL", "count": 20}
          ]
        },
        {
          "name": "lower",
          "side": "lower",
          "apl_angstrom2": 64.0,
          "composition": [
            {"lipid": "POPG", "count": 60},
            {"lipid": "DOPG", "count": 40},
            {"lipid": "CHOL", "count": 10}
          ]
        }
      ]
    }
  ],
  "environment": {
    "ions": {
      "neutralize": true,
      "salt_molarity_mol_l": 0.15,
      "cation": "Na+",
      "anion": "Cl-"
    },
    "solvent": {"enabled": true}
  },
  "outputs": {
    "coordinates": "outputs/asymmetric.gro",
    "topology": "outputs/asymmetric.top",
    "manifest": "outputs/asymmetric.json"
  }
}
```

## Holes and Protein Footprints

`exclusions` are circular. `regions` support circle, ellipse, rectangle, and
polygon geometry.

```json
{
  "schema_version": "warp-cg.build.v1",
  "run_id": "membrane-holes",
  "mode": "membrane",
  "system": {
    "box_size_angstrom": [120.0, 120.0, 140.0]
  },
  "membranes": [
    {
      "name": "bilayer",
      "leaflets": [
        {
          "name": "upper",
          "side": "upper",
          "composition": [{"lipid": "POPC", "count": 80}],
          "exclusions": [
            {
              "name": "protein-footprint",
              "center_angstrom": [0.0, 0.0],
              "radius_angstrom": 10.0
            }
          ],
          "regions": [
            {
              "name": "elliptical-pore",
              "role": "hole",
              "geometry": {
                "shape": "ellipse",
                "center_angstrom": [24.0, 0.0],
                "radius_angstrom": [8.0, 14.0],
                "rotate_degrees": 15.0
              }
            }
          ]
        },
        {
          "name": "lower",
          "side": "lower",
          "composition": [{"lipid": "POPC", "count": 80}]
        }
      ]
    }
  ],
  "environment": {
    "ions": {"neutralize": false},
    "solvent": {"enabled": true}
  },
  "outputs": {
    "coordinates": "outputs/holey.gro",
    "topology": "outputs/holey.top",
    "manifest": "outputs/holey.json"
  }
}
```

## Constrained Patch

A `patch` defines the allowed placement union for its leaflet:

```json
{
  "schema_version": "warp-cg.build.v1",
  "run_id": "disc-patch",
  "mode": "membrane",
  "system": {
    "box_size_angstrom": [120.0, 120.0, 100.0]
  },
  "membranes": [
    {
      "name": "disc",
      "leaflets": [
        {
          "name": "upper",
          "side": "upper",
          "regions": [
            {
              "name": "disc-boundary",
              "role": "patch",
              "geometry": {
                "shape": "circle",
                "center_angstrom": [0.0, 0.0],
                "radius_angstrom": 45.0
              }
            }
          ],
          "composition": [{"lipid": "DMPC", "count": 80}]
        },
        {
          "name": "lower",
          "side": "lower",
          "regions": [
            {
              "name": "disc-boundary",
              "role": "patch",
              "geometry": {
                "shape": "circle",
                "center_angstrom": [0.0, 0.0],
                "radius_angstrom": 45.0
              }
            }
          ],
          "composition": [{"lipid": "DMPC", "count": 80}]
        }
      ]
    }
  ],
  "environment": {
    "ions": {"neutralize": false},
    "solvent": {"enabled": true}
  },
  "outputs": {
    "coordinates": "outputs/disc.gro",
    "topology": "outputs/disc.top",
    "manifest": "outputs/disc.json"
  }
}
```

{% hint style="warning" %}
Regions do not contain a `composition` field. For multiple spatial lipid
compositions, use separate membrane/leaflet definitions or generate the desired
component layout upstream.
{% endhint %}

## Validate

```bash
warp-cg build validate request.json
warp-cg build run request.json --stream ndjson
```
