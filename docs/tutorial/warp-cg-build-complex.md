---
description: Complex CG systems with membrane stacks, solvent zones, and inserted solutes
icon: layers
---

# Complex CG Systems

## Stacked Membranes

Use `stacked_membranes` when the inter-layer distances and solvent gaps should
be normalized by the builder:

```json
{
  "schema_version": "warp-cg.build.v1",
  "run_id": "stacked-bilayers",
  "mode": "membrane",
  "system": {
    "box_size_angstrom": [200.0, 100.0, 1.0]
  },
  "membranes": [],
  "stacked_membranes": [
    {
      "name": "three-bilayers",
      "pbc": "split",
      "distance_angstrom": [40.0],
      "distance_type": ["surface"],
      "layers": [
        {
          "membrane": {
            "name": "outer-a",
            "leaflets": [
              {
                "name": "upper",
                "side": "upper",
                "composition": [
                  {"lipid": "POPC", "fraction": 5.0},
                  {"lipid": "CHOL", "fraction": 1.0}
                ]
              },
              {
                "name": "lower",
                "side": "lower",
                "composition": [
                  {"lipid": "POPC", "fraction": 5.0},
                  {"lipid": "CHOL", "fraction": 1.0}
                ]
              }
            ]
          },
          "solvent": {"name": "pbc-gap"}
        },
        {
          "membrane": {
            "name": "middle",
            "leaflets": [
              {
                "name": "upper",
                "side": "upper",
                "composition": [{"lipid": "POPE", "fraction": 1.0}]
              },
              {
                "name": "lower",
                "side": "lower",
                "composition": [{"lipid": "POPE", "fraction": 1.0}]
              }
            ]
          },
          "solvent": {"name": "middle-gap"}
        }
      ]
    }
  ],
  "environment": {
    "ions": {"neutralize": false},
    "solvent": {"enabled": false, "molarity_mol_l": 0.0}
  },
  "outputs": {
    "coordinates": "outputs/stacked.gro",
    "topology": "outputs/stacked.top",
    "manifest": "outputs/stacked.json"
  }
}
```

The normalizer expands layers into `membranes`, creates solvent zones, and
updates the box height.

## Phase-Separated Solvent Zones

Zones are axis-aligned boxes defined by `center_angstrom` and
`box_size_angstrom`:

```json
{
  "schema_version": "warp-cg.build.v1",
  "run_id": "solvent-zones",
  "mode": "membrane",
  "system": {
    "box_size_angstrom": [80.0, 80.0, 80.0]
  },
  "membranes": [
    {
      "name": "bilayer",
      "leaflets": [
        {
          "name": "upper",
          "side": "upper",
          "composition": [{"lipid": "POPC", "count": 32}]
        },
        {
          "name": "lower",
          "side": "lower",
          "composition": [{"lipid": "POPC", "count": 32}]
        }
      ]
    }
  ],
  "environment": {
    "ions": {
      "neutralize": true,
      "salt_molarity_mol_l": 0.0,
      "cation": "Na+",
      "anion": "Cl-"
    },
    "solvent": {
      "enabled": true,
      "zones": [
        {
          "name": "right",
          "center_angstrom": [20.0, 0.0, 0.0],
          "box_size_angstrom": [40.0, 80.0, 80.0],
          "salt_molarity_mol_l": 0.05
        },
        {
          "name": "left",
          "center_angstrom": [-20.0, 0.0, 0.0],
          "box_size_angstrom": [40.0, 80.0, 80.0],
          "salt_molarity_mol_l": 0.15
        }
      ]
    }
  },
  "outputs": {
    "coordinates": "outputs/zones.gro",
    "topology": "outputs/zones.top",
    "manifest": "outputs/zones.json"
  }
}
```

Ion species are configured by `environment.ions`; individual zones set their
own `salt_molarity_mol_l`.

## Mixed Solvent Species

```json
{
  "enabled": true,
  "species": [
    {"name": "W", "ratio": 0.8},
    {"name": "ETH", "ratio": 0.15},
    {"name": "DMSO", "ratio": 0.05}
  ]
}
```

Species names must resolve through the bundled solvent registry or provide the
required physical parameters accepted by the schema.

## Inserted Solutes

`solutes` is a top-level array. Known built-in component names can be inserted
without coordinate files:

```json
{
  "schema_version": "warp-cg.build.v1",
  "run_id": "inserted-solutes",
  "mode": "membrane",
  "system": {
    "box_size_angstrom": [80.0, 80.0, 80.0]
  },
  "membranes": [
    {
      "name": "bilayer",
      "leaflets": [
        {
          "name": "upper",
          "side": "upper",
          "composition": [{"lipid": "POPC", "count": 32}]
        },
        {
          "name": "lower",
          "side": "lower",
          "composition": [{"lipid": "POPC", "count": 32}]
        }
      ]
    }
  ],
  "solutes": [
    {"name": "ARG", "count": 2},
    {"name": "GLY", "count": 3},
    {"name": "TYR", "count": 1}
  ],
  "environment": {
    "ions": {
      "neutralize": true,
      "salt_molarity_mol_l": 0.0
    },
    "solvent": {"enabled": true}
  },
  "outputs": {
    "coordinates": "outputs/inserted.gro",
    "topology": "outputs/inserted.top",
    "manifest": "outputs/inserted.json"
  }
}
```

Custom components can use `coordinates`, `definition`, explicit `beads`, and
typed placement fields. Inspect the live schema before constructing a custom
component:

```bash
warp-cg build schema --kind request
```

## Pipeline

```bash
warp-cg build validate complex_request.json
warp-cg build run complex_request.json --stream ndjson
warp-cg simulate plan simulation_request.json --engine gromacs
```
