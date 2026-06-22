---
description: The complete learning path for agents and their humans
icon: graduation-cap
---

# Tutorial

Welcome to the warp-md tutorial — where your agent learns to analyze molecular simulations like a seasoned computational chemist.

{% hint style="info" %}
**Estimated time**: 30-45 minutes for the full journey. Your agent will emerge enlightened.
{% endhint %}

---

## The Learning Path

{% stepper %}
{% step %}
## [Loading Systems & Trajectories](loading-systems.md)

Your agent meets its first atoms. Load molecular systems from PDB/GRO and open DCD/XTC trajectories.

**Time**: ~5 minutes
{% endstep %}

{% step %}
## [Atom Selections](selections.md)

Master the selection language — your agent's way of pointing at specific atoms. Boolean logic included.

**Time**: ~5 minutes
{% endstep %}

{% step %}
## [Basic Analyses](basic-analyses.md)

The fundamentals: Rg, RMSD, MSD, and RDF. Four Plans that cover most use cases.

**Time**: ~10 minutes
{% endstep %}

{% step %}
## [Advanced Analyses](advanced-analyses.md)

Power moves: polymer metrics, rotational correlation, conductivity, dielectric properties, and more.

**Time**: ~15 minutes
{% endstep %}

{% step %}
## [CLI Usage](cli-usage.md)

The command-line interface: your agent's native language. JSON output, structured schemas, zero ambiguity.

**Time**: ~5 minutes
{% endstep %}

{% step %}
## [Polymer Building (warp-build)](warp-build.md)

Build polymers from monomers, validate geometry, hand off to packing.

**Time**: ~10 minutes
{% endstep %}

{% step %}
## [Molecule Packing (warp-pack)](warp-pack.md)

Solvate proteins, build boxes, add ions — the packing workflow.

**Time**: ~10 minutes
{% endstep %}

{% step %}
## [Coarse-Graining (warp-cg)](warp-cg.md)

Map molecules to Martini beads — SMILES, trajectories, and polymer sources.

**Time**: ~15 minutes
{% endstep %}

{% step %}
## [CG System Building (warp-cg build)](warp-cg-build.md)

Assemble CG membranes, solvent, ions — GROMACS-ready output.

**Time**: ~15 minutes
{% endstep %}

{% step %}
## [Advanced CG Building](warp-cg-build-advanced.md)

Asymmetric membranes, phase separation, holes, and patches — complex membrane architectures.

**Time**: ~20 minutes
{% endstep %}

{% step %}
## [Complex CG Systems](warp-cg-build-complex.md)

Stacked membranes, nanodiscs, ion gradients, solute flooding — full CG world building.

**Time**: ~20 minutes
{% endstep %}
{% endstepper %}

---

## Before You Begin

Make sure you have:

- [x] Installed warp-md ([Installation Guide](../getting-started/installation.md))
- [x] A sample PDB/GRO topology file
- [x] A sample DCD/XTC trajectory file

{% hint style="success" %}
**No test files?** Check the [Validation Guide](../development/validation.md) for instructions on downloading sample data. We've got you covered.
{% endhint %}

---

## Quick Reference Cheat Sheet

| Concept | How to Use It |
|---------|---------------|
| System loading | `System.from_file()` |
| Selections | `system.select("backbone")` |
| Trajectories | `Trajectory.open_dcd()`, `Trajectory.open_xtc()` |
| Analyses | `RgPlan`, `RmsdPlan`, `MsdPlan`, `RdfPlan`, ... |
| Device | `device="auto"` (GPU if available), `"cpu"`, `"cuda"` |



## See Also

| Section | What You'll Find |
|---------|-----------------|
| [Examples](../examples/README.md) | Copy-paste ready analysis, build, and pack examples |
| [Architecture](../architecture/README.md) | How warp-md works under the hood |
| [API Reference](../reference/README.md) | Every class, every parameter |
| [Agent Contract Helpers](../guides/agent-contract-helpers.md) | Validate, normalize, inspect inputs |
| [Guides](../guides/warp-build.md) | Full warp-build reference |
| [Guides](../guides/warp-cg.md) | Full warp-cg reference |
| [Guides](../guides/packing.md) | Full warp-pack reference |

---

<a href="loading-systems.md" class="button primary" data-icon="play">Start the Tutorial →</a>
