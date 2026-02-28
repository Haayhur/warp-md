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

The fundamentals: Rg, RMSD, MSD, and RDF. Four Plans that cover most use cases. (There are 92 more where those came from.)

**Time**: ~10 minutes
{% endstep %}

{% step %}
## [Advanced Analyses](advanced-analyses.md)

Power moves: polymer metrics, rotational correlation, conductivity, dielectric properties, and 77 more Plans in the full API. For when your agent needs to show off.

**Time**: ~15 minutes
{% endstep %}

{% step %}
## [CLI Usage](cli-usage.md)

The command-line interface: your agent's native language. JSON output, structured schemas, zero ambiguity.

**Time**: ~5 minutes
{% endstep %}
{% endstepper %}

---

## Before You Begin

Make sure you have:

- [x] Installed warp-md ([Installation Guide](../getting-started/installation.md))
- [x] A sample PDB/GRO topology file
- [x] A sample DCD/XTC trajectory file

{% hint style="success" %}
**No test files?** Check the [Validation Guide](../guides/validation.md) for instructions on downloading sample data. We've got you covered.
{% endhint %}

---

## Quick Reference Cheat Sheet

| Concept | How to Use It |
|---------|---------------|
| System loading | `System.from_pdb()`, `System.from_gro()` |
| Selections | `system.select("backbone")` |
| Trajectories | `Trajectory.open_dcd()`, `Trajectory.open_xtc()` |
| Analyses | `RgPlan`, `RmsdPlan`, `MsdPlan`, `RdfPlan`, ... |
| Device | `device="auto"` (GPU if available), `"cpu"`, `"cuda"` |

---

<a href="loading-systems.md" class="button primary" data-icon="play">Start the Tutorial →</a>
