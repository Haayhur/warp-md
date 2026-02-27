---
description: >-
  96 analysis Plans. Rust core. GPU-optional. Agent-first.
  The largest open-source MD analysis toolkit designed for autonomous AI agents.
icon: rocket
layout:
  width: wide
  title:
    visible: true
  description:
    visible: true
  tableOfContents:
    visible: true
  outline:
    visible: false
  pagination:
    visible: true
---

# warp-md

**Your agent's Swiss Army knife for molecular simulation — now with 96 blades.**

Whether you're an AI agent orchestrating a drug discovery pipeline or a researcher who just wants answers without writing a Fortran thesis, warp-md has your back. Load trajectories, select atoms, and run **96 analysis Plans** — from basic Rg to GIST solvation thermodynamics — all with a unified API that speaks JSON fluently.

{% hint style="info" %}
**One language, zero surprises.** All units are Angstrom internally. GRO/XTC files? Auto-converted from nm. Pydantic-validated JSON schemas for every request and response. Your agent can focus on science, not plumbing.
{% endhint %}

---

## Choose an Installation Path

Pick one path and move to [Quick Start](getting-started/quickstart.md):

### Path A: Install from PyPI

```bash
pip install warp-md
python -c "import warp_md; print(warp_md.System)"
```

Optional CUDA wheel:

```bash
pip install warp-md-cuda
```

### Path B: Build from Source (git clone)

```bash
git clone https://github.com/Haayhur/warp-md.git
cd warp-md
pip install maturin
maturin develop
python -c "import warp_md; print(warp_md.System)"
```

For full details, see [Installation](getting-started/installation.md).

---

## What Your Agent Can Do

<table data-view="cards">
    <thead>
        <tr>
            <th>Capability</th>
            <th data-card-target data-type="content-ref">Link</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>📊 96 Analysis Plans</strong><br>Rg, RMSD, PCA, GIST, DSSP, RDF, MSD, surface area, NMR, density profiles — the whole buffet, dessert included</td>
            <td><a href="reference/analysis-plans.md">Analysis Arsenal</a></td>
        </tr>
        <tr>
            <td><strong>🤖 Agent-Native Contract</strong><br>Pydantic schemas, JSON envelopes, NDJSON streaming, deterministic exit codes — built for machine consumption</td>
            <td><a href="reference/agent-schema.md">Agent Schema</a></td>
        </tr>
        <tr>
            <td><strong>🚀 GPU Turbo Mode</strong><br>Optional CUDA kernels for when CPU just isn't fast enough</td>
            <td><a href="guides/cuda-gpu.md">CUDA Guide</a></td>
        </tr>
        <tr>
            <td><strong>📦 World Building</strong><br>Pack molecules into simulation boxes with warp-pack. 10+ input formats.</td>
            <td><a href="guides/packing.md">Packing Guide</a></td>
        </tr>
        <tr>
            <td><strong>🔬 Trajectory Surgery</strong><br>Align, image, strip, center, superpose, replicate — transform trajectories without re-running simulations</td>
            <td><a href="reference/transforms.md">Transforms</a></td>
        </tr>
        <tr>
            <td><strong>🧬 Full Structural Suite</strong><br>PCA, DSSP, dihedrals, pucker, NMR order parameters, covariance matrices — deep structural insight</td>
            <td><a href="reference/structural-analysis.md">Structural Analysis</a></td>
        </tr>
    </tbody>
</table>

---

## First Analysis — 5 Lines, No PhD Required

```python
from warp_md import System, Trajectory, RgPlan

system = System.from_pdb("topology.pdb")
traj = Trajectory.open_xtc("trajectory.xtc", system)
rg = RgPlan(system.select("protein")).run(traj, system)
print(f"Mean Rg: {rg.mean():.2f} Å")  # Done. That was easy.
```

---

## The Full Arsenal at a Glance

| Category | Plans | Highlights |
|----------|-------|------------|
| **Core metrics** | Rg, RMSD, MSD, RDF | The fundamentals — GPU-accelerated |
| **RMSD variants** | Symmetric, distance-based, pairwise, per-residue | Every flavor of structural deviation |
| **Fluctuations** | RMSF, B-factors, atomic fluctuations | Crystallography meets dynamics |
| **PCA & modes** | PCA, mode analysis, projection | Collective motions, visualized |
| **Geometry** | Distance, angle, dihedral, COM, principal axes | Measure anything between atoms |
| **Dihedrals** | Single, multi, permute, pucker, torsion diffusion | Conformational analysis from backbone to sugars |
| **Solvation** | GIST, water count, watershell, density, volmap | Where water goes and why it matters |
| **Surface** | LCPO surface area, molecular surface | Exposed and buried surface |
| **Transport** | Conductivity, dielectric, dipole alignment, rotational ACF | Ionic liquid and electrolyte properties |
| **Polymer** | End-to-end, contour length, persistence length, chain Rg | Polymer physics toolkit |
| **NMR** | iRED vectors, order parameters, J-coupling | NMR observables from trajectories |
| **Transforms** | Align, center, image, strip, superpose, replicate | Trajectory preprocessing |
| **Correlation** | Atomic correlation, velocity ACF, cross-correlation, wavelet | Time-series analysis |
| **Contacts** | H-bonds, native contacts, minimum distance, Hausdorff | Interaction analysis |
| **DSSP** | Secondary structure assignment | Helix/sheet/coil per frame |
| **Energy** | Energy analysis, LIE, decomposition | Thermodynamic analysis |

---

## Explore the Docs

| Section | What You'll Find |
|---------|------------------|
| [Getting Started](getting-started/installation.md) | From zero to simulation in minutes |
| [Tutorial](tutorial/README.md) | The complete learning path |
| [Agent Schema](reference/agent-schema.md) | The contract your agent signs |
| [Guides](guides/packing.md) | Deep dives for specific workflows |
| [API Reference](reference/README.md) | Every class, every parameter |

<a href="getting-started/quickstart.md" class="button primary" data-icon="bolt">Quick Start →</a>
