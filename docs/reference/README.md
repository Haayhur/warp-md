---
description: The complete agent manual — 96 Plans, 88 functional APIs, one unified contract
icon: book
---

# API Reference

Everything your agent needs to know about warp-md — every class, every parameter, every option. This isn't "20+ analyses." It's **96 Plan classes** and **88 functional APIs** organized into clear categories.

---

## Quick Navigation

<table data-view="cards">
    <thead>
        <tr>
            <th>Section</th>
            <th data-card-target data-type="content-ref">Link</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Core Concepts</strong><br>System, Trajectory, Selection, Grouping, Device</td>
            <td><a href="core-concepts.md">Core Concepts</a></td>
        </tr>
        <tr>
            <td><strong>Agent Schema & Contract</strong><br>Pydantic schemas, JSON envelopes, NDJSON streaming</td>
            <td><a href="agent-schema.md">Agent Schema</a></td>
        </tr>
        <tr>
            <td><strong>Analysis Plans</strong><br>Rg, RMSD, MSD, RDF — the core 4</td>
            <td><a href="analysis-plans.md">Analysis Plans</a></td>
        </tr>
        <tr>
            <td><strong>Structural Analysis</strong><br>PCA, DSSP, RMSD variants, fluctuations, dihedrals, covariance</td>
            <td><a href="structural-analysis.md">Structural Analysis</a></td>
        </tr>
        <tr>
            <td><strong>Geometry & Distance</strong><br>Distances, angles, COM, neighbor search, native contacts</td>
            <td><a href="geometry-distance.md">Geometry & Distance</a></td>
        </tr>
        <tr>
            <td><strong>Solvation & Density</strong><br>GIST, water shells, density, surface area, volmaps</td>
            <td><a href="solvation-density.md">Solvation & Density</a></td>
        </tr>
        <tr>
            <td><strong>Transforms & Imaging</strong><br>Align, center, image, strip, superpose, replicate</td>
            <td><a href="transforms.md">Transforms</a></td>
        </tr>
        <tr>
            <td><strong>NMR & Spectroscopy</strong><br>iRED, order parameters, J-coupling, IR spectra, lipid SCD</td>
            <td><a href="nmr-spectroscopy.md">NMR & Spectroscopy</a></td>
        </tr>
        <tr>
            <td><strong>Polymer Plans</strong><br>End-to-end, persistence length, chain Rg, distributions</td>
            <td><a href="polymer-plans.md">Polymer Plans</a></td>
        </tr>
        <tr>
            <td><strong>Advanced Plans</strong><br>RotAcf, conductivity, dielectric, ion-pair, structure factor</td>
            <td><a href="advanced-plans.md">Advanced Plans</a></td>
        </tr>
        <tr>
            <td><strong>Builder Helpers</strong><br>Charges, group types, selection utilities</td>
            <td><a href="builder-helpers.md">Builder Helpers</a></td>
        </tr>
        <tr>
            <td><strong>CLI Reference</strong><br>Command-line interface, config runner, streaming</td>
            <td><a href="cli.md">CLI Reference</a></td>
        </tr>
    </tbody>
</table>

---

## Plan Coverage by Category

| Category | Plans | Doc Page |
|----------|-------|----------|
| Core metrics (Rg, RMSD, MSD, RDF) | 4 | [Analysis Plans](analysis-plans.md) |
| RMSD variants + fluctuations | 7 | [Structural Analysis](structural-analysis.md) |
| PCA, modes, covariance | 5 | [Structural Analysis](structural-analysis.md) |
| Dihedrals & ring pucker | 9 | [Structural Analysis](structural-analysis.md) |
| Geometry & distance | 12 | [Geometry & Distance](geometry-distance.md) |
| Neighbor search & contacts | 6 | [Geometry & Distance](geometry-distance.md) |
| Solvation, GIST, density | 7 | [Solvation & Density](solvation-density.md) |
| Surface area | 2 | [Solvation & Density](solvation-density.md) |
| Transforms & imaging | 14 | [Transforms & Imaging](transforms.md) |
| NMR, correlation, spectroscopy | 5 | [NMR & Spectroscopy](nmr-spectroscopy.md) |
| Transport properties | 6 | [Advanced Plans](advanced-plans.md) |
| Polymer | 6 | [Polymer Plans](polymer-plans.md) |
| Diffusion (torsion/toroidal) | 2 | [Structural Analysis](structural-analysis.md) |
| Structure validation | 2 | [Structural Analysis](structural-analysis.md) |
| Thermodynamic (equipartition) | 1 | [Advanced Plans](advanced-plans.md) |
| H-bond | 1 | [Advanced Plans](advanced-plans.md) |
| **Total** | **96** | |

---

## Rust Crates

Under the hood, warp-md is a family of Rust crates:

| Crate | What It Does |
|-------|--------------|
| `traj-core` | System topology, selection engine, boxes, errors (`#![forbid(unsafe_code)]`) |
| `traj-io` | PDB/GRO topology, DCD/XTC trajectories |
| `traj-engine` | Plans, executor, CPU fallback, device dispatch |
| `traj-gpu` | CUDA context + buffers (optional feature) |
| `traj-kernels` | CUDA kernels (compiled with nvrtc) |
| `traj-py` | Python bindings (PyO3, 96 Plan classes) |
| `warp-pack` | CPU packing engine + IO for initial coordinates |
| `warp-pep` | Peptide builder + mutation engine (internal coordinate geometry) |

---

## Units Convention

{% hint style="info" %}
All units are **Angstrom** internally unless noted otherwise. Your agent doesn't need to worry about unit conversions.
{% endhint %}

| Format | Native Unit | What warp-md Does |
|--------|-------------|-------------------|
| PDB | Angstrom | Nothing (already correct) |
| GRO | nm | × 10 → Angstrom |
| DCD | Angstrom* | Optional `length_scale` |
| XTC | nm | × 10 → Angstrom |

\* Some DCD files use nm — set `length_scale=10.0` to convert.

---

## Two API Styles

warp-md offers both **Plan-based** and **functional** APIs:

{% tabs %}
{% tab title="Plan API (Recommended for Agents)" %}
```python
from warp_md import RgPlan

plan = RgPlan(selection, mass_weighted=False)
result = plan.run(traj, system, device="auto")
```

Plans are composable, GPU-aware, and return NumPy arrays.
{% endtab %}

{% tab title="Functional API" %}
```python
from warp_md.analysis import rmsf, dssp, pca

rmsf_values = rmsf(traj, system, selection)
ss_assignments = dssp(traj, system, selection)
pca_result = pca(traj, system, selection)
```

Functional APIs wrap Plans with convenient defaults. 88 functions available.
{% endtab %}
{% endtabs %}
