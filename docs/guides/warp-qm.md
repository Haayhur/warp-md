---
description: Quantum mechanics orchestration for agent workflows
icon: atom
---

# QM Orchestration (warp-qm)

When classical force fields start hallucinating chemistry or you need actual electronic structure properties (or at least a very good semi-empirical guess), `warp-qm` is your agent's quantum mechanics engine room. 

It manages the messy world of external QM inputs, coordinates, basis sets, and charge fits, presenting a neat, schema-validated JSON interface.

{% hint style="info" %}
`warp-qm` is a native orchestration tool. Rust handles the contracts, schemas, event streams, and result envelopes; the actual quantum chemistry is delegated to external backends like ORCA, Multiwfn, Psi4, or xTB.
{% endhint %}

---

## Engine Setup

Because we value your disk space, external binaries are not bundled with `warp-md`. You must tell the agent where to find them using environment variables:

```bash
# Point to your ORCA and Multiwfn installations
export WARP_QM_ORCA=/path/to/orca
export WARP_QM_ORCA_2MKL=/path/to/orca_2mkl
export WARP_QM_MULTIWFN=/path/to/Multiwfn
export WARP_QM_MULTIWFN_LIB_DIR=/path/to/multiwfn/lib
```

### The Readiness Check

Before submitting a heavy calculation, run the diagnostic tools to make sure everything is wired up:

{% tabs %}
{% tab title="CLI" %}
```bash
# Check what backends are visible and configured
warp-qm capabilities --json

# Run a full diagnostic check on your environment
warp-qm doctor --json \
  --orca-executable /path/to/orca \
  --multiwfn-executable /path/to/Multiwfn \
  --threads 1
```
{% endtab %}

{% tab title="Python" %}
```python
from warp_md import qm

# Probe system capabilities
caps = qm.capabilities()
print("Available engines:", caps["supported_engines"])

# Run doctor diagnosis
doctor_report = qm.doctor_json(
    orca_executable="/path/to/orca",
    multiwfn_executable="/path/to/Multiwfn"
)
print("Is RESP2 workflow ready?", doctor_report["ready"]["resp2_workflow"])
```
{% endtab %}
{% endtabs %}

---

## Generic Calculations

Use `generic_run` tasks when your agent needs raw, unadulterated control over the underlying engine inputs.

### ORCA Custom Input

Submit a specific input template with coordinates resolved at runtime:

```json
{
  "schema_version": "warp-qm.agent.v1",
  "request_id": "custom-orca-opt",
  "engine": {
    "name": "orca",
    "settings": {
      "input_file": "job.inp",
      "export_molden": true
    }
  },
  "molecule": {
    "source": {"kind": "engine_input", "format": "orca"},
    "charge": 0,
    "multiplicity": 1
  },
  "task": {"kind": "generic_run", "method": "custom"}
}
```

### Multiwfn Analysis

Run a custom script menu against a wavefunction file:

```json
{
  "schema_version": "warp-qm.agent.v1",
  "request_id": "custom-multiwfn-esp",
  "engine": {
    "name": "multiwfn",
    "settings": {
      "input_file": "wavefunction.fchk",
      "menu_script": "7\n18\n1\ny\n0\n0\nq\n",
      "expected_outputs": ["wavefunction.chg"]
    }
  },
  "molecule": {
    "source": {"kind": "file", "path": "wavefunction.fchk", "format": "fchk"},
    "charge": 0,
    "multiplicity": 1
  },
  "task": {"kind": "generic_run", "method": "multiwfn"}
}
```

---

## RESP & RESP2 Charge Fitting

For force field parameterization, Restrained Electrostatic Potential (RESP) charges are the gold standard. `warp-qm` makes RESP and RESP2 workflows painless.

- **RESP**: Multiwfn runs a two-stage fit against electrostatic potential on grid points.
- **RESP2**: Optimizes/runs single points in both gas phase and solvent phase, calculates gas and solvent RESP charges, then mixes them:
  $$q_{\text{RESP2}} = (1 - \delta) \cdot q_{\text{gas}} + \delta \cdot q_{\text{solvent}}$$
  *(Typically, $\delta = 0.5$ for neutral systems).*

### The High-Level RESP2 Workflow

Instead of writing input files, parsing logs, and writing scripts manually, define a workflow request:

```json
{
  "schema_version": "warp-qm.agent.v1",
  "request_id": "ethanol-resp2",
  "engine": {
    "name": "workflow",
    "settings": {
      "qm_engine": "orca",
      "fit_engine": "multiwfn",
      "gas": {"method": "HF", "basis": "6-31G(d)", "keywords": []},
      "solution": {"method": "HF", "basis": "6-31G(d)", "keywords": ["CPCM(Water)"]},
      "resp2": {"delta": 0.5}
    }
  },
  "molecule": {
    "source": {"kind": "file", "path": "ethanol.xyz", "format": "xyz"},
    "charge": 0,
    "multiplicity": 1
  },
  "task": {"kind": "resp2_workflow", "method": "HF", "basis": "6-31G(d)", "charge_model": "resp2"},
  "runtime": {"work_dir": "results/qm/resp2"}
}
```

Run it via the CLI:
```bash
warp-qm run resp2_request.json --stream
```

This automates:
1. Gas-phase geometry optimization & ESP generation
2. Solvated-phase geometry optimization & ESP generation
3. Extraction of wavefunction files via `orca_2mkl`
4. Two-stage RESP fitting in Multiwfn for both gas and solvent
5. Mixing charges and emitting a unified `fit/charge_manifest.json`

---

## Polymer Charge Projection & Tiling

{% hint style="warning" %}
**Never run QM calculations on a 100-mer polymer.** You will melt the CPU and deplete your funding.
{% endhint %}

Instead, optimize a representative monomer/dimer with **fake caps** (linkers) to preserve local chemistry, fit charges, and project them onto the real polymer structure.

### Capped Oligomer Paradigm

```
Fake Cap (Redistributed)           Real Monomer (Tiled)             Fake Cap (Redistributed)
        [ACE] -----------> [-NH - CH(R) - CO-] <----------- [NME]
```

1. **Fake Caps**: Kept during QM/RESP calculations to avoid edge effects, but their charges are redistributed to adjacent real atoms before deployment.
2. **Real Caps**: Kept as chemically real atoms for polymer termini (assigned through `head` or `tail` sets).

### Projecting Charges

Compile a repeat unit charge set (e.g., `mid`) to a 100-mer polymer sequence:

```bash
# Tile a repeat charge manifest into a 100-mer sequence
warp-qm project-charges charge_manifest.json \
  --repeat-count 100 \
  --repeat-set mid \
  --terminal-policy repeat_tiled_no_terminal_specific_charges \
  --charge-format warp-build-charge \
  --out charge_manifest.warp-build.json
```

Convert a small-molecule manifest to `warp-build-charge` without tiling:
```bash
warp-qm convert-manifest charge_manifest.json \
  --to warp-build-charge \
  --out charge_manifest.warp-build.json
```

---

## Projection Inference

Writing charge projection maps by hand is error-prone. Let `warp-qm` infer them for you by comparing structural templates (MOL2 format):

```bash
warp-qm projection infer \
  --training-mol2 capped_oligomer.mol2 \
  --middle-mol2 middle.mol2 \
  --start-mol2 start.mol2 \
  --end-mol2 end.mol2 \
  --policy fake_caps_redistributed_region_sets \
  --out charge_projection.json
```

The algorithm compares atom names across templates to map:
- Where the repeat unit is in the training oligomer.
- Which cap atoms are "fake" and need their charges redistributed to their attached real neighbors.

---

## Error Handling

Stable error codes returned in the JSON envelope:

| Error Code | Reason | What to Check |
|------------|--------|---------------|
| `E_ORCA_EXECUTABLE_MISSING` | ORCA path is invalid | Verify `WARP_QM_ORCA` |
| `E_MULTIWFN_EXECUTABLE_MISSING` | Multiwfn path is invalid | Verify `WARP_QM_MULTIWFN` |
| `E_ORCA_NONCONVERGENCE` | Self-Consistent Field (SCF) failed | Tweak keywords or geometry |
| `E_ORCA_PROCESS` | ORCA exited nonzero | Check output files for crashes |
| `E_RESP2_ATOM_COUNT_MISMATCH` | Gas/solvent structures mismatch | Did you change the topology? |
| `E_CHARGE_PROJECTION_INVALID` | Caps cannot be mapped to repeat unit | Check template atom names |
