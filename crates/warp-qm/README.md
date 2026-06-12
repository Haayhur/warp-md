# warp-qm

Native QM orchestration CLI for agent workflows. Rust owns the contract, schemas, event stream, artifacts, and result envelopes; engine-specific chemistry is delegated to adapters or agent-supplied input files.

## Engine setup

ORCA and Multiwfn are external tools and are not bundled.

```bash
export WARP_QM_ORCA=/path/to/orca
export WARP_QM_MULTIWFN=/path/to/Multiwfn
export WARP_QM_MULTIWFN_LIB_DIR=/path/to/multiwfn/lib
```

Check runnable surface:

```bash
warp-qm capabilities --json
```

## Generic engine runs

Use `generic_run` when the agent needs exact engine input control.

ORCA:

```json
{
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

Multiwfn:

```json
{
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

Psi4 currently supports `generic_run` only.

## RESP and RESP2

`resp_fit` with `charge_model: "resp"` runs Multiwfn standard two-stage RESP:

```text
7 -> 18 -> 1
```

`charge_model: "resp2"` runs standard RESP on gas and solvent inputs, then mixes:

```text
q_resp2 = (1 - delta) * q_gas_resp + delta * q_solvent_resp
```

Example settings:

```json
{
  "engine": {
    "name": "multiwfn",
    "settings": {
      "gas_input_file": "mol_gas.fchk",
      "solvent_input_file": "mol_solv.fchk",
      "delta": 0.5
    }
  },
  "task": {
    "kind": "resp_fit",
    "method": "multiwfn",
    "charge_model": "resp2"
  }
}
```

## Polymer charge policy

Do not RESP-fit a full long polymer by default. Use a capped monomer or representative short model, then deploy the charge map.

Fake/linker caps:

- included in QM/RESP
- redistributed to adjacent real atoms
- excluded from deployable force-field charges

Real terminal caps:

- kept as chemically real atoms
- assigned through explicit `head` and `tail` deployable sets

Example projection section in `charge_manifest.json`:

```json
{
  "projection": {
    "policy": "fake_caps_redistributed_repeat_unit",
    "redistribution": [
      {"source_atom": 5, "target_atoms": [0]},
      {"source_atom": 8, "target_atoms": [1]}
    ],
    "deployable_sets": [
      {
        "name": "mid",
        "role": "interior_repeat",
        "atom_indices": [0, 1, 2, 3, 4, 6, 7, 9, 10]
      }
    ]
  }
}
```

Deploy the repeat charges:

```bash
warp-qm project-charges charge_manifest.json \
  --repeat-count 100 \
  --repeat-set mid \
  --terminal-policy repeat_tiled_no_terminal_specific_charges \
  --out polymer_charge_manifest.json
```

Example:

```bash
warp-qm project-charges paa_resp_charge_manifest.json \
  --repeat-count 10 \
  --repeat-set mid \
  --terminal-policy repeat_tiled_no_terminal_specific_charges \
  --out polymer_charge_manifest_n10.json
```

## Validation status

The CLI contract, schemas, event stream, ORCA/Multiwfn execution paths, RESP, RESP2 mixing, and polymer charge deployment are tested. Canonical force-field parity against AmberTools/Gaussian RESP benchmark data should be treated as a separate scientific validation gate.
