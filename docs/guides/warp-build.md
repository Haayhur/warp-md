---
description: Native warp-build stage for agent-first chain construction
icon: dna
---

# Warp Build

`warp-build` owns polymer construction. `warp-pack` owns world assembly around the built chain.

{% hint style="info" %}
**Stable boundary:** build the polymer first, then hand the build manifest to `warp-pack` for solvent, ions, box, and morphology.
{% endhint %}

---

## The Build/Pack Split

Use the tools like this:

1. `warp-build` or `warp_md.build` compiles a reusable source bundle into a target chain
2. `warp-pack` consumes the build manifest and assembles the world

That split keeps:

- polymer semantics in `warp-build`
- world-build semantics in `warp-pack`
- planner contracts explicit and machine-readable

---

## CLI Reference

| Subcommand | Flags | What It Does |
|-----------|-------|-------------|
| `schema` | `--kind request|result|event|source_bundle|build_manifest|charge_manifest|topology_graph`, `--format json|yaml`, `--json`, `--out PATH` | Print a build contract schema variant |
| `example` | `--mode aligned`, `--bundle-path PATH`, `--format json|yaml`, `--json` | Print an example build request |
| `example-bundle` | `--out PATH`, `--format json|yaml`, `--json` | Materialize an example source bundle to disk |
| `capabilities` | `--format json|yaml`, `--json` | Print build capabilities fingerprint |
| `inspect-source <path>` | `--format json|yaml`, `--json` | Inspect a source bundle: validate referenced artifacts and junction selectors against residue templates |
| `validate <request>` | `--deep`, `--shallow`, `--format json|yaml`, `--json` | Validate a build request (defaults to deep geometry/QC preflight) |
| `run <request>` | `--stream` | Execute a build request and emit artifacts |

---

## Quick Start

{% tabs %}
{% tab title="Native CLI" %}
```bash
# Schema and examples
warp-build schema --kind request
warp-build example-bundle --out source.bundle.json
warp-build example --mode aligned --bundle-path source.bundle.json > request.json

# Validate and run
warp-build validate request.json
warp-build run request.json --stream
```

Validation defaults to deep geometry/QC preflight. Use an explicit shallow override only when you want a fast schema/contracts check:

```json
{"validation": {"depth": "shallow"}}
```

Deep validate responses include `preflight_cache.cache_key`, `preflight_cache.request_digest`, and `preflight_cache.input_digest`. These identify the exact normalized request and the build inputs checked by preflight, so agents can correlate validate/run decisions deterministically.

By default, validation does not write final build artifacts. Agents that intentionally want validate-to-run reuse can opt into an explicit artifact cache:

```json
{
  "validation": {
    "depth": "deep",
    "cache_mode": "record",
    "cache_dir": ".warp-build-cache"
  }
}
```

Then run with `cache_mode: "prefer"` to reuse a matching cache when available, or `cache_mode: "require"` to fail if the cache is absent or stale. Cache matching uses `input_digest`, which ignores artifact output destinations and validation cache controls.

{% hint style="warning" %}
`warp-build` is the public CLI wrapper. If the native builder binary is not on `PATH`, set `WARP_BUILD_BINARY` to the built `warp-build` executable.
{% endhint %}
{% endtab %}

{% tab title="Python Wrapper" %}
```python
from warp_md import build as wb

request = wb.example_request("aligned")
bundle = wb.example_bundle()
caps = wb.capabilities()

result = wb.validate(request)
exit_code, envelope = wb.run(request, stream=False)
print(exit_code, envelope["status"])
```

If the native binary is not on `PATH`, point the wrapper at it:

```bash
export WARP_BUILD_BINARY="$PWD/target/debug/warp-build"
```
{% endtab %}
{% endtabs %}

---

## What `warp-build` Produces

A successful build emits:

- built coordinates
- `inpcrd`
- charge handoff manifest
- build manifest
- topology graph
- copied `ffxml` artifact only when the source bundle provides a transferable, non-placeholder `forcefield_ref`
- `prmtop` when the selected handoff path is exact-transfer, forcefield-backed, or synthetic minimizer-ready

The build manifest is the canonical handoff artifact for downstream `warp-pack`.

### Topology Graph Schema (`warp-build.topology-graph.v5`)

The `topology_graph` output file contains a rich JSON representation of the built polymer topology, useful for analytical introspection, force field assignments, or visualization:

<details>
<summary>Key Graph Fields</summary>

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | `str` | Always `"warp-build.topology-graph.v5"` |
| `request_id` | `str` | The triggering request's ID |
| `bundle_id` | `str` | The source parameters bundle ID |
| `atoms` | `list[Atom]` | Flat array of atom records |
| `bonds` | `list[Pair]` | Global atom index bond pairs `{"a": idx, "b": idx}` |
| `angles` | `list[Angle]` | Global atom index angles `{"a", "b", "c"}` |
| `dihedrals` | `list[Torsion]` | Global atom index dihedrals `{"a", "b", "c", "d"}` |
| `impropers` | `list[Torsion]` | Global atom index improper dihedrals |
| `exclusions` | `list[Exclusion]` | Lists of excluded nonbonded atoms per atom |
| `residues` | `list[Residue]` | Array of residue structures |
| `sequence` | `list[str]` | Target polymer token sequence |
| `open_ports` | `list[OpenPort]` | Active connection ports left uncapped |
| `applied_caps` | `list[AppliedCap]` | Capping groups applied to ports |
| `relax_metadata` | `RelaxMetadata?` | Metrics from geometry relaxation (e.g. max clash) |

</details>

<details>
<summary>Sub-record Specifications</summary>

**`Atom` Record**:
```json
{
  "index": 0,
  "name": "C1",
  "element": "C",
  "resid": 1,
  "resname": "MMA",
  "charge_e": -0.15,
  "mass": 12.011,
  "atom_type_index": 4,
  "amber_atom_type": "c3",
  "lj_class": "CT",
  "position": [0.0, 1.2, -0.4],
  "neighbors": [1, 2, 3]
}
```

**`Residue` Record**:
```json
{
  "resid": 1,
  "resname": "MMA",
  "node_id": "monomer_0",
  "request_node_id": "monomer_0",
  "sequence_token": "M",
  "token_kind": "unit",
  "source_token": "M",
  "branch_depth": 0,
  "branch_path": "0",
  "atom_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
  "ports": [
    { "name": "head", "attach_atom": "C1", "leaving_atoms": ["H1"] },
    { "name": "tail", "attach_atom": "C2", "leaving_atoms": ["H2"] }
  ]
}
```

</details>



## End-to-End Verification

Use the lightweight verifier when you need behavior evidence beyond unit tests:

```bash
python scripts/validation/verify_warp_build_e2e.py \
  --warp-build-bin target/debug/warp-build \
  --workdir results/manifest/warp_build_e2e
```

The verifier runs the native CLI, materializes an example bundle, inspects it,
records a deep preflight cache, runs with `--stream` and `cache_mode=require`,
then checks emitted artifacts, manifest digests, topology graph consistency,
QC status, stream event ordering, and basic coordinate sanity. It also builds a
branched PMMA fixture as a non-linear polymer form and verifies the handoff
manifest/topology graph records branch depth. It exits nonzero if the binary is
stale or the observed behavior no longer matches the contract.

For broader generated-bundle audits, including Python-generated 3mer training
data from simple linear templates through bulky, star, graph, and fixture
branched cases:

```bash
python scripts/validation/audit_warp_build_usecases.py \
  --warp-build-bin target/debug/warp-build
```

This writes a CSV and summary JSON under
`results/manifest/warp_build_usecase_audit/` and preserves failing debug
coordinates for clash diagnosis.

Successful handoff tiers:

- `md_ready`: transferable source topology + charge handoff available
- `forcefield_backed`: training source was unreliable, validated `ffxml` fallback selected, production-capable topology emitted
- `minimizable_synthetic`: geometry-QC-clean build with synthetic UFF-like topology emitted for downstream minimization
- `graph_bonded_only`: coordinates + `inpcrd` + `topology_graph`, but no topology-capable or charge-capable handoff

---

## Public Build Model

The public request shape is intentionally narrow:

- reusable source bundle
- explicit target build mode
- explicit realization mode
- deterministic or seeded output artifacts

Common target modes:

- `linear_homopolymer`
- `linear_sequence_polymer`
- `block_copolymer`
- `random_copolymer`
- `star_polymer`
- `branched_polymer`
- `polymer_graph`

Common realization modes:

- `extended`
- `random_walk`
- `aligned`
- `ensemble`

Common stereochemistry/tacticity modes (`target.stereochemistry.mode`):

- `inherit` (default, copy from training oligomer configuration)
- `training` (use exact training sample sequence configurations)
- `isotactic` (all chiral centers identical)
- `syndiotactic` (alternating chiral centers)
- `atactic` (randomly assigned chiral centers)

For the most deterministic production handoff tests, use `aligned` or `extended` for linear chains. These modes build an axis-biased zigzag chain and run a final synthetic-topology relaxation when emitting synthetic PRMTOP/INPCRD artifacts.

`random_walk`, `star_polymer`, `branched_polymer`, and `polymer_graph` builds can also be reported as `minimizable_synthetic` when geometry QC is clean. The builder uses spatial-hash-backed residue-envelope self-avoidance and rejects folded-back placements that would create severe nonbonded overlaps. This contract means topology-loadable, no-overlap, downstream-minimizer-safe starting coordinates; it does not claim that the emitted coordinates are already at the same local minimum OpenMM or AmberTools would find.

For optional downstream-force-field polishing, use:

```bash
python scripts/validation/polish_warp_build_openmm.py \
  --topology outputs/chain.prmtop \
  --inpcrd outputs/chain.inpcrd \
  --out-pdb outputs/chain.openmm_minimized.pdb \
  --report-json outputs/chain.openmm_minimized.json
```

This is intentionally outside the core `warp-build run` contract because it depends on OpenMM availability and uses downstream force-field semantics.

---

## Minimal Request

```json
{
  "schema_version": "warp-build.agent.v1",
  "request_id": "polymer-build-50mer-001",
  "source_ref": {
    "bundle_id": "example_polymer_bundle_v1",
    "bundle_path": "source.bundle.json"
  },
  "target": {
    "mode": "linear_homopolymer",
    "repeat_unit": "A",
    "n_repeat": 50,
    "termini": {
      "head": "default",
      "tail": "default"
    },
    "stereochemistry": {
      "mode": "inherit"
    }
  },
  "realization": {
    "conformation_mode": "aligned",
    "alignment_axis": "z"
  },
  "artifacts": {
    "coordinates": "outputs/polymer_50mer.pdb",
    "build_manifest": "outputs/polymer_50mer.build.json",
    "charge_manifest": "outputs/polymer_50mer.charge.json"
  }
}
```

If you want best-effort recovery artifacts on QC failure instead of a hard error, set:

```json
{
  "realization": {
    "qc_policy": "salvage"
  }
}
```

Salvage writes non-final outputs, marks the build with `acceptance_state = "salvaged"`, returns `status = "salvaged"`, and exits nonzero so automation does not treat QC hard-fails as accepted MD-ready output.

---

## Source Bundles

The source bundle carries chemistry-heavy reusable state so planners do not have to rebuild it inline:

- training context
- unit library
- motif library
- junction definitions
- token/adjacency capability metadata
- charge/topology provenance

Inspect a bundle before use:

```bash
warp-build inspect-source source.bundle.json
```

`inspect-source` validates referenced source artifacts and junction selectors against residue templates. Template/selector mismatches are reported before `validate` or `run`; placeholder `ffxml` files are warned and are not emitted as MD-ready force-field handoff artifacts.

Deep validation now includes a training-source assessment and parameter-source selection step. The decision is surfaced in `preflight.parameter_source_decision`, the run summary, and the build manifest.

Selection policy:

- `trusted` / `risky` training source -> `synthetic_pdb`
- `unreliable` + `source_topology_ref` -> exact transferable topology path
- `unreliable` + validated `forcefield_ref` -> `forcefield_backed`
- `unreliable` + no stronger source -> validation error / run rejection

That means structure-first bundles no longer collapse to one behavior:

- good structure-only source -> `minimizable_synthetic`
- bad structure + strong fallback -> `md_ready` or `forcefield_backed`
- bad structure + no strong fallback -> rejected early

---

## Handoff into `warp-pack`

Once the chain is built, hand the manifest into `warp-pack`:

```json
{
  "schema_version": "warp-pack.agent.v1",
  "polymer_build": {
    "build_manifest": "outputs/polymer_50mer.build.json"
  },
  "environment": {
    "box": {
      "mode": "padding",
      "padding_angstrom": 12.0,
      "shape": "cubic"
    },
    "solvent": {
      "mode": "explicit",
      "model": "tip3p"
    },
    "ions": {
      "neutralize": {"enabled": true},
      "salt": {"name": "nacl", "molar": 0.15}
    },
    "morphology": {
      "mode": "single_chain_solution"
    }
  },
  "outputs": {
    "coordinates": "outputs/system.pdb",
    "format": "pdb-strict",
    "manifest": "outputs/system_manifest.json",
    "preserve_topology_graph": true,
    "write_conect": true
  }
}
```

For multi-component systems, prefer `components[]` with `source.kind = "polymer_build"`.

---

## Agent Notes

Planner-safe flow:

1. inspect source bundle capabilities
2. validate request
3. run build
4. hand `build_manifest` to `warp-pack`
5. treat `topology_graph` and manifests as canonical provenance

Do not target old inline polymer build through `warp-pack`. New integrations should always use `warp-build` first.

---

## See Also

- [Packing Guide](packing.md)
- [Streaming Progress API](streaming-progress.md)
- [API Reference](../reference/README.md)
