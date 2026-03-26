# Warp-Pack Polymer Build Consumer Spec

Status: requested consumer contract  
Date: 2026-03-11  
Owner: downstream co-scientist integration consumer  
Audience: `warp-pack` / `warp-md` builder agent  
Scope: polymer build stage contract; future-proof boundary into world-build, neutralization, and production-system assembly

## Why this doc exists

This is the strong version of the ask.

The co-scientist side does not want a generic packer. It wants a stable polymer production-builder surface that a planner can target without inventing chemistry semantics outside `warp-pack`.

Today, too much intent is reconstructed in co-scientist glue:

- polymer parametrization yields reusable artifacts such as `trimer_pdb`, `prmtop`, `inpcrd`, and `polymer_ffxml` in `packages/workflows/atomic/resp_gaff_param.yaml`
- polymer packing semantics are inferred in `packages/controllers/polymer_pack_helpers.py`
- downstream simulation expects a durable chain/world artifact boundary in `packages/workflows/atomic/run_md_polymer_pdb.yaml`

That is the wrong ownership split.

I want `warp-pack` to own the public semantics of:

- polymer build target
- chain-growth mode
- termini policy
- charge handoff
- build manifest
- later, world-build intent

This document is not conservative. It is the future-proof contract request.

## Hard design stance

### One public contract, layered internals

I want one public agent-safe contract family.

Internally, you can split it into:

- polymer topology model
- polymer coordinate builder
- charge/topology handoff
- world-build / packing / solvation / ions

But the consumer should not need to stitch those semantics together manually.

### Explicit stage boundary

This document focuses on the **polymer build stage** only.

Input:

- reusable polymer parameter source, usually derived from a training oligomer such as a `3mer`
- target production-chain specification

Output:

- built polymer coordinates
- build manifest
- charge handoff artifact
- topology / parameter provenance

Not in scope for this stage:

- solvent placement
- ion placement
- neutralization execution
- final periodic box construction
- density-driven bulk packing

Those belong to the later world-build stage. But the warp-build outputs must be strong enough for that stage to consume directly.

### The key chemistry rule

The training oligomer and the production target are not the same thing.

I need the contract to preserve:

- training source: typically `3mer`
- build target: typically `50mer`

The builder must never silently collapse:

- `3mer` source
- into `50mer` production chain

without an explicit target build request.

## Consumer use cases

These are the concrete cases I need to support.

### Core case

- derive reusable polymer parameters from a `3mer`
- build a `50mer` production chain
- emit coordinates plus charge/provenance manifests
- hand it to world-build for solvation, neutralization, salt, box, morphology

### Sequence case

- derive or supply a reusable polymer parameter source
- build a sequence-defined linear copolymer
- preserve sequence and termini in the manifest

### Tacticity-aware case

- build PMMA-like systems where tacticity matters
- choose `inherit`, `isotactic`, `syndiotactic`, `atactic`, or later an explicit stereochemical pattern

### Multi-target case

- same parameter source
- build `10mer`, `20mer`, `50mer`, `100mer`
- same source provenance
- deterministic or seeded output

## Reference repo feature map

This is the feature borrowing plan.

### 1. `polyconstruct`

Primary references:

- `polyconstruct/polytop/polytop/Polymer.py`
- `polyconstruct/polyconf/polyconf/Polymer.py`
- `polyconstruct/polyconf_examples/02a_build_PMMA_isotactic.py`
- `polyconstruct/polyconf_examples/02b_build_PMMA_syndiotactic.py`
- `polyconstruct/polyconf_examples/02c_build_PMMA_atactic.py`

What I want from it:

- polymer-specific topology vocabulary
- explicit junction/linker semantics
- separation between topology assembly and coordinate/conformation growth
- tacticity-aware chain-building concepts

What I do not want copied as the public contract:

- repo-specific file-format assumptions
- GROMACS-centric output model as the public API
- arbitrary topology surgery surfaced directly to the planner

### 2. `mbuild`

Primary references:

- `mbuild/mbuild/lib/recipes/polymer.py`
- `mbuild/mbuild/tests/test_polymer.py`
- `mbuild/mbuild/port.py`
- `mbuild/docs/topic_guides/recipes.rst`

What I want from it:

- `Port`-style connection abstraction
- clean sequence and end-group public API
- builder API that expresses intent instead of low-level geometry
- a compositional model where homopolymer is sugar over sequence

What I do not want copied as the public contract:

- broad materials framework
- PACKMOL-style public dependency semantics
- generic `Compound` graph exposed as planner input

### 3. `pysimm`

Primary references:

- `pysimm/pysimm/apps/random_walk.py`

Negative reference:

- `pysimm/pysimm/apps/polymatic.py`

What I want from it:

- random-walk chain-growth heuristics
- seeded stochastic coordinate growth
- optional cheap self-avoidance ideas

What I do not want copied as the public contract:

- reactive polymerization workflow
- LAMMPS-in-the-loop construction requirements
- simulation-heavy builder dependency

## What the builder should own

The `warp-pack` side should own:

- warp-build request schema
- validation and normalization
- source-vs-target semantics
- connection/junction interpretation
- termini application semantics
- conformation-generation semantics
- build manifest generation
- charge handoff generation or preservation
- streaming progress events

The co-scientist side should not own:

- synthetic build semantics
- low-level coordinate-growth policy
- hidden chain-assembly assumptions
- inferred build manifests
- duplicated validation logic

## Missing contract today: source bundle

This is the main missing layer.

The build request should not force the caller to assemble chemistry-heavy source semantics inline every time. If the caller has to provide:

- coordinates
- topology refs
- force-field refs
- charge refs
- junction definitions
- supported sequence tokens
- supported tacticity modes

inside each build request, then the caller still owns too much chemistry glue.

The clean fix is a companion source artifact/schema:

- `polymer-param-source.bundle.v1`

Then `warp-build.agent.v1` can stay lean:

- `source_ref`
- `target`
- `realization`
- `artifacts`

while the bundle owns:

- source provenance
- token library
- junction definitions
- source capabilities
- charge provenance
- training context

## Requested public contract family

### Contract family names

I want these public contract layers eventually:

1. `polymer-param-source.bundle.v1`
2. `warp-build.agent.v1`
3. `warp-pack.agent.v1` for world-build
4. optionally a later umbrella contract:
   - `warp-md.system-build.v1`

For now, this document defines `warp-build.agent.v1`.

## Level 1 features: must-have

These are the v1 features that unblock downstream integration.

### 1. Explicit source-bundle / build-target split

Need:

- source bundle ref, usually built from a training oligomer such as `3mer`
- production-chain target, e.g. `50mer`

Why:

- preserves chemistry meaning
- avoids source/target collapse
- enables later reproducibility and review
- keeps chemistry-heavy source assembly out of every build request

### 2. Linear homopolymer build

Need:

- `linear_homopolymer`

Why:

- core production-chain path

### 3. Linear sequence polymer build

Need:

- `linear_sequence_polymer`

Why:

- avoids painting the API into a homopolymer-only corner
- allows copolymers without redesigning the contract
- but only when the source bundle explicitly advertises token support

### 4. Explicit connection-site model

Need:

- named connection sites such as `head` and `tail`
- later-ready internal support for arbitrary named junctions

Why:

- agent/runtime must not rely on atom order conventions
- reusable polymer source must carry connection semantics
- junction semantics must be durable enough for a real builder, not just a naming hint

### 5. Termini control

Need:

- `default` termini
- explicit head/tail cap policy

Why:

- chemically meaningful
- affects charge and topology handoff
- normalized request and manifest must resolve `default` into an explicit applied policy

### 6. Conformation modes

Need:

- `extended`
- `random_walk`

Why:

- enough to cover practical v1 production-chain initialization

### 7. Build manifest

Need:

- machine-readable manifest artifact

Why:

- downstream world-build
- review/trace
- reproducibility

### 8. Charge handoff artifact

Need:

- target charge manifest or preserved total-charge handoff

Why:

- later neutralization should not require deep parameter parsing in world-build
- this artifact is mandatory, not optional

### 9. Validation + normalization

Need:

- `validate`
- normalized request output

Why:

- planner-safe
- repairable errors
- deterministic downstream behavior
- validation should be pure; it should not invent randomness by default

### 10. Streaming progress

Need:

- NDJSON or equivalent stable event stream

Why:

- raw operator trace
- better UX and auditability

## Level 2 features: strongly wanted

These are the features I want you to design in now, even if some ship slightly after the minimum path.

### 1. Sequence-aware linear builder

Public semantics:

- `linear_homopolymer`
- `linear_sequence_polymer`

Requested behavior:

- homopolymer is sugar over single-token sequence
- sequence tokens map to named monomer/unit definitions
- deterministic ordering
- validation of unknown tokens
- validation that the source bundle supports each token and the allowed token-to-token junction combinations

Reference pulls:

- `mbuild/mbuild/lib/recipes/polymer.py`
- `mbuild/mbuild/tests/test_polymer.py`
- `polyconstruct/polytop/polytop/Polymer.py`

### 2. Explicit connection / junction model

Public semantics:

- `head`
- `tail`

Internal semantics:

- named junctions beyond head/tail
- branch-ready internal graph model even if branching is deferred publicly

Requested behavior:

- connection definitions preserved in manifest
- failure if source model lacks required junction definitions
- no hidden atom-index-based magic in the public contract
- enough semantics to define real attachment behavior, not just labels

Minimum durable junction semantics should be capable of expressing:

- attach atom selector
- leaving atom selector(s), if any
- bond order / connection type
- naming scope or template scope
- optional orientation helpers or anchor atoms

Reference pulls:

- `mbuild/mbuild/port.py`
- `polyconstruct/polytop/polytop/Polymer.py`

### 3. Source-vs-target separation as a first-class rule

Requested behavior:

- request must separately encode the training oligomer source and the production target
- manifest must record both
- validator must reject ambiguity

Consumer requirement:

- `3mer` is parameter source
- `50mer` is build target

Not acceptable:

- implicitly treating source coordinates as production target

### 4. End-group / termini policy

Requested behavior:

- explicit `head` and `tail` termini policy
- `default` as minimal v1
- named cap identifiers later
- normalized request and manifest record actual applied termini after resolving `default`

Reference pulls:

- `mbuild/mbuild/lib/recipes/polymer.py`
- `mbuild/mbuild/tests/test_polymer.py`
- `polyconstruct/polytop/polytop/Polymer.py`

### 5. Conformation generation

Requested v1 modes:

- `extended`
- `random_walk`

Requested future modes:

- `helical_guess`
- `backbone_aligned`
- `ensemble_seed`

Requested behavior:

- deterministic `extended`
- seeded reproducible `random_walk`
- compact provenance in manifest
- later cheap self-avoidance / clash-avoidance acceptable

Reference pulls:

- `polyconstruct/polyconf/polyconf/Polymer.py`
- `pysimm/pysimm/apps/random_walk.py`

Not requested:

- reactive growth
- in-build MD relaxation loop
- LAMMPS-coupled construction

### 6. Optional stereochemistry / tacticity control

Requested public values:

- `inherit`
- `isotactic`
- `syndiotactic`
- `atactic`
- later: explicit pattern

Requested behavior:

- clean validation if unsupported for the source model
- preserved in manifest

Reference pulls:

- `polyconstruct/polyconf_examples/02a_build_PMMA_isotactic.py`
- `polyconstruct/polyconf_examples/02b_build_PMMA_syndiotactic.py`
- `polyconstruct/polyconf_examples/02c_build_PMMA_atactic.py`

### 7. Build manifest with strong provenance

Requested behavior:

- manifest is mandatory
- includes the full normalized request verbatim
- includes source bundle provenance and digests
- includes target, realization, applied termini, output paths, net charge if known, topology linkage, structured warnings
- includes builder version and algorithm/backend version

### 8. Charge handoff artifact

Requested behavior:

- builder either emits or forwards a stable `charge_manifest`
- world-build later can use it for neutralization

Preferred minimal format:

```json
{
  "version": "warp-pack.charge-manifest.v1",
  "solute_path": "polymer_50mer.pdb",
  "source_topology_ref": "polymer_3mer.prmtop",
  "target_topology_ref": null,
  "forcefield_ref": "polymer.ffxml",
  "charge_derivation": "source_bundle_transfer",
  "net_charge_e": 0.0,
  "atom_count": 4210
}
```

### 9. Strict validation and normalization

Requested behavior:

- schema validation
- semantic validation
- normalized request output
- repairable errors with structured paths and codes
- structured warnings with the same shape as errors

### 10. Seeded determinism

Requested behavior:

- explicit seed accepted for stochastic modes
- seed emitted in normalized request, manifest, result envelope
- reproducibility promise should be scoped to:
  - same builder version
  - same algorithm/backend version
  - same source bundle
  - same seed
  - same deterministic backend conditions

### 11. Minimal QC hooks

Requested checks:

- repeat count matches built chain
- connection graph continuous for linear polymer
- termini applied as requested
- atom/residue count nonzero
- basic topology linkage preserved

This is builder sanity, not simulation QC.

## Level 3 features: blank-check future scope

These are not required for v1, but should influence internal architecture now.

### 1. Sequence copolymers with repeat expansion

Need later:

- `sequence` plus `repeat_count`
- block copolymer expansion
- alternating/random patterns

### 2. Explicit stereochemical patterns

Need later:

- not just `atactic`
- explicit monomer-by-monomer stereo assignment or repeating stereo motifs

### 3. Branched / star / comb / dendritic polymers

Need later:

- internal graph model should not make this impossible
- public API can remain linear-only in v1

### 4. Multi-chain blueprint generation

Need later:

- multiple production chains built from one source contract
- later handoff to bulk morphology world-build

### 5. Ordered initial conformers

Need later:

- aligned backbone seeds
- lamellar seed geometry
- ordered-bulk-ready chain blueprints

### 6. Ensemble builder mode

Need later:

- multiple candidate conformers from one source contract
- useful for robust starting-state selection

### 7. Monomer/junction inference tools

Need later:

- optional helper commands to infer or validate junction definitions from source artifacts
- not a blocker for v1

### 8. Richer charge handoff

Need later:

- per-atom charge arrays
- residue charge summaries
- atom mapping between source and built target

## Neutralization requirement

The builder asked what it should consume for neutralization later.

My answer:

- yes, I will provide a charge format
- I do not need `warp-pack` to parse every force-field artifact first

Preferred first-class input:

- `charge_manifest.json`

Acceptable first native chemistry format:

- `prmtop`

Deferred:

- deep `ffxml` parsing

Why:

- for neutralization, total solute charge is the first hard requirement
- world-build does not need to become a general parameter interpreter to do useful work

## Capability discovery

Validation is not enough.

The planner and caller need to know, before trial-and-error, whether a source supports:

- sequence tokens
- token adjacency rules
- tacticity modes
- conformation modes
- junction inference
- termini policies
- charge-transfer support

That capability surface can live in either:

- the source bundle itself
- or a companion `capabilities` command

Best case: both.

## Requested CLI / tool surface

I want a proper builder-facing surface.

### 1. Validation

```bash
warp-build validate request.json --format json
```

Needed behavior:

- schema validation
- semantic validation
- normalized request in success mode
- warnings and errors with the same structured shape

### 2. Schema

```bash
warp-build schema --format json
```

Needed behavior:

- public request schema
- enums
- required/optional fields
- examples if possible
- source bundle schema too

### 3. Examples

```bash
warp-build example --mode homopolymer --format json
warp-build example --mode sequence --format json
```

### 4. Run

```bash
warp-build run request.json --stream ndjson
```

Needed behavior:

- stable result envelope
- streamed progress events

### 5. Capabilities

```bash
warp-build capabilities --format json
warp-build inspect-source source.bundle.json --format json
```

Needed behavior:

- supported contract versions
- builder version
- algorithm/backend version
- supported conformation modes
- supported tacticity modes
- source-token library and adjacency support
- supported termini policies

## Requested `polymer-param-source.bundle.v1` schema

This is the chemistry-heavy source layer.

The goal is to keep this detail out of every individual build request.

### Top-level source bundle

```json
{
  "schema_version": "polymer-param-source.bundle.v1",
  "bundle_id": "pmma_param_bundle_v1",
  "training_context": {},
  "provenance": {},
  "unit_library": {},
  "junction_library": {},
  "capabilities": {},
  "artifacts": {},
  "charge_model": {}
}
```

### `training_context`

```json
{
  "mode": "oligomer_training",
  "training_oligomer_n": 3,
  "notes": "RESP/GAFF2 surrogate training"
}
```

This is descriptive, not a universal rule that every source must be `>= 3`.

### `unit_library`

This is required for any sequence-capable source.

```json
{
  "A": {
    "display_name": "PMMA repeat unit",
    "junctions": {
      "head": "pmma_head",
      "tail": "pmma_tail"
    }
  },
  "B": {
    "display_name": "alternate repeat unit",
    "junctions": {
      "head": "alt_head",
      "tail": "alt_tail"
    }
  }
}
```

### `junction_library`

This is where durable builder semantics live.

```json
{
  "pmma_head": {
    "attach_atom": {"scope": "unit", "selector": "name C1"},
    "leaving_atoms": [],
    "bond_order": 1,
    "anchor_atoms": [
      {"scope": "unit", "selector": "name C1"},
      {"scope": "unit", "selector": "name C2"}
    ]
  },
  "pmma_tail": {
    "attach_atom": {"scope": "unit", "selector": "name O4"},
    "leaving_atoms": [],
    "bond_order": 1,
    "anchor_atoms": [
      {"scope": "unit", "selector": "name O4"},
      {"scope": "unit", "selector": "name C3"}
    ]
  }
}
```

### `capabilities`

```json
{
  "supported_target_modes": [
    "linear_homopolymer",
    "linear_sequence_polymer"
  ],
  "supported_conformation_modes": [
    "extended",
    "random_walk"
  ],
  "supported_tacticity_modes": [
    "inherit",
    "isotactic",
    "syndiotactic",
    "atactic"
  ],
  "supported_termini_policies": [
    "default",
    "source_default"
  ],
  "sequence_token_support": {
    "tokens": ["A", "B"],
    "allowed_adjacencies": [
      ["A", "A"],
      ["A", "B"],
      ["B", "A"]
    ]
  },
  "charge_transfer_supported": true
}
```

### `artifacts`

```json
{
  "source_coordinates": "pmma_trimer.pdb",
  "source_topology_ref": "pmma_trimer.prmtop",
  "forcefield_ref": "pmma_polymer.ffxml",
  "source_charge_manifest": "pmma_trimer_charge.json"
}
```

## Requested `warp-build.agent.v1` schema

### Top-level request

```json
{
  "schema_version": "warp-build.agent.v1",
  "request_id": "warp-build-001",
  "source_ref": {},
  "target": {},
  "realization": {},
  "artifacts": {}
}
```

### `source_ref`

This points to the source bundle, not raw chemistry fragments.

```json
{
  "bundle_id": "pmma_param_bundle_v1",
  "bundle_path": "pmma_param_bundle.json",
  "bundle_digest": "sha256:..."
}
```

Required fields:

- one stable source reference
- enough identity to audit the exact source bundle used

### `target`

This is polymer identity, not realization policy.

Homopolymer example:

```json
{
  "mode": "linear_homopolymer",
  "repeat_unit": "A",
  "n_repeat": 50,
  "termini": {
    "head": "default",
    "tail": "default"
  },
  "stereochemistry": {
    "mode": "syndiotactic"
  }
}
```

Sequence example:

```json
{
  "mode": "linear_sequence_polymer",
  "sequence": ["A", "A", "B", "A"],
  "repeat_count": 12,
  "termini": {
    "head": "default",
    "tail": "default"
  },
  "stereochemistry": {
    "mode": "inherit"
  }
}
```

Allowed v1 `mode`:

- `linear_homopolymer`
- `linear_sequence_polymer`

Rules:

- `linear_homopolymer` requires `repeat_unit` and `n_repeat`
- `linear_sequence_polymer` requires `sequence`
- `repeat_count` may expand the provided sequence motif
- `sequence` is valid only when the source bundle advertises token support

### `realization`

This is how coordinates are generated for the requested target.

```json
{
  "conformation_mode": "random_walk",
  "seed": 12345
}
```

Required:

- `conformation_mode`

Rules:

- `extended` may omit seed
- `random_walk` should either require seed or let `run` materialize one and report it in result + manifest
- `validate` should not invent a seed by default

### `artifacts`

These are requested outputs, not chemistry semantics.

```json
{
  "coordinates": "polymer_50mer.pdb",
  "build_manifest": "polymer_50mer.build.json",
  "charge_manifest": "polymer_50mer.charge.json"
}
```

Required:

- `coordinates`
- `build_manifest`
- `charge_manifest`

## Example full request

```json
{
  "schema_version": "warp-build.agent.v1",
  "request_id": "pmma-build-50mer-001",
  "source_ref": {
    "bundle_id": "pmma_param_bundle_v1",
    "bundle_path": "pmma_param_bundle.json",
    "bundle_digest": "sha256:..."
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
      "mode": "syndiotactic"
    }
  },
  "realization": {
    "conformation_mode": "random_walk",
    "seed": 12345
  },
  "artifacts": {
    "coordinates": "pmma_50mer.pdb",
    "build_manifest": "pmma_50mer.build.json",
    "charge_manifest": "pmma_50mer.charge.json"
  }
}
```

## Result envelope

Success:

```json
{
  "schema_version": "warp-build.agent.v1",
  "status": "ok",
  "request_id": "pmma-build-50mer-001",
  "artifacts": {
    "coordinates": "pmma_50mer.pdb",
    "build_manifest": "pmma_50mer.build.json",
    "charge_manifest": "pmma_50mer.charge.json"
  },
  "summary": {
    "build_mode": "linear_homopolymer",
    "n_repeat": 50,
    "atom_count": 4210,
    "total_repeat_units": 50,
    "conformation_mode": "random_walk",
    "seed": 12345
  },
  "warnings": []
}
```

Error:

```json
{
  "schema_version": "warp-build.agent.v1",
  "status": "error",
  "request_id": "pmma-build-50mer-001",
  "errors": [
    {
      "code": "E_INVALID_TARGET",
      "path": "/target/n_repeat",
      "message": "n_repeat must be >= 1"
    }
  ],
  "warnings": []
}
```

## Build manifest

This artifact is mandatory.

```json
{
  "version": "warp-build.manifest.v1",
  "request_id": "pmma-build-50mer-001",
  "normalized_request": {
    "schema_version": "warp-build.agent.v1",
    "request_id": "pmma-build-50mer-001"
  },
  "source_bundle": {
    "bundle_id": "pmma_param_bundle_v1",
    "bundle_digest": "sha256:..."
  },
  "target": {
    "mode": "linear_homopolymer",
    "repeat_unit": "A",
    "n_repeat": 50,
    "termini": {
      "head": "source_default",
      "tail": "source_default"
    },
    "stereochemistry": {
      "mode": "syndiotactic"
    }
  },
  "realization": {
    "conformation_mode": "random_walk",
    "seed": 12345
  },
  "artifacts": {
    "coordinates": "pmma_50mer.pdb"
  },
  "summary": {
    "atom_count": 4210,
    "total_repeat_units": 50,
    "net_charge_e": 0.0
  },
  "warnings": [],
  "provenance": {
    "schema_version": "warp-build.agent.v1",
    "builder_version": "warp-build 1.0.0",
    "algorithm_version": "random_walk.v1"
  }
}
```

## Charge manifest

This is the preferred world-build neutralization handoff.

```json
{
  "version": "warp-pack.charge-manifest.v1",
  "solute_path": "pmma_50mer.pdb",
  "source_topology_ref": "pmma_trimer.prmtop",
  "target_topology_ref": null,
  "forcefield_ref": "pmma_polymer.ffxml",
  "charge_derivation": "source_bundle_transfer",
  "net_charge_e": 0.0,
  "atom_count": 4210
}
```

Future-compatible extension:

- later include per-atom charges or residue charge summaries
- not required for first integration

## Validation rules

These should be hard failures.

1. source bundle schema and digest must be valid.
2. source bundle must advertise support for the requested target mode.
3. `target.mode` must be valid.
4. `linear_homopolymer` requires `repeat_unit` and `n_repeat`.
5. `linear_sequence_polymer` requires `sequence`.
6. each sequence token must exist in the source bundle token library.
7. requested token adjacencies must be supported by the source bundle.
8. `n_repeat` and `repeat_count` must be `>= 1`.
9. `realization.conformation_mode` must be a supported value.
10. if `stereochemistry.mode` is unsupported for the source bundle, reject cleanly.
11. if required junction semantics are missing and cannot be inferred, reject.
12. source/target ambiguity must be rejected.
13. output artifacts must not collide illegally.
14. seed must be valid for stochastic modes.

Requested error shape:

```json
{
  "status": "error",
  "valid": false,
  "errors": [
    {
      "code": "E_MISSING_FIELD",
      "path": "/source_ref",
      "message": "head junction definition is required for linear polymer growth"
    }
  ],
  "warnings": []
}
```

## Normalization behavior

I want normalized requests returned in validation success mode.

Normalization should:

- resolve `default` termini into explicit applied policy
- fill deterministic defaults where appropriate
- canonicalize enum values
- preserve explicit user choices
- not invent randomness during `validate`

If `random_walk` is allowed without an explicit seed:

- `run` may materialize one
- result + manifest must record it

## Streaming event model

I want raw event streaming.

Command:

```bash
warp-build run request.json --stream ndjson
```

Requested event types:

- `run_started`
- `source_loaded`
- `phase_started`
- `chain_growth_started`
- `chain_growth_progress`
- `chain_growth_completed`
- `manifest_written`
- `phase_completed`
- `run_completed`
- `run_failed`

Example:

```json
{"event":"run_started","request_id":"pmma-build-50mer-001"}
{"event":"source_loaded","training_oligomer_n":3}
{"event":"phase_started","phase":"chain_growth"}
{"event":"chain_growth_progress","completed_repeats":20,"target_repeats":50,"progress_pct":40.0}
{"event":"chain_growth_completed","target_repeats":50,"elapsed_ms":842}
{"event":"manifest_written","path":"pmma_50mer.build.json"}
{"event":"run_completed","artifacts":{"coordinates":"pmma_50mer.pdb","build_manifest":"pmma_50mer.build.json"}}
```

Each event should include where relevant:

- `request_id`
- `phase`
- `elapsed_ms`
- `progress_pct`
- `target_repeats`
- `artifact`
- structured `error`

## Acceptance tests I want

These are the golden tests that would convince me the interface is real.

### 1. Homopolymer build from oligomer source

Input:

- `3mer` source
- `linear_homopolymer`
- `n_repeat = 50`
- `extended`

Verify:

- built `50mer` coordinates
- manifest records the source bundle and training context
- `total_repeat_units = 50`

### 2. Build `n_repeat = 1` from a training source

Input:

- training source bundle
- `linear_homopolymer`
- `n_repeat = 1`

Verify:

- valid build
- no false ambiguity error

### 3. Build `n_repeat = 2` from a training source

Input:

- training source bundle
- `linear_homopolymer`
- `n_repeat = 2`

Verify:

- valid build
- no hardcoded assumption that production target must exceed training length

### 4. Seeded random-walk reproducibility

Input:

- same request twice
- same seed

Verify:

- same build-decision digest, or same output digest, under the same builder version and backend conditions

### 5. Sequence copolymer build

Input:

- `linear_sequence_polymer`
- `sequence = ["A", "A", "B", "A"]`
- `repeat_count = 12`

Verify:

- sequence expansion preserved in manifest
- ordering deterministic

### 6. Unknown sequence token

Input:

- `linear_sequence_polymer`
- unknown token

Verify:

- structured validation failure

### 7. Source lacks token library for sequence mode

Input:

- sequence request
- source bundle with no sequence token support

Verify:

- structured validation failure

### 8. Termini override

Input:

- explicit head/tail policy

Verify:

- manifest records applied termini
- output respects policy

### 9. Default termini resolution

Input:

- `default` termini

Verify:

- manifest records resolved applied policy, not just raw `default`

### 10. Unsupported stereochemistry request

Input:

- stereochemistry mode unsupported by source bundle

Verify:

- structured validation failure

### 11. Missing junction definitions

Input:

- no usable junction info

Verify:

- fail before build

### 12. Missing seed in `random_walk`

Input:

- `random_walk`
- no seed

Verify:

- either structured validation failure
- or successful run with materialized seed recorded in result + manifest

### 13. Charge handoff emission

Input:

- successful build

Verify:

- `charge_manifest` emitted or preserved
- total net charge available to downstream world-build

### 14. Target length equals source length

Input:

- explicit target equal to training source length

Verify:

- valid request
- not treated as ambiguous solely because the lengths match

## What I do not need in v1

Keep these out unless already cheap:

- branched polymers
- dendrimers
- crosslinked networks
- reactive polymerization
- in-build MD relaxation
- LAMMPS dependency
- arbitrary chemistry graph editor as the public surface

## Recommended implementation order

### Phase 1

- `warp-build.agent.v1` request/result schema
- validation
- `linear_homopolymer`
- explicit `head` / `tail` junction model
- `default` termini
- `extended`
- manifest
- charge handoff

### Phase 2

- `linear_sequence_polymer`
- seeded `random_walk`
- stronger normalization
- streaming progress

### Phase 3

- tacticity control
- richer provenance
- optional inference helpers for junctions

### Phase 4

- architecture prep for multi-chain and branched topologies
- but keep public v1 stable

## Final consumer request

Build `warp-build.agent.v1` as a first-class public contract.

Make it:

- explicit about source vs target
- explicit about junctions and termini
- explicit about conformation mode and seed
- manifest-first
- validation-first
- charge-handoff aware

Internally, steal aggressively from:

- `polyconstruct` for polymer semantics
- `mbuild` for public API shape and connection abstraction
- `pysimm` for seeded random-walk growth

But do not expose their raw internal complexity as the public interface.

If this stage is implemented well, the later `warp-pack` world-build contract becomes straightforward:

- consume built polymer coordinates
- consume charge manifest
- solvate
- neutralize
- salt
- build morphology / box

That is the tight boundary I want.

## Non-normative reference appendix

These are the concrete repos and files reviewed while drafting the contract. They are informative, not normative.

- `https://github.com/OMaraLab/polyconstruct`
  - `polytop/polytop/Polymer.py`
  - `polyconf/polyconf/Polymer.py`
  - `polyconf_examples/02a_build_PMMA_isotactic.py`
  - `polyconf_examples/02b_build_PMMA_syndiotactic.py`
  - `polyconf_examples/02c_build_PMMA_atactic.py`
- `https://github.com/mosdef-hub/mbuild`
  - `mbuild/lib/recipes/polymer.py`
  - `mbuild/tests/test_polymer.py`
  - `mbuild/port.py`
  - `docs/topic_guides/recipes.rst`
- `https://github.com/polysimtools/pysimm`
  - `pysimm/apps/random_walk.py`
  - negative reference: `pysimm/apps/polymatic.py`
