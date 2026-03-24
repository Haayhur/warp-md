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

## Quick Start

{% tabs %}
{% tab title="Native CLI" %}
```bash
# Schema and examples
warp-build schema --kind request
warp-build example-bundle --out source.bundle.json
warp-build example --mode random_walk --bundle-path source.bundle.json > request.json

# Validate and run
warp-build validate request.json
warp-build run request.json --stream
```

{% hint style="warning" %}
`warp-build` is the public CLI wrapper. If the native builder binary is not on `PATH`, set `WARP_BUILD_BINARY` to the built `warp-build` executable.
{% endhint %}
{% endtab %}

{% tab title="Python Wrapper" %}
```python
from warp_md import build as wb

request = wb.example_request("random_walk")
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
- `prmtop`
- charge handoff manifest
- build manifest
- topology graph

The build manifest is the canonical handoff artifact for downstream `warp-pack`.

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

---

## Minimal Request

```json
{
  "version": "warp-build.agent.v1",
  "source_ref": "pmma.bundle.json",
  "target": {
    "mode": "linear_homopolymer",
    "repeat_token": "A",
    "n_repeat": 50,
    "termini": {
      "head": "default",
      "tail": "default"
    },
    "tacticity": "inherit"
  },
  "realization": {
    "conformation_mode": "random_walk",
    "seed": 12345
  },
  "artifacts": {
    "coordinates": "outputs/pmma_50mer.pdb",
    "manifest": "outputs/pmma_50mer.build.json"
  }
}
```

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

---

## Handoff into `warp-pack`

Once the chain is built, hand the manifest into `warp-pack`:

```json
{
  "version": "warp-pack.agent.v1",
  "polymer_build": {
    "build_manifest": "outputs/pmma_50mer.build.json"
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
      "neutralize": true,
      "salt_molar": 0.15,
      "cation": "Na+",
      "anion": "Cl-"
    },
    "morphology": {
      "mode": "single_chain_solution"
    }
  },
  "outputs": {
    "coordinates": "outputs/system.pdb",
    "manifest": "outputs/system_manifest.json"
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
