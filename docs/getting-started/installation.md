---
description: From zero to simulation in minutes
icon: download
---

# Installation

Ready to give your agent superpowers? Let's get warp-md running.

---

## Get Started in 30 Seconds

{% tabs %}
{% tab title="PyPI (One-Liner)" %}
```bash
# The fast lane — no compilation, no waiting, no excuses
pip install warp-md

# Want GPU superpowers? We've got you covered
pip install warp-md-cuda

# Verify your agent's new toolkit
python -c "import warp_md; print('🎉', warp_md.System)"
```

{% hint style="success" %}
**Fastest path to molecular enlightenment.** Pre-built wheels, batteries included, GPU falls back gracefully to CPU.
{% endhint %}
{% endtab %}

{% tab title="Clone & Build (Bleeding Edge)" %}
```bash
# For those who like their code fresh from the oven
git clone https://github.com/haayhur/warp-md.git
cd warp-md

# Build Python bindings (go grab a ☕ — Rust is optimizing 47 crates for you)
maturin develop

# Victory lap
python -c "import warp_md; print('🎉', warp_md.System)"
```

{% hint style="info" %}
**Source builds get the latest fixes** and let you hack on the internals. Requires Rust 1.70+ and `pip install maturin`.
{% endhint %}
{% endtab %}

{% tab title="CLI (Agent's Best Friend)" %}
```bash
# Already installed? The CLI materializes automatically
warp-md --help

# Discover what's possible — full parameter schemas included  
warp-md list-plans --json --details

# Run analysis with JSON envelope output
warp-md rg --topology protein.pdb --traj trajectory.xtc --selection "protein"

# Batch mode with streaming progress events
warp-md run config.json --stream ndjson
```

{% hint style="success" %}
**Agent pro-tip:** All analysis commands emit structured JSON envelopes by default — perfect for machine consumption.
{% endhint %}
{% endtab %}

{% tab title="Rust (Speed Demons)" %}
```bash
# For the enthusiasts who measure in nanoseconds
cargo test

# Unleash the GPU (requires CUDA_HOME set)
cargo test -p traj-engine --features cuda

# Build everything with all features
cargo build --release --all-features
```

{% hint style="info" %}
**Pure Rust builds** give you maximum control. Use this when embedding warp-md in Rust projects or contributing to the core.
{% endhint %}
{% endtab %}
{% endtabs %}

---

## Optional: GPU Turbo Mode

{% hint style="info" %}
**No GPU? No problem.** warp-md gracefully falls back to CPU. CUDA is for when you need that extra *oomph*.
{% endhint %}

{% tabs %}
{% tab title="PyPI GPU Package" %}
If you installed `warp-md-cuda` from PyPI, CUDA support is already enabled! Just ensure your system has CUDA installed.

```bash
# Verify CUDA installation
nvcc --version
```

If CUDA isn't found, download from the [CUDA Toolkit page](https://developer.nvidia.com/cuda-downloads).
{% endtab %}

{% tab title="Source Build with CUDA" %}

{% stepper %}
{% step %}
## Check CUDA Installation

```bash
nvcc --version
```

If this says "command not found", visit the [CUDA Toolkit download page](https://developer.nvidia.com/cuda-downloads).
{% endstep %}

{% step %}
## Tell Rust Where CUDA Lives

```bash
export CUDA_HOME=/usr/local/cuda
# Windows folks:
# set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0
```
{% endstep %}

{% step %}
## Build with CUDA

```bash
cargo test -p traj-engine --features cuda
```

Your agent now has access to ~15x speedups on compatible analyses.
{% endstep %}
{% endstepper %}

{% endtab %}
{% endtabs %}

---

## CLI: Your Agent's Native Interface

After installing Python bindings, the CLI materializes automatically:

```bash
warp-md --help
warp-md list-plans --json  # See what's available, agent-style
```

{% hint style="success" %}
**Pro tip**: Analysis and `run` commands emit JSON envelopes by default; discovery commands support `--json`/`--format json`.
{% endhint %}

---

## Troubleshooting

<details>
<summary>maturin: command not found</summary>

Make sure pip installed it in your PATH:

```bash
pip install --user maturin
export PATH="$HOME/.local/bin:$PATH"
```
</details>

<details>
<summary>CUDA not detected</summary>

Set `CUDA_HOME` or `CUDA_PATH` to your CUDA installation. The `cudarc` crate needs this breadcrumb to find the runtime.
</details>

<details>
<summary>Atom count does not match system</summary>

Your topology and trajectory are having a disagreement. Make sure `System.from_pdb()` and `Trajectory.open_*()` are looking at the same molecular system.
</details>

---

<a href="quickstart.md" class="button primary">Next: Quick Start →</a>
