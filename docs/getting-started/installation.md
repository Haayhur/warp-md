---
description: From zero to simulation in minutes
icon: download
---

# Installation

Ready to give your agent superpowers? Let's get warp-md running.

---

## Choose Your Install Path

### Path A: Install from PyPI (recommended)

```bash
pip install warp-md
python -c "import warp_md; print(warp_md.System)"
```

Optional CUDA wheel:

```bash
pip install warp-md-cuda
```

Use this path when you want the fastest setup with prebuilt wheels.

---

### Path B: Build from Source (git clone)

Prerequisites:
- Rust toolchain
- Python 3.9+
- `maturin`

```bash
git clone https://github.com/Haayhur/warp-md.git
cd warp-md
pip install maturin
maturin develop
python -c "import warp_md; print(warp_md.System)"
```

Optional source checks:

```bash
cargo test
python -m pytest python/warp_md/tests
```

Use this path when you want latest source changes or to contribute to core modules.

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
