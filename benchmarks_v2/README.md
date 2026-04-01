# benchmarks_v2

New per-plan benchmark layout.

## Structure
- `plans/<plan_slug>/mdanalysis.py`
- `plans/<plan_slug>/mdtraj.py`
- `plans/<plan_slug>/gmx.py`
- `plans/<plan_slug>/warp_md.py`
- `plans/<plan_slug>/warp_md_gpu.py`
- `plan_index.json`

## Run Example
```bash
python benchmarks_v2/plans/rg/warp_md.py --top <top.pdb> --traj <traj.xtc> --outdir benchmarks_v2/results/rg/warp_md
python benchmarks_v2/plans/rg/mdanalysis.py --top <top.pdb> --traj <traj.xtc> --outdir benchmarks_v2/results/rg/mdanalysis
python benchmarks_v2/plans/rg/mdtraj.py --top <top.pdb> --traj <traj.xtc> --outdir benchmarks_v2/results/rg/mdtraj
python benchmarks_v2/plans/rg/gmx.py --top <top.pdb> --traj <traj.xtc> --gromacs-tpr <topol.tpr> --outdir benchmarks_v2/results/rg/gmx
python benchmarks_v2/plans/rg/warp_md_gpu.py --top <top.pdb> --traj <traj.xtc> --outdir benchmarks_v2/results/rg/warp_md_gpu
```

Each tool writes:
- `summary.csv` (speed + error summary + checksums)
- `values/*.csv` (real per-frame series when available)
- `plots/*.png` (time series plots when series available)
- `raw/*` (source JSON/NPZ)
- `status.json`
