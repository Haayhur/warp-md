from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
SCRIPT = REPO / 'scripts/bench/benchmark_structural_metrics.py'

def _python_exec() -> str:
    v = REPO / '.venv' / 'bin' / 'python'
    if v.exists():
        return str(v)
    return sys.executable

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', required=True)
    ap.add_argument('--traj', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--gromacs-tpr', default=None)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_json = outdir / 'raw.json'
    raw_npz = outdir / 'series.npz'
    cmd = [_python_exec(), str(SCRIPT), '--top', args.top, '--traj', args.traj, '--json-out', str(raw_json), '--repeats', str(args.repeats)]
    help_out = subprocess.run([_python_exec(), str(SCRIPT), '--help'], capture_output=True, text=True, check=False)
    h = (help_out.stdout or '') + '\n' + (help_out.stderr or '')
    if '--device' in h:
        cmd.extend(['--device', args.device])
    if '--series-npz-out' in h:
        cmd.extend(['--series-npz-out', str(raw_npz)])
    if HERE.name == 'gmx.py':
        if '--with-gromacs' in h:
            cmd.append('--with-gromacs')
        if '--gromacs-tpr' in h and args.gromacs_tpr:
            cmd.extend(['--gromacs-tpr', str(args.gromacs_tpr)])
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=False)
    wall = time.perf_counter() - t0
    (outdir / 'stdout.txt').write_text(proc.stdout or '', encoding='utf-8')
    (outdir / 'stderr.txt').write_text(proc.stderr or '', encoding='utf-8')
    (outdir / 'wall_clock_sec.txt').write_text(f'{wall:.6f}\n', encoding='utf-8')
    raise SystemExit(proc.returncode)

if __name__ == '__main__':
    main()
