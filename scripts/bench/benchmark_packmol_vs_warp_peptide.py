#!/usr/bin/env python3
"""Pinned-core benchmark: warp-pack vs Packmol on a peptide packing case."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


WARP_PROFILE_RE = re.compile(
    r"warp-pack profile \(s\): total=([0-9.]+).*movebad=([0-9.]+).*gencan=([0-9.]+)"
)
PACKMOL_SUCCESS_RE = re.compile(r"Success!")
PACKMOL_IMPERFECT_RE = re.compile(r"ENDED WITHOUT PERFECT PACKING", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark warp-pack vs Packmol on a peptide case. "
            "Writes per-run packed PDBs for direct inspection."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/bench/packmol_vs_warp_peptide",
        help="Output directory for configs, logs, packed PDBs, and summary JSON.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional explicit summary JSON path (default: <output-dir>/summary.json).",
    )
    parser.add_argument(
        "--peptide-pdb",
        default=None,
        help="Existing peptide PDB input. If omitted, script builds one via warp-pep.",
    )
    parser.add_argument(
        "--sequence",
        default="ACDEFGHIKLMN",
        help="Peptide sequence used when --peptide-pdb is omitted.",
    )
    parser.add_argument(
        "--no-center-peptide",
        action="store_true",
        help="Disable coordinate recentering before packing.",
    )
    parser.add_argument(
        "--warp-pep-bin",
        default="target/debug/warp-pep",
        help="Path to warp-pep binary used for sequence-to-PDB generation.",
    )
    parser.add_argument(
        "--warp-pep-preset",
        default="alpha-helix",
        help="Optional warp-pep backbone preset when generating from --sequence.",
    )
    parser.add_argument(
        "--warp-pack-bin",
        default="target/debug/warp-pack",
        help="Path to warp-pack binary.",
    )
    parser.add_argument(
        "--packmol-bin",
        default="tmp/packmol/packmol",
        help="Path to Packmol executable.",
    )
    parser.add_argument("--taskset-bin", default="taskset", help="Path to taskset executable.")
    parser.add_argument("--no-pin", action="store_true", help="Disable CPU-core pinning.")
    parser.add_argument("--cpu-core", type=int, default=0, help="CPU core index for pinning.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Untimed warmup runs per tool.")
    parser.add_argument("--runs", type=int, default=10, help="Timed runs per tool.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=600.0,
        help="Timeout per subprocess run.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for both tools.")
    parser.add_argument("--count", type=int, default=14, help="Number of peptide copies.")
    parser.add_argument("--box", type=float, default=32.0, help="Orthorhombic box size (A).")
    parser.add_argument("--distance", type=float, default=2.1, help="Minimum distance / tolerance (A).")
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--movefrac", type=float, default=0.2)
    parser.add_argument("--nloop", type=int, default=12)
    parser.add_argument("--nloop0", type=int, default=2)
    parser.add_argument("--gencan-maxit", type=int, default=6)
    parser.add_argument("--gencan-step", type=float, default=0.1)
    parser.add_argument("--discale", type=float, default=1.1)
    parser.add_argument("--precision", type=float, default=0.001)
    return parser.parse_args()


def resolve_executable(raw: str) -> str:
    path = Path(raw)
    if path.exists():
        return str(path.resolve())
    found = shutil.which(raw)
    if found:
        return found
    raise FileNotFoundError(f"executable not found: {raw}")


def p95(values: List[float]) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = 0.95 * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def stat_block(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "stdev": None,
            "min": None,
            "max": None,
        }
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": p95(values),
        "stdev": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def write_packmol_input(
    inp_path: Path,
    peptide_pdb: Path,
    output_pdb: Path,
    count: int,
    box: float,
    distance: float,
    seed: int,
) -> None:
    text = "\n".join(
        [
            f"tolerance {distance}",
            "filetype pdb",
            f"output {output_pdb}",
            f"seed {seed}",
            f"structure {peptide_pdb}",
            f"  number {count}",
            f"  inside box 0.0 0.0 0.0 {box} {box} {box}",
            "end structure",
            "",
        ]
    )
    inp_path.write_text(text, encoding="utf-8")


def recenter_pdb(input_path: Path, output_path: Path) -> Dict[str, Any]:
    lines = input_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    parsed: List[tuple[int, float, float, float]] = []
    for idx, line in enumerate(lines):
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            parsed.append((idx, x, y, z))

    if not parsed:
        raise RuntimeError(f"no ATOM/HETATM coordinates found in {input_path}")

    xs = [item[1] for item in parsed]
    ys = [item[2] for item in parsed]
    zs = [item[3] for item in parsed]
    center_x = 0.5 * (min(xs) + max(xs))
    center_y = 0.5 * (min(ys) + max(ys))
    center_z = 0.5 * (min(zs) + max(zs))

    for idx, x, y, z in parsed:
        nx = x - center_x
        ny = y - center_y
        nz = z - center_z
        line = lines[idx]
        tail = line[54:] if len(line) > 54 else ""
        lines[idx] = f"{line[:30]}{nx:8.3f}{ny:8.3f}{nz:8.3f}{tail}"

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "center_shift": [center_x, center_y, center_z],
        "span_before": [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)],
        "atom_count": len(parsed),
    }


def run_subprocess(
    cmd: List[str],
    cwd: Path,
    timeout_seconds: float,
    env: Optional[Dict[str, str]] = None,
    stdin_path: Optional[Path] = None,
) -> tuple[float, subprocess.CompletedProcess[str]]:
    start = time.perf_counter()
    stdin_handle = None
    try:
        if stdin_path is not None:
            stdin_handle = stdin_path.open("r", encoding="utf-8")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                stdin=stdin_handle,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = float(time.perf_counter() - start)
            raise RuntimeError(
                f"command timed out after {timeout_seconds}s: {' '.join(cmd)} "
                f"(elapsed={elapsed:.3f}s)"
            ) from exc
    finally:
        if stdin_handle is not None:
            stdin_handle.close()
    elapsed = float(time.perf_counter() - start)
    return elapsed, proc


def write_log(path: Path, cmd: List[str], elapsed: float, proc: subprocess.CompletedProcess[str]) -> None:
    lines = [
        f"$ {' '.join(cmd)}",
        f"[exit={proc.returncode}] [elapsed={elapsed:.6f}s]",
        "",
        "=== stdout ===",
        proc.stdout or "",
        "",
        "=== stderr ===",
        proc.stderr or "",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def classify_packmol_run(
    proc: subprocess.CompletedProcess[str],
    merged_output: str,
    output_pdb: Path,
) -> Optional[Dict[str, Any]]:
    if proc.returncode == 0 and PACKMOL_SUCCESS_RE.search(merged_output) is not None:
        return {
            "status": "success",
            "converged": True,
            "forced_output": None,
        }
    if proc.returncode == 173 and PACKMOL_IMPERFECT_RE.search(merged_output) is not None and output_pdb.exists():
        forced_path = output_pdb.with_name(output_pdb.name + "_FORCED")
        return {
            "status": "imperfect",
            "converged": False,
            "forced_output": str(forced_path) if forced_path.exists() else None,
        }
    return None


def ensure_peptide_input(
    args: argparse.Namespace,
    inputs_dir: Path,
    repo_root: Path,
) -> tuple[Path, Optional[str]]:
    peptide_path = inputs_dir / "peptide_input.pdb"
    if args.peptide_pdb:
        src = Path(args.peptide_pdb).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"peptide pdb not found: {src}")
        peptide_path.write_text(src.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        return peptide_path, None

    warp_pep_bin = resolve_executable(args.warp_pep_bin)
    cmd = [warp_pep_bin, "build", "--sequence", args.sequence, "--output", str(peptide_path)]
    preset = str(args.warp_pep_preset).strip()
    if preset:
        cmd.extend(["--preset", preset])
    elapsed, proc = run_subprocess(cmd, repo_root, timeout_seconds=args.timeout_seconds)
    log_path = inputs_dir / "warp_pep.log"
    write_log(log_path, cmd, elapsed, proc)
    if proc.returncode != 0 or not peptide_path.exists():
        raise RuntimeError(f"warp-pep failed; inspect {log_path}")
    return peptide_path, warp_pep_bin


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir).resolve()
    inputs_dir = output_dir / "inputs"
    warp_dir = output_dir / "warp_pack"
    packmol_dir = output_dir / "packmol"
    for path in (output_dir, inputs_dir, warp_dir, packmol_dir):
        path.mkdir(parents=True, exist_ok=True)

    warp_pack_bin = resolve_executable(args.warp_pack_bin)
    packmol_bin = resolve_executable(args.packmol_bin)

    taskset_bin = None
    if not args.no_pin:
        taskset_bin = resolve_executable(args.taskset_bin)

    peptide_pdb_raw, warp_pep_bin = ensure_peptide_input(args, inputs_dir, repo_root)
    recenter_info: Optional[Dict[str, Any]] = None
    peptide_pdb = peptide_pdb_raw
    if not args.no_center_peptide:
        centered_path = inputs_dir / "peptide_input_centered.pdb"
        recenter_info = recenter_pdb(peptide_pdb_raw, centered_path)
        peptide_pdb = centered_path

    warp_cfg = {
        "box": {"size": [args.box, args.box, args.box], "shape": "orthorhombic"},
        "structures": [{"path": str(peptide_pdb), "count": args.count}],
        "seed": args.seed,
        "min_distance": args.distance,
        "max_attempts": args.max_attempts,
        "avoid_overlap": True,
        "disable_movebad": False,
        "movefrac": args.movefrac,
        "nloop": args.nloop,
        "nloop0": args.nloop0,
        "gencan_maxit": args.gencan_maxit,
        "gencan_step": args.gencan_step,
        "discale": args.discale,
        "precision": args.precision,
        "check": False,
    }
    warp_cfg_path = inputs_dir / "warp_config.json"
    warp_cfg_path.write_text(json.dumps(warp_cfg, indent=2), encoding="utf-8")

    cmd_prefix: List[str] = []
    if taskset_bin:
        cmd_prefix = [taskset_bin, "-c", str(args.cpu_core)]

    warp_runs: List[Dict[str, Any]] = []
    packmol_runs: List[Dict[str, Any]] = []

    warp_env = dict(os.environ)
    warp_env["WARP_PACK_PROFILE"] = "1"

    for idx in range(args.warmup_runs):
        run_name = f"warmup_{idx + 1:02d}"
        out_pdb = warp_dir / f"{run_name}.pdb"
        log_path = warp_dir / f"{run_name}.log"
        cmd = cmd_prefix + [
            warp_pack_bin,
            "--config",
            str(warp_cfg_path),
            "--output",
            str(out_pdb),
            "--format",
            "pdb",
        ]
        elapsed, proc = run_subprocess(cmd, repo_root, args.timeout_seconds, env=warp_env)
        write_log(log_path, cmd, elapsed, proc)
        if proc.returncode != 0:
            raise RuntimeError(f"warp-pack warmup failed; inspect {log_path}")

    for idx in range(args.warmup_runs):
        run_name = f"warmup_{idx + 1:02d}"
        out_pdb = packmol_dir / f"{run_name}.pdb"
        inp_path = packmol_dir / f"{run_name}.inp"
        log_path = packmol_dir / f"{run_name}.log"
        write_packmol_input(
            inp_path=inp_path,
            peptide_pdb=peptide_pdb,
            output_pdb=out_pdb,
            count=args.count,
            box=args.box,
            distance=args.distance,
            seed=args.seed,
        )
        cmd = cmd_prefix + [packmol_bin]
        elapsed, proc = run_subprocess(
            cmd,
            cwd=repo_root,
            timeout_seconds=args.timeout_seconds,
            stdin_path=inp_path,
        )
        write_log(log_path, cmd, elapsed, proc)
        merged = (proc.stdout or "") + (proc.stderr or "")
        packmol_status = classify_packmol_run(proc, merged, out_pdb)
        if packmol_status is None:
            raise RuntimeError(f"packmol warmup failed; inspect {log_path}")

    for idx in range(args.runs):
        run_name = f"run_{idx + 1:02d}"
        out_pdb = warp_dir / f"{run_name}.pdb"
        log_path = warp_dir / f"{run_name}.log"
        cmd = cmd_prefix + [
            warp_pack_bin,
            "--config",
            str(warp_cfg_path),
            "--output",
            str(out_pdb),
            "--format",
            "pdb",
        ]
        elapsed, proc = run_subprocess(cmd, repo_root, args.timeout_seconds, env=warp_env)
        write_log(log_path, cmd, elapsed, proc)
        merged = (proc.stdout or "") + (proc.stderr or "")
        match = WARP_PROFILE_RE.search(merged)
        if proc.returncode != 0 or not match:
            raise RuntimeError(f"warp-pack run failed; inspect {log_path}")
        warp_runs.append(
            {
                "run": idx + 1,
                "wall": elapsed,
                "profile_total": float(match.group(1)),
                "profile_movebad": float(match.group(2)),
                "profile_gencan": float(match.group(3)),
                "output_pdb": str(out_pdb),
                "log": str(log_path),
            }
        )

    for idx in range(args.runs):
        run_name = f"run_{idx + 1:02d}"
        out_pdb = packmol_dir / f"{run_name}.pdb"
        inp_path = packmol_dir / f"{run_name}.inp"
        log_path = packmol_dir / f"{run_name}.log"
        write_packmol_input(
            inp_path=inp_path,
            peptide_pdb=peptide_pdb,
            output_pdb=out_pdb,
            count=args.count,
            box=args.box,
            distance=args.distance,
            seed=args.seed,
        )
        cmd = cmd_prefix + [packmol_bin]
        elapsed, proc = run_subprocess(
            cmd,
            cwd=repo_root,
            timeout_seconds=args.timeout_seconds,
            stdin_path=inp_path,
        )
        write_log(log_path, cmd, elapsed, proc)
        merged = (proc.stdout or "") + (proc.stderr or "")
        packmol_status = classify_packmol_run(proc, merged, out_pdb)
        if packmol_status is None:
            raise RuntimeError(f"packmol run failed; inspect {log_path}")
        packmol_runs.append(
            {
                "run": idx + 1,
                "wall": elapsed,
                "status": packmol_status["status"],
                "converged": packmol_status["converged"],
                "output_pdb": str(out_pdb),
                "forced_output_pdb": packmol_status["forced_output"],
                "inp": str(inp_path),
                "log": str(log_path),
            }
        )

    warp_wall = [float(item["wall"]) for item in warp_runs]
    warp_total = [float(item["profile_total"]) for item in warp_runs]
    warp_movebad = [float(item["profile_movebad"]) for item in warp_runs]
    warp_gencan = [float(item["profile_gencan"]) for item in warp_runs]
    packmol_wall = [float(item["wall"]) for item in packmol_runs]

    summary: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case": {
            "peptide_pdb_source": str(peptide_pdb_raw),
            "peptide_pdb_used": str(peptide_pdb),
            "sequence": args.sequence if args.peptide_pdb is None else None,
            "warp_pep_preset": args.warp_pep_preset if args.peptide_pdb is None else None,
            "count": args.count,
            "box": [args.box, args.box, args.box],
            "distance": args.distance,
            "seed": args.seed,
            "cpu_affinity": "unpinned" if args.no_pin else f"core {args.cpu_core}",
            "centered_peptide": not args.no_center_peptide,
            "centering": recenter_info,
            "warmup_runs": args.warmup_runs,
            "timed_runs": args.runs,
        },
        "binaries": {
            "warp_pep": warp_pep_bin,
            "warp_pack": warp_pack_bin,
            "packmol": packmol_bin,
            "taskset": taskset_bin,
        },
        "warp_pack": {
            "wall": stat_block(warp_wall),
            "profile_total": stat_block(warp_total),
            "profile_movebad": stat_block(warp_movebad),
            "profile_gencan": stat_block(warp_gencan),
            "runs": warp_runs,
        },
        "packmol": {
            "wall": stat_block(packmol_wall),
            "converged_runs": sum(1 for item in packmol_runs if bool(item.get("converged"))),
            "imperfect_runs": sum(1 for item in packmol_runs if item.get("status") == "imperfect"),
            "runs": packmol_runs,
        },
    }
    summary["speedup_packmol_over_warp_wall_mean"] = (
        summary["packmol"]["wall"]["mean"] / summary["warp_pack"]["wall"]["mean"]
        if summary["packmol"]["wall"]["mean"] is not None and summary["warp_pack"]["wall"]["mean"] is not None
        else None
    )
    summary["speedup_packmol_over_warp_wall_median"] = (
        summary["packmol"]["wall"]["median"] / summary["warp_pack"]["wall"]["median"]
        if summary["packmol"]["wall"]["median"] is not None and summary["warp_pack"]["wall"]["median"] is not None
        else None
    )

    summary_path = Path(args.json_out).resolve() if args.json_out else output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("benchmark complete")
    print(f"summary: {summary_path}")
    print(f"warp-pack pdbs: {warp_dir}")
    print(f"packmol pdbs: {packmol_dir}")
    print(
        "median wall (s): "
        f"warp-pack={summary['warp_pack']['wall']['median']:.3f}, "
        f"packmol={summary['packmol']['wall']['median']:.3f}, "
        f"speedup={summary['speedup_packmol_over_warp_wall_median']:.3f}x"
    )
    if summary["packmol"]["imperfect_runs"] > 0:
        print(
            "packmol status: "
            f"{summary['packmol']['converged_runs']} converged, "
            f"{summary['packmol']['imperfect_runs']} imperfect-best-effort"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
