from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, TextIO, Tuple

GPU_PLATFORMS = {"CUDA", "OPENCL", "HIP"}
MAX_FORCE_GROUPS = 32


DEFAULT_PROTOCOL: Dict[str, Any] = {
    "prefix": "eq_npt",
    "temperature": 300.0,
    "pressure": 1.0,
    "friction": 10.0,
    "eq_timestep_fs": 10.0,
    "prod_timestep_fs": 20.0,
    "cutoff_nm": 1.2,
    "eq_ns": 50.0,
    "prod_ns": 0.0,
    "production_ensemble": "npt",
    "platform": "CUDA",
    "precision": "mixed",
    "device": "0",
    "cpu_threads": None,
    "seed": None,
    "minimize_iterations": 20_000,
    "barostat_frequency": 25,
    "report_interval_steps": 1_000,
    "trajectory_interval_steps": 10_000,
    "checkpoint_interval_steps": 50_000,
    "energy_interval_steps": 1_000,
    "status_interval_steps": 50_000,
    "trajectory_format": "xtc",
    "energy_log": True,
    "epsilon_r": None,
    "defines_file": None,
    "defines": [],
    "dry_run": False,
}


def _imports():
    try:
        import martini_openmm as martini  # type: ignore
        import openmm as mm  # type: ignore
        import openmm.app as app  # type: ignore
        import openmm.unit as unit  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "martini_openmm and openmm are required for real Martini/OpenMM runs. "
            "Install martini-openmm plus OpenMM in the runner environment."
        ) from exc
    return martini, mm, app, unit


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _steps_from_ns(duration_ns: float, timestep_fs: float) -> int:
    return int(round(duration_ns * 1_000_000.0 / timestep_fs))


def _merged_protocol(protocol: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(DEFAULT_PROTOCOL)
    if protocol:
        for key, value in protocol.items():
            if value is not None:
                merged[key] = value
    return merged


def _read_defines(protocol: Dict[str, Any]) -> Dict[str, bool]:
    defines: Dict[str, bool] = {}
    defines_file = protocol.get("defines_file")
    if defines_file:
        with Path(defines_file).open(encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.split("#", 1)[0].strip()
                if line:
                    defines[line] = True
    for value in protocol.get("defines") or []:
        text = str(value).strip()
        if text:
            defines[text] = True
    return defines


def _make_platform(mm: Any, protocol: Dict[str, Any]) -> Tuple[Any, Dict[str, str]]:
    platform = mm.Platform.getPlatformByName(str(protocol["platform"]))
    platform_name = platform.getName().upper()
    properties: Dict[str, str] = {}
    if platform_name in GPU_PLATFORMS:
        properties["Precision"] = str(protocol["precision"])
        if protocol.get("device") is not None:
            properties["DeviceIndex"] = str(protocol["device"])
    elif platform_name == "CPU" and protocol.get("cpu_threads") is not None:
        properties["Threads"] = str(protocol["cpu_threads"])
    return platform, properties


def _make_integrator(mm: Any, unit: Any, protocol: Dict[str, Any], timestep_fs: float) -> Any:
    integrator = mm.LangevinIntegrator(
        float(protocol["temperature"]) * unit.kelvin,
        float(protocol["friction"]) / unit.picosecond,
        timestep_fs * unit.femtoseconds,
    )
    if protocol.get("seed") is not None:
        integrator.setRandomNumberSeed(int(protocol["seed"]))
    return integrator


def _load_system(martini: Any, app: Any, unit: Any, gro: Path, top: Path, protocol: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    conf = app.GromacsGroFile(str(gro))
    top_kwargs: Dict[str, Any] = {"periodicBoxVectors": conf.getPeriodicBoxVectors()}
    defines = _read_defines(protocol)
    if defines:
        top_kwargs["defines"] = defines
    if protocol.get("epsilon_r") is not None:
        top_kwargs["epsilon_r"] = float(protocol["epsilon_r"])
    martini_top = martini.MartiniTopFile(str(top), **top_kwargs)
    system = martini_top.create_system(
        nonbonded_cutoff=float(protocol["cutoff_nm"]) * unit.nanometers
    )
    return conf, martini_top, system


def _add_barostat(mm: Any, unit: Any, system: Any, protocol: Dict[str, Any]) -> None:
    system.addForce(
        mm.MonteCarloBarostat(
            float(protocol["pressure"]) * unit.atmospheres,
            float(protocol["temperature"]) * unit.kelvin,
            int(protocol["barostat_frequency"]),
        )
    )


def _remove_barostats(mm: Any, system: Any) -> None:
    for index in reversed(range(system.getNumForces())):
        if isinstance(system.getForce(index), mm.MonteCarloBarostat):
            system.removeForce(index)


def _assign_force_groups(system: Any) -> int:
    n_forces = system.getNumForces()
    n_groups = min(n_forces, MAX_FORCE_GROUPS)
    for index in range(n_groups):
        system.getForce(index).setForceGroup(index)
    if n_forces > MAX_FORCE_GROUPS:
        print(
            f"WARNING: system has {n_forces} forces, only the first {MAX_FORCE_GROUPS} "
            "will be energy-decomposed.",
            file=sys.stderr,
        )
    return n_groups


def _add_reporters(app: Any, simulation: Any, protocol: Dict[str, Any], outdir: Path, phase: str) -> Optional[Path]:
    prefix = outdir / phase
    trajectory_path: Optional[Path] = None
    trajectory_format = str(protocol["trajectory_format"])
    if trajectory_format == "xtc":
        if not hasattr(app, "XTCReporter"):
            raise RuntimeError("OpenMM app.XTCReporter is unavailable; choose trajectory_format=dcd")
        trajectory_path = prefix.with_suffix(".xtc")
        simulation.reporters.append(app.XTCReporter(str(trajectory_path), int(protocol["trajectory_interval_steps"])))
    elif trajectory_format == "dcd":
        trajectory_path = prefix.with_suffix(".dcd")
        simulation.reporters.append(app.DCDReporter(str(trajectory_path), int(protocol["trajectory_interval_steps"])))
    simulation.reporters.append(
        app.StateDataReporter(
            str(prefix.with_suffix(".log")),
            int(protocol["report_interval_steps"]),
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            density=True,
            speed=True,
            volume=True,
            separator="\t",
        )
    )
    simulation.reporters.append(
        app.CheckpointReporter(str(prefix.with_suffix(".chk")), int(protocol["checkpoint_interval_steps"]))
    )
    return trajectory_path


def _write_pdb(app: Any, simulation: Any, path: Path, enforce_periodic_box: bool = False) -> None:
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=enforce_periodic_box)
    simulation.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    with path.open("w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), handle, keepIds=True)


def _open_energy_log(system: Any, outdir: Path, phase: str, n_energy_groups: int) -> TextIO:
    handle = (outdir / f"{phase}_ener.log").open("w", encoding="utf-8")
    handle.write("# Energy decomposition log\n")
    handle.write("# Columns are tab-separated. Energies are in kJ/mol.\n")
    handle.write("# time_ps")
    for index in range(n_energy_groups):
        force = system.getForce(index)
        handle.write(f"\tgroup_{index}_{type(force).__name__}")
    handle.write("\n")
    return handle


def _log_energy_decomposition(unit: Any, simulation: Any, handle: TextIO, n_energy_groups: int) -> None:
    state = simulation.context.getState(getEnergy=True)
    handle.write(f"{state.getTime().value_in_unit(unit.picoseconds):.6f}")
    for index in range(n_energy_groups):
        energy = simulation.context.getState(getEnergy=True, groups=1 << index).getPotentialEnergy()
        handle.write(f"\t{energy.value_in_unit(unit.kilojoules_per_mole):.8f}")
    handle.write("\n")
    handle.flush()


def _run_steps(
    unit: Any,
    simulation: Any,
    total_steps: int,
    label: str,
    protocol: Dict[str, Any],
    energy_handle: Optional[TextIO] = None,
    n_energy_groups: int = 0,
) -> None:
    if total_steps <= 0:
        print(f"Skipping {label}: 0 steps")
        return
    print(f"Running {label}: {total_steps:,} steps")
    completed = 0
    started = time.time()
    next_status = min(int(protocol["status_interval_steps"]), total_steps)
    step_interval = int(protocol["energy_interval_steps"] if energy_handle is not None else protocol["status_interval_steps"])
    while completed < total_steps:
        chunk = min(step_interval, total_steps - completed)
        simulation.step(chunk)
        completed += chunk
        if energy_handle is not None:
            _log_energy_decomposition(unit, simulation, energy_handle, n_energy_groups)
        if completed >= next_status or completed == total_steps:
            print(f"  {label}: {completed:,}/{total_steps:,} steps completed in {time.time() - started:.1f} s")
            next_status += int(protocol["status_interval_steps"])


def _copy_state(source: Any, target: Any) -> None:
    state = source.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    a, b, c = state.getPeriodicBoxVectors()
    target.context.setPeriodicBoxVectors(a, b, c)
    target.context.setPositions(state.getPositions())
    target.context.setVelocities(state.getVelocities())


def _save_outputs(simulation: Any, outdir: Path, phase: str, app: Any) -> None:
    _write_pdb(app, simulation, outdir / f"{phase}_last.pdb", enforce_periodic_box=True)
    simulation.saveCheckpoint(str(outdir / f"{phase}_last.chk"))
    simulation.saveState(str(outdir / f"{phase}_last.xml"))


def run_martini_openmm(gro: Path, top: Path, outdir: Path, protocol: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = _merged_protocol(protocol)
    if merged.get("dry_run"):
        outdir.mkdir(parents=True, exist_ok=True)
        trajectory = outdir / f"{merged['prefix']}.{merged['trajectory_format'] if merged['trajectory_format'] != 'none' else 'txt'}"
        trajectory.write_text("dry-run trajectory placeholder\n", encoding="utf-8")
        return {"status": "completed", "trajectory": str(trajectory), "dry_run": True}

    martini, mm, app, unit = _imports()
    outdir.mkdir(parents=True, exist_ok=True)
    conf, martini_top, system = _load_system(martini, app, unit, gro, top, merged)
    _add_barostat(mm, unit, system, merged)
    n_energy_groups = _assign_force_groups(system)
    platform, properties = _make_platform(mm, merged)
    eq_integrator = _make_integrator(mm, unit, merged, float(merged["eq_timestep_fs"]))
    simulation = app.Simulation(martini_top.topology, system, eq_integrator, platform, properties)
    simulation.context.setPositions(conf.getPositions())
    if merged.get("seed") is None:
        simulation.context.setVelocitiesToTemperature(float(merged["temperature"]) * unit.kelvin)
    else:
        simulation.context.setVelocitiesToTemperature(float(merged["temperature"]) * unit.kelvin, int(merged["seed"]))

    _write_pdb(app, simulation, outdir / "beforemin.pdb")
    simulation.minimizeEnergy(maxIterations=int(merged["minimize_iterations"]))
    _write_pdb(app, simulation, outdir / "min.pdb")

    eq_trajectory = _add_reporters(app, simulation, merged, outdir, str(merged["prefix"]))
    eq_steps = _steps_from_ns(float(merged["eq_ns"]), float(merged["eq_timestep_fs"]))
    if merged.get("energy_log"):
        with _open_energy_log(system, outdir, str(merged["prefix"]), n_energy_groups) as handle:
            _run_steps(unit, simulation, eq_steps, "NPT equilibration", merged, handle, n_energy_groups)
    else:
        _run_steps(unit, simulation, eq_steps, "NPT equilibration", merged)
    _save_outputs(simulation, outdir, str(merged["prefix"]), app)

    trajectory = eq_trajectory
    prod_steps = _steps_from_ns(float(merged["prod_ns"]), float(merged["prod_timestep_fs"]))
    if prod_steps > 0:
        phase = f"prod_{merged['production_ensemble']}"
        if merged["production_ensemble"] == "nvt":
            _remove_barostats(mm, system)
        prod_integrator = _make_integrator(mm, unit, merged, float(merged["prod_timestep_fs"]))
        prod = app.Simulation(martini_top.topology, system, prod_integrator, platform, properties)
        _copy_state(simulation, prod)
        trajectory = _add_reporters(app, prod, merged, outdir, phase)
        _run_steps(unit, prod, prod_steps, f"production {str(merged['production_ensemble']).upper()}", merged)
        _save_outputs(prod, outdir, phase, app)

    return {
        "status": "completed",
        "trajectory": str(trajectory) if trajectory is not None else None,
        "outdir": str(outdir),
        "platform": platform.getName(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Martini/OpenMM minimization, equilibration, and optional production.")
    parser.add_argument("--gro", required=True, type=Path)
    parser.add_argument("--top", required=True, type=Path)
    parser.add_argument("--outdir", type=Path, default=Path("run"))
    parser.add_argument("--prefix", default=DEFAULT_PROTOCOL["prefix"])
    parser.add_argument("--temperature", type=_positive_float, default=DEFAULT_PROTOCOL["temperature"])
    parser.add_argument("--pressure", type=_positive_float, default=DEFAULT_PROTOCOL["pressure"])
    parser.add_argument("--friction", type=_non_negative_float, default=DEFAULT_PROTOCOL["friction"])
    parser.add_argument("--eq-timestep-fs", type=_positive_float, default=DEFAULT_PROTOCOL["eq_timestep_fs"])
    parser.add_argument("--prod-timestep-fs", type=_positive_float, default=DEFAULT_PROTOCOL["prod_timestep_fs"])
    parser.add_argument("--cutoff-nm", type=_positive_float, default=DEFAULT_PROTOCOL["cutoff_nm"])
    parser.add_argument("--eq-ns", type=_non_negative_float, default=DEFAULT_PROTOCOL["eq_ns"])
    parser.add_argument("--prod-ns", type=_non_negative_float, default=DEFAULT_PROTOCOL["prod_ns"])
    parser.add_argument("--production-ensemble", choices=["npt", "nvt"], default=DEFAULT_PROTOCOL["production_ensemble"])
    parser.add_argument("--platform", default=DEFAULT_PROTOCOL["platform"])
    parser.add_argument("--precision", choices=["single", "mixed", "double"], default=DEFAULT_PROTOCOL["precision"])
    parser.add_argument("--device", default=DEFAULT_PROTOCOL["device"])
    parser.add_argument("--cpu-threads", type=_positive_int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--minimize-iterations", type=_positive_int, default=DEFAULT_PROTOCOL["minimize_iterations"])
    parser.add_argument("--barostat-frequency", type=_positive_int, default=DEFAULT_PROTOCOL["barostat_frequency"])
    parser.add_argument("--report-interval-steps", type=_positive_int, default=DEFAULT_PROTOCOL["report_interval_steps"])
    parser.add_argument("--trajectory-interval-steps", type=_positive_int, default=DEFAULT_PROTOCOL["trajectory_interval_steps"])
    parser.add_argument("--checkpoint-interval-steps", type=_positive_int, default=DEFAULT_PROTOCOL["checkpoint_interval_steps"])
    parser.add_argument("--energy-interval-steps", type=_positive_int, default=DEFAULT_PROTOCOL["energy_interval_steps"])
    parser.add_argument("--status-interval-steps", type=_positive_int, default=DEFAULT_PROTOCOL["status_interval_steps"])
    parser.add_argument("--trajectory-format", choices=["xtc", "dcd", "none"], default=DEFAULT_PROTOCOL["trajectory_format"])
    parser.add_argument("--no-energy-log", action="store_true")
    parser.add_argument("--epsilon-r", type=float)
    parser.add_argument("--defines-file")
    parser.add_argument("--define", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-out")
    return parser


def _protocol_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "prefix": args.prefix,
        "temperature": args.temperature,
        "pressure": args.pressure,
        "friction": args.friction,
        "eq_timestep_fs": args.eq_timestep_fs,
        "prod_timestep_fs": args.prod_timestep_fs,
        "cutoff_nm": args.cutoff_nm,
        "eq_ns": args.eq_ns,
        "prod_ns": args.prod_ns,
        "production_ensemble": args.production_ensemble,
        "platform": args.platform,
        "precision": args.precision,
        "device": args.device,
        "cpu_threads": args.cpu_threads,
        "seed": args.seed,
        "minimize_iterations": args.minimize_iterations,
        "barostat_frequency": args.barostat_frequency,
        "report_interval_steps": args.report_interval_steps,
        "trajectory_interval_steps": args.trajectory_interval_steps,
        "checkpoint_interval_steps": args.checkpoint_interval_steps,
        "energy_interval_steps": args.energy_interval_steps,
        "status_interval_steps": args.status_interval_steps,
        "trajectory_format": args.trajectory_format,
        "energy_log": not args.no_energy_log,
        "epsilon_r": args.epsilon_r,
        "defines_file": args.defines_file,
        "defines": args.define,
        "dry_run": args.dry_run,
    }


def run_cli(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_martini_openmm(args.gro, args.top, args.outdir, _protocol_from_args(args))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(result, indent=2))
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    try:
        return run_cli(argv)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
