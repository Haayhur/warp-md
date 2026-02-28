"""Programmatic runner API for agents.

Example usage:
    from warp_md.runner import run_analyses
    
    result = run_analyses({
        "version": "warp-md.agent.v1",
        "system": {"path": "topology.pdb"},
        "trajectory": {"path": "traj.xtc"},
        "analyses": [
            {"name": "rg", "selection": "protein"},
            {"name": "dssp", "mask": "protein"},
        ],
    })
    
    for r in result.results:
        print(r.analysis, r.status, r.out)
"""

from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import ValidationError

from .agent_schema import (
    AGENT_RESULT_SCHEMA_VERSION,
    CheckpointEvent,
    RunEnvelope,
    RunErrorEnvelope,
    RunRequest,
    RunResultEntry,
    RunSuccessEnvelope,
    classify_error,
    validate_run_request,
)
from .cli_api import _load_system, _load_trajectory
from .cli_builders import PLAN_BUILDERS
from .cli_config import _default_out
from .cli_output import _artifact_metadata, _save_output, _summary_from_output


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_plan_name(name: str) -> str:
    if name in PLAN_BUILDERS:
        return name
    alt = name.replace("-", "_")
    if alt in PLAN_BUILDERS:
        return alt
    raise ValueError(f"unknown analysis name: {name}")


def run_analyses(
    request: Union[str, Path, Dict[str, Any], RunRequest],
    *,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
    chunk_frames: Optional[int] = None,
    fail_fast: bool = True,
    on_checkpoint: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_analysis_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> RunEnvelope:
    """Run analyses from structured input and return typed result.
    
    Args:
        request: One of:
            - Path to JSON/YAML config file
            - Dict matching warp-md.agent.v1 schema
            - RunRequest Pydantic model
        output_dir: Override output directory (default: from request or temp)
        device: Override device (default: from request or "auto")
        chunk_frames: Override chunk size
        fail_fast: Stop on first failure (default True)
        on_checkpoint: Optional callback for progress events
        on_analysis_complete: Optional callback after each analysis
    
    Returns:
        RunEnvelope (success or error) with typed results
    """
    run_start = time.perf_counter()
    started_at = _now_utc_iso()
    results: List[Dict[str, Any]] = []
    warnings: List[str] = []
    
    # Parse request
    if isinstance(request, (str, Path)):
        path = Path(request)
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                cfg_raw = yaml.safe_load(path.read_text())
            except ImportError:
                raise ValueError("YAML config requires PyYAML installed")
        else:
            cfg_raw = json.loads(path.read_text())
    elif isinstance(request, dict):
        cfg_raw = request
    elif isinstance(request, RunRequest):
        cfg_raw = request.model_dump(mode="python")
    else:
        raise TypeError("request must be path, dict, or RunRequest")
    
    # Validate
    try:
        cfg = validate_run_request(cfg_raw)
    except ValidationError as exc:
        return RunErrorEnvelope(
            schema_version=AGENT_RESULT_SCHEMA_VERSION,
            status="error",
            exit_code=2,
            run_id=cfg_raw.get("run_id") if isinstance(cfg_raw, dict) else None,
            output_dir=output_dir,
            system=None,
            trajectory=None,
            analysis_count=0,
            started_at=started_at,
            finished_at=_now_utc_iso(),
            elapsed_ms=int((time.perf_counter() - run_start) * 1000),
            warnings=warnings,
            results=[],
            error={
                "code": "E_CONFIG_VALIDATION",
                "message": str(exc),
                "context": {},
            },
        )
    
    # Apply overrides
    run_id = cfg.run_id
    system_spec = cfg.system_spec
    traj_spec = cfg.trajectory_spec
    out_dir = output_dir or cfg.output_dir or tempfile.mkdtemp(prefix="warp_md_")
    default_device = device or cfg.device or "auto"
    default_chunk = chunk_frames or cfg.chunk_frames
    
    # Load system
    try:
        system = _load_system(system_spec)
    except Exception as exc:
        return RunErrorEnvelope(
            schema_version=AGENT_RESULT_SCHEMA_VERSION,
            status="error",
            exit_code=4,
            run_id=run_id,
            output_dir=out_dir,
            system=system_spec,
            trajectory=traj_spec,
            analysis_count=len(cfg.analyses),
            started_at=started_at,
            finished_at=_now_utc_iso(),
            elapsed_ms=int((time.perf_counter() - run_start) * 1000),
            warnings=warnings,
            results=[RunResultEntry(**r) for r in results],
            error={
                "code": "E_SYSTEM_LOAD",
                "message": str(exc),
                "context": {"stage": "system_load"},
            },
        )
    
    # Create output dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Load trajectory
    traj = None
    used_names: Dict[str, int] = {}
    
    for index, analysis in enumerate(cfg.analyses):
        item = analysis.model_dump(mode="python")
        requested_name = item["name"]
        
        try:
            name = _resolve_plan_name(requested_name)
        except Exception as exc:
            if fail_fast:
                return RunErrorEnvelope(
                    schema_version=AGENT_RESULT_SCHEMA_VERSION,
                    status="error",
                    exit_code=3,
                    run_id=run_id,
                    output_dir=out_dir,
                    system=system_spec,
                    trajectory=traj_spec,
                    analysis_count=len(cfg.analyses),
                    started_at=started_at,
                    finished_at=_now_utc_iso(),
                    elapsed_ms=int((time.perf_counter() - run_start) * 1000),
                    warnings=warnings,
                    results=[RunResultEntry(**r) for r in results],
                    error={
                        "code": "E_ANALYSIS_UNKNOWN",
                        "message": str(exc),
                        "context": {"analysis": requested_name, "index": index},
                    },
                )
            warnings.append(f"skipped unknown analysis: {requested_name}")
            continue
        
        out_path = item.get("out") or _default_out(name, out_dir, used_names)
        
        # Load trajectory on first use
        if traj is None:
            try:
                traj = _load_trajectory(traj_spec, system)
            except Exception as exc:
                return RunErrorEnvelope(
                    schema_version=AGENT_RESULT_SCHEMA_VERSION,
                    status="error",
                    exit_code=4,
                    run_id=run_id,
                    output_dir=out_dir,
                    system=system_spec,
                    trajectory=traj_spec,
                    analysis_count=len(cfg.analyses),
                    started_at=started_at,
                    finished_at=_now_utc_iso(),
                    elapsed_ms=int((time.perf_counter() - run_start) * 1000),
                    warnings=warnings,
                    results=[RunResultEntry(**r) for r in results],
                    error={
                        "code": "E_TRAJECTORY_LOAD",
                        "message": str(exc),
                        "context": {"analysis": name, "index": index},
                    },
                )
        
        # Build plan
        try:
            plan = PLAN_BUILDERS[name](system, item)
        except Exception as exc:
            if fail_fast:
                return RunErrorEnvelope(
                    schema_version=AGENT_RESULT_SCHEMA_VERSION,
                    status="error",
                    exit_code=3,
                    run_id=run_id,
                    output_dir=out_dir,
                    system=system_spec,
                    trajectory=traj_spec,
                    analysis_count=len(cfg.analyses),
                    started_at=started_at,
                    finished_at=_now_utc_iso(),
                    elapsed_ms=int((time.perf_counter() - run_start) * 1000),
                    warnings=warnings,
                    results=[RunResultEntry(**r) for r in results],
                    error={
                        "code": "E_ANALYSIS_SPEC",
                        "message": str(exc),
                        "context": {"analysis": name, "index": index},
                    },
                )
            warnings.append(f"skipped {name}: {exc}")
            continue
        
        # Run
        analysis_start = time.perf_counter()
        dev = item.get("device", default_device)
        chunk = item.get("chunk_frames", default_chunk)
        
        try:
            output = plan.run(traj, system, chunk_frames=chunk, device=dev)
            saved_path = _save_output(out_path, output)
        except Exception as exc:
            if fail_fast:
                return RunErrorEnvelope(
                    schema_version=AGENT_RESULT_SCHEMA_VERSION,
                    status="error",
                    exit_code=4,
                    run_id=run_id,
                    output_dir=out_dir,
                    system=system_spec,
                    trajectory=traj_spec,
                    analysis_count=len(cfg.analyses),
                    started_at=started_at,
                    finished_at=_now_utc_iso(),
                    elapsed_ms=int((time.perf_counter() - run_start) * 1000),
                    warnings=warnings,
                    results=[RunResultEntry(**r) for r in results],
                    error={
                        "code": "E_RUNTIME_EXEC",
                        "message": str(exc),
                        "context": {"analysis": name, "index": index},
                    },
                )
            warnings.append(f"{name} failed: {exc}")
            continue
        
        entry = _summary_from_output(output, name, Path(saved_path))
        entry["analysis"] = name
        entry["out"] = saved_path
        entry["status"] = "ok"
        entry["device"] = dev
        entry["chunk_frames"] = chunk
        entry["timing_ms"] = int((time.perf_counter() - analysis_start) * 1000)
        entry["artifact"] = _artifact_metadata(saved_path)
        results.append(entry)
        
        if on_analysis_complete:
            on_analysis_complete(entry)
    
    return RunSuccessEnvelope(
        schema_version=AGENT_RESULT_SCHEMA_VERSION,
        status="ok",
        exit_code=0,
        run_id=run_id,
        output_dir=out_dir,
        system=system_spec,
        trajectory=traj_spec,
        analysis_count=len(results),
        started_at=started_at,
        finished_at=_now_utc_iso(),
        elapsed_ms=int((time.perf_counter() - run_start) * 1000),
        warnings=warnings,
        results=[RunResultEntry(**r) for r in results],
    )


__all__ = ["run_analyses"]
