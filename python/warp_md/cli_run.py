from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import ValidationError

from .agent_schema import (
    AGENT_RESULT_SCHEMA_VERSION,
    render_agent_schema,
    validate_run_request,
)
from .cli_api import _load_system, _load_trajectory
from .cli_args import REGISTRY, add_shared_args
from .cli_builders import CLI_TO_PLAN, PLAN_BUILDERS
from .cli_config import _default_out, _load_config, example_config
from .cli_output import _artifact_metadata, _save_output, _summary_from_output
from .cli_parse import _load_system_from_args, _load_traj_from_args
from .cli_specs import SPEC_BUILDERS
from .pack.data import available_water_models, water_pdb
from .atlas_api import (
    DEFAULT_ATLAS_API_BASE_URL,
    AtlasApiError,
    download_atlas_trajectory,
)

_EXIT_OK = 0
_EXIT_CONFIG_VALIDATION = 2
_EXIT_ANALYSIS_SPEC = 3
_EXIT_RUNTIME_EXEC = 4
_EXIT_INTERNAL = 5


def _resolve_cli_version() -> str:
    for name in ("warp-md", "warp_md"):
        try:
            return _pkg_version(name)
        except PackageNotFoundError:
            continue
    return "dev"


class RunContractError(Exception):
    def __init__(
        self,
        exit_code: int,
        code: str,
        message: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        details: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.code = code
        self.context = context or {}
        self.details = details


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _emit_event(stream_mode: str, event: str, **payload: Any) -> None:
    if stream_mode != "ndjson":
        return
    record = {"event": event}
    record.update(payload)
    print(json.dumps(record, default=str), flush=True)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _normalize_validation_errors(errors: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    normalized: list[Dict[str, Any]] = []
    for item in errors:
        loc = item.get("loc", ())
        if isinstance(loc, (list, tuple)):
            field = ".".join(str(part) for part in loc)
        else:
            field = str(loc)
        entry: Dict[str, Any] = {
            "type": item.get("type", "value_error"),
            "field": field,
            "message": item.get("msg", "validation error"),
        }
        if "ctx" in item:
            entry["context"] = _json_safe(item.get("ctx"))
        normalized.append(entry)
    return normalized


def _analysis_error_details(index: int, analysis: str, message: str) -> list[Dict[str, Any]]:
    # Parse builder-style errors such as "rdf.sel_a selection is required".
    field = f"analyses.{index}"
    match = re.match(r"(?P<prefix>[a-zA-Z0-9_]+)\.(?P<param>[a-zA-Z0-9_]+)\b", message)
    if match:
        prefix = match.group("prefix")
        param = match.group("param")
        aliases = {analysis, analysis.replace("_", "-"), analysis.replace("-", "_")}
        if prefix in aliases:
            field = f"analyses.{index}.{param}"
    return [{"type": "value_error", "field": field, "message": message}]


def _progress_snapshot(completed: int, total: int, elapsed_ms: int) -> Dict[str, Any]:
    if total <= 0:
        return {
            "completed": completed,
            "total": total,
            "progress_pct": 0.0,
            "eta_ms": None,
        }
    progress_pct = round(100.0 * float(completed) / float(total), 3)
    if completed <= 0 or completed >= total:
        eta_ms: Optional[int] = 0 if completed >= total else None
    else:
        avg_ms = float(elapsed_ms) / float(completed)
        eta_ms = int(avg_ms * float(total - completed))
    return {
        "completed": completed,
        "total": total,
        "progress_pct": progress_pct,
        "eta_ms": eta_ms,
    }


def _build_success_envelope(
    *,
    status: str,
    run_id: Optional[str],
    output_dir: str,
    system_spec: Dict[str, Any],
    traj_spec: Dict[str, Any],
    started_at: str,
    finished_at: str,
    elapsed_ms: int,
    results: list[Dict[str, Any]],
    warnings: list[str],
) -> Dict[str, Any]:
    return {
        "schema_version": AGENT_RESULT_SCHEMA_VERSION,
        "status": status,
        "exit_code": _EXIT_OK,
        "run_id": run_id,
        "output_dir": output_dir,
        "system": system_spec,
        "trajectory": traj_spec,
        "analysis_count": len(results),
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_ms": elapsed_ms,
        "warnings": warnings,
        "results": results,
    }


def _build_error_envelope(
    *,
    exit_code: int,
    code: str,
    message: str,
    started_at: str,
    finished_at: str,
    elapsed_ms: int,
    run_id: Optional[str],
    output_dir: Optional[str],
    system_spec: Optional[Dict[str, Any]],
    traj_spec: Optional[Dict[str, Any]],
    results: list[Dict[str, Any]],
    warnings: list[str],
    context: Optional[Dict[str, Any]] = None,
    details: Optional[Any] = None,
    debug_errors: bool = False,
    exc: Optional[BaseException] = None,
) -> Dict[str, Any]:
    error: Dict[str, Any] = {
        "code": code,
        "message": message,
        "context": _json_safe(context or {}),
    }
    if details is not None:
        error["details"] = _json_safe(details)
    if debug_errors and exc is not None:
        error["traceback"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )

    return {
        "schema_version": AGENT_RESULT_SCHEMA_VERSION,
        "status": "error",
        "exit_code": exit_code,
        "run_id": run_id,
        "output_dir": output_dir,
        "system": system_spec,
        "trajectory": traj_spec,
        "analysis_count": len(results),
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_ms": elapsed_ms,
        "warnings": warnings,
        "results": results,
        "error": error,
    }


def _resolve_plan_name(name: str) -> str:
    if name in PLAN_BUILDERS:
        return name
    alt_name = name.replace("-", "_")
    if alt_name in PLAN_BUILDERS:
        return alt_name
    raise ValueError(f"unknown analysis name: {name}")


def run_config(
    config_path: str,
    dry_run: bool = False,
    stream: Optional[str] = None,
    debug_errors: bool = False,
) -> tuple[int, Dict[str, Any], str]:
    run_start = time.perf_counter()
    started_at = _now_utc_iso()
    results: list[Dict[str, Any]] = []
    warnings: list[str] = []
    stream_mode = stream or "none"
    run_id: Optional[str] = None
    output_dir: Optional[str] = None
    system_spec: Optional[Dict[str, Any]] = None
    traj_spec: Optional[Dict[str, Any]] = None

    try:
        cfg_raw = _load_config(config_path)
    except Exception as exc:
        envelope = _build_error_envelope(
            exit_code=_EXIT_CONFIG_VALIDATION,
            code="E_CONFIG_LOAD",
            message=str(exc),
            started_at=started_at,
            finished_at=_now_utc_iso(),
            elapsed_ms=int((time.perf_counter() - run_start) * 1000),
            run_id=run_id,
            output_dir=output_dir,
            system_spec=system_spec,
            traj_spec=traj_spec,
            results=results,
            warnings=warnings,
            context={"config_path": config_path},
            debug_errors=debug_errors,
            exc=exc,
        )
        _emit_event(stream_mode, "run_failed", final_envelope=envelope)
        return _EXIT_CONFIG_VALIDATION, envelope, stream_mode

    if isinstance(cfg_raw, dict):
        run_id_value = cfg_raw.get("run_id")
        if isinstance(run_id_value, str) and run_id_value.strip():
            run_id = run_id_value.strip()
        if stream is None:
            raw_stream = cfg_raw.get("stream")
            if raw_stream in ("none", "ndjson"):
                stream_mode = raw_stream

    try:
        cfg = validate_run_request(cfg_raw)
    except ValidationError as exc:
        envelope = _build_error_envelope(
            exit_code=_EXIT_CONFIG_VALIDATION,
            code="E_CONFIG_VALIDATION",
            message="invalid run config",
            started_at=started_at,
            finished_at=_now_utc_iso(),
            elapsed_ms=int((time.perf_counter() - run_start) * 1000),
            run_id=run_id,
            output_dir=output_dir,
            system_spec=system_spec,
            traj_spec=traj_spec,
            results=results,
            warnings=warnings,
            context={"config_path": config_path},
            details=_normalize_validation_errors(exc.errors()),
            debug_errors=debug_errors,
            exc=exc,
        )
        _emit_event(stream_mode, "run_failed", final_envelope=envelope)
        return _EXIT_CONFIG_VALIDATION, envelope, stream_mode

    run_id = cfg.run_id or run_id
    stream_mode = stream if stream is not None else cfg.stream
    system_spec = cfg.system_spec
    traj_spec = cfg.trajectory_spec
    output_dir = cfg.output_dir
    default_device = cfg.device
    default_chunk = cfg.chunk_frames
    total_analyses = len(cfg.analyses)

    _emit_event(
        stream_mode,
        "run_started",
        run_id=run_id,
        config_path=config_path,
        dry_run=dry_run,
        analysis_count=total_analyses,
        **_progress_snapshot(0, total_analyses, 0),
    )

    try:
        try:
            system = _load_system(system_spec)
        except Exception as exc:
            raise RunContractError(
                _EXIT_RUNTIME_EXEC,
                "E_SYSTEM_LOAD",
                str(exc),
                context={"stage": "system_load"},
            ) from exc
        if not dry_run:
            try:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                raise RunContractError(
                    _EXIT_RUNTIME_EXEC,
                    "E_OUTPUT_DIR",
                    str(exc),
                    context={"output_dir": output_dir},
                ) from exc

        used_names: Dict[str, int] = {}
        traj = None
        for index, analysis in enumerate(cfg.analyses):
            item = analysis.model_dump(mode="python")
            requested_name = item["name"]
            try:
                name = _resolve_plan_name(requested_name)
            except Exception as exc:
                raise RunContractError(
                    _EXIT_ANALYSIS_SPEC,
                    "E_ANALYSIS_UNKNOWN",
                    str(exc),
                    context={"analysis": requested_name, "index": index},
                    details=[
                        {
                            "type": "value_error",
                            "field": f"analyses.{index}.name",
                            "message": str(exc),
                        }
                    ],
                ) from exc

            out_path = item.get("out") or _default_out(name, output_dir, used_names)
            _emit_event(
                stream_mode,
                "analysis_started",
                index=index,
                analysis=name,
                out=out_path,
                **_progress_snapshot(len(results), total_analyses, int((time.perf_counter() - run_start) * 1000)),
            )
            entry: Dict[str, Any] = {
                "analysis": name,
                "out": out_path,
            }

            if dry_run:
                entry["status"] = "dry_run"
                results.append(entry)
                _emit_event(
                    stream_mode,
                    "analysis_completed",
                    index=index,
                    analysis=name,
                    status="dry_run",
                    out=out_path,
                    **_progress_snapshot(
                        len(results),
                        total_analyses,
                        int((time.perf_counter() - run_start) * 1000),
                    ),
                )
                continue

            if traj is None:
                try:
                    traj = _load_trajectory(traj_spec, system)
                except Exception as exc:
                    raise RunContractError(
                        _EXIT_RUNTIME_EXEC,
                        "E_TRAJECTORY_LOAD",
                        str(exc),
                        context={"analysis": name, "index": index},
                    ) from exc

            try:
                plan = PLAN_BUILDERS[name](system, item)
            except Exception as exc:
                raise RunContractError(
                    _EXIT_ANALYSIS_SPEC,
                    "E_ANALYSIS_SPEC",
                    str(exc),
                    context={"analysis": name, "index": index},
                    details=_analysis_error_details(index, name, str(exc)),
                ) from exc

            analysis_start = time.perf_counter()
            device = item.get("device", default_device)
            chunk = item.get("chunk_frames", default_chunk)
            try:
                output = plan.run(traj, system, chunk_frames=chunk, device=device)
                saved_path = _save_output(out_path, output)
            except Exception as exc:
                raise RunContractError(
                    _EXIT_RUNTIME_EXEC,
                    "E_RUNTIME_EXEC",
                    str(exc),
                    context={"analysis": name, "index": index},
                ) from exc

            entry.update(_summary_from_output(output, name, Path(saved_path)))
            entry["status"] = "ok"
            entry["device"] = device
            entry["chunk_frames"] = chunk
            entry["timing_ms"] = int((time.perf_counter() - analysis_start) * 1000)
            entry["artifact"] = _artifact_metadata(saved_path)
            entry["out"] = saved_path
            results.append(entry)
            _emit_event(
                stream_mode,
                "analysis_completed",
                index=index,
                analysis=name,
                status="ok",
                out=saved_path,
                timing_ms=entry["timing_ms"],
                **_progress_snapshot(
                    len(results),
                    total_analyses,
                    int((time.perf_counter() - run_start) * 1000),
                ),
            )
    except RunContractError as exc:
        if "analysis" in exc.context:
            _emit_event(
                stream_mode,
                "analysis_failed",
                index=exc.context.get("index"),
                analysis=exc.context.get("analysis"),
                error={
                    "code": exc.code,
                    "message": str(exc),
                    "context": exc.context,
                },
                **_progress_snapshot(
                    len(results),
                    total_analyses,
                    int((time.perf_counter() - run_start) * 1000),
                ),
            )
        envelope = _build_error_envelope(
            exit_code=exc.exit_code,
            code=exc.code,
            message=str(exc),
            started_at=started_at,
            finished_at=_now_utc_iso(),
            elapsed_ms=int((time.perf_counter() - run_start) * 1000),
            run_id=run_id,
            output_dir=output_dir,
            system_spec=system_spec,
            traj_spec=traj_spec,
            results=results,
            warnings=warnings,
            context=exc.context,
            details=exc.details,
            debug_errors=debug_errors,
            exc=exc,
        )
        _emit_event(stream_mode, "run_failed", final_envelope=envelope)
        return exc.exit_code, envelope, stream_mode
    except Exception as exc:
        envelope = _build_error_envelope(
            exit_code=_EXIT_INTERNAL,
            code="E_INTERNAL",
            message="internal run failure",
            started_at=started_at,
            finished_at=_now_utc_iso(),
            elapsed_ms=int((time.perf_counter() - run_start) * 1000),
            run_id=run_id,
            output_dir=output_dir,
            system_spec=system_spec,
            traj_spec=traj_spec,
            results=results,
            warnings=warnings,
            context={"config_path": config_path},
            details=str(exc),
            debug_errors=debug_errors,
            exc=exc,
        )
        _emit_event(stream_mode, "run_failed", final_envelope=envelope)
        return _EXIT_INTERNAL, envelope, stream_mode

    envelope = _build_success_envelope(
        status="dry_run" if dry_run else "ok",
        run_id=run_id,
        output_dir=output_dir,
        system_spec=system_spec,
        traj_spec=traj_spec,
        started_at=started_at,
        finished_at=_now_utc_iso(),
        elapsed_ms=int((time.perf_counter() - run_start) * 1000),
        results=results,
        warnings=warnings,
    )
    _emit_event(stream_mode, "run_completed", final_envelope=envelope)
    return _EXIT_OK, envelope, stream_mode


def _action_type(action: argparse.Action) -> str:
    if isinstance(
        action,
        (
            argparse._StoreTrueAction,
            argparse._StoreFalseAction,
            argparse.BooleanOptionalAction,
        ),
    ):
        return "bool"
    if action.type is int:
        return "int"
    if action.type is float:
        return "float"
    return "string"


def _plan_argument_metadata(command_name: str) -> list[Dict[str, Any]]:
    parser = argparse.ArgumentParser(add_help=False)
    add_shared_args(parser)
    if command_name in REGISTRY:
        REGISTRY[command_name](parser)
    metadata: list[Dict[str, Any]] = []
    for action in parser._actions:
        if not action.option_strings:
            continue
        if action.dest == "help":
            continue
        item: Dict[str, Any] = {
            "flags": action.option_strings,
            "dest": action.dest,
            "required": bool(getattr(action, "required", False)),
            "type": _action_type(action),
        }
        if action.choices is not None:
            item["choices"] = list(action.choices)
        if action.default is not argparse.SUPPRESS and action.default is not None:
            item["default"] = action.default
        if action.help:
            item["help"] = action.help
        metadata.append(item)
    return metadata


def list_water_models(fmt: str = "text") -> None:
    names = available_water_models()
    if fmt == "json":
        payload = {
            "water_models": [
                {"name": model, "pdb": water_pdb(model)}
                for model in names
            ]
        }
        print(json.dumps(payload, indent=2))
        return
    for model in names:
        print(model)


def run_atlas_fetch(args: argparse.Namespace) -> tuple[int, Dict[str, Any]]:
    try:
        payload = download_atlas_trajectory(
            dataset=args.dataset,
            kind=args.kind,
            pdb_chain=args.pdb_chain,
            out=args.out,
            base_url=args.base_url,
            timeout=float(args.timeout),
            retries=int(args.retries),
            retry_wait=float(args.retry_wait),
            resume=bool(args.resume),
        )
    except (ValueError, AtlasApiError) as exc:
        return _EXIT_RUNTIME_EXEC, {
            "status": "error",
            "exit_code": _EXIT_RUNTIME_EXEC,
            "error": {
                "code": "E_ATLAS_FETCH",
                "message": str(exc),
            },
        }
    except Exception as exc:
        return _EXIT_INTERNAL, {
            "status": "error",
            "exit_code": _EXIT_INTERNAL,
            "error": {
                "code": "E_INTERNAL",
                "message": str(exc),
            },
        }
    payload["exit_code"] = _EXIT_OK
    return _EXIT_OK, payload


def list_plans_with_details(fmt: str = "text", details: bool = False) -> None:
    names = sorted(CLI_TO_PLAN.keys())
    if fmt == "json":
        if not details:
            print(json.dumps({"plans": names}, indent=2))
            return
        payload = {
            "plans": [
                {
                    "name": name,
                    "plan": CLI_TO_PLAN[name],
                    "arguments": _plan_argument_metadata(name),
                }
                for name in names
            ]
        }
        print(json.dumps(payload, indent=2))
        return
    for name in names:
        print(name)


def build_plan_from_args(args: argparse.Namespace, system):
    plan_name = CLI_TO_PLAN[args.analysis]
    spec = SPEC_BUILDERS[args.analysis](args, system)
    return PLAN_BUILDERS[plan_name](system, spec)


def _single_specs_from_args(args: argparse.Namespace) -> tuple[Dict[str, Any], Dict[str, Any]]:
    system_spec: Dict[str, Any] = {"path": args.topology}
    if getattr(args, "topology_format", None):
        system_spec["format"] = args.topology_format

    traj_spec: Dict[str, Any] = {"path": args.traj}
    if getattr(args, "traj_format", None):
        traj_spec["format"] = args.traj_format
    if getattr(args, "traj_length_scale", None) is not None:
        traj_spec["length_scale"] = args.traj_length_scale
    return system_spec, traj_spec


def run_single_analysis(args: argparse.Namespace) -> tuple[int, Dict[str, Any]]:
    run_start = time.perf_counter()
    started_at = _now_utc_iso()
    results: list[Dict[str, Any]] = []
    warnings: list[str] = []
    run_id: Optional[str] = None
    output_dir: Optional[str] = None
    system_spec, traj_spec = _single_specs_from_args(args)
    plan_name = CLI_TO_PLAN[args.analysis]
    debug_errors = bool(getattr(args, "debug_errors", False))
    if not getattr(args, "print_summary", True):
        warnings.append(
            "--no-summary is deprecated and ignored; single-analysis commands emit JSON envelopes"
        )
    if getattr(args, "summary_format", "json") != "json":
        warnings.append(
            "--summary-format is deprecated and ignored; single-analysis commands emit JSON envelopes"
        )

    try:
        try:
            system = _load_system_from_args(args)
        except Exception as exc:
            raise RunContractError(
                _EXIT_RUNTIME_EXEC,
                "E_SYSTEM_LOAD",
                str(exc),
                context={"analysis": plan_name},
            ) from exc
        try:
            traj = _load_traj_from_args(args, system)
        except Exception as exc:
            raise RunContractError(
                _EXIT_RUNTIME_EXEC,
                "E_TRAJECTORY_LOAD",
                str(exc),
                context={"analysis": plan_name},
            ) from exc
        try:
            plan = build_plan_from_args(args, system)
        except Exception as exc:
            raise RunContractError(
                _EXIT_ANALYSIS_SPEC,
                "E_ANALYSIS_SPEC",
                str(exc),
                context={"analysis": plan_name, "index": 0},
                details=_analysis_error_details(0, plan_name, str(exc)),
            ) from exc

        analysis_start = time.perf_counter()
        try:
            output = plan.run(traj, system, chunk_frames=args.chunk_frames, device=args.device)
            out_path = Path(args.out or f"{args.analysis}.npz")
            saved_path = _save_output(str(out_path), output)
        except Exception as exc:
            raise RunContractError(
                _EXIT_RUNTIME_EXEC,
                "E_RUNTIME_EXEC",
                str(exc),
                context={"analysis": plan_name, "index": 0},
            ) from exc

        entry = _summary_from_output(output, plan_name, Path(saved_path))
        entry["status"] = "ok"
        entry["device"] = args.device
        entry["chunk_frames"] = args.chunk_frames
        entry["timing_ms"] = int((time.perf_counter() - analysis_start) * 1000)
        entry["artifact"] = _artifact_metadata(saved_path)
        entry["out"] = saved_path
        results.append(entry)
        output_dir = str(Path(saved_path).parent)
    except RunContractError as exc:
        envelope = _build_error_envelope(
            exit_code=exc.exit_code,
            code=exc.code,
            message=str(exc),
            started_at=started_at,
            finished_at=_now_utc_iso(),
            elapsed_ms=int((time.perf_counter() - run_start) * 1000),
            run_id=run_id,
            output_dir=output_dir,
            system_spec=system_spec,
            traj_spec=traj_spec,
            results=results,
            warnings=warnings,
            context=exc.context,
            details=exc.details,
            debug_errors=debug_errors,
            exc=exc,
        )
        return exc.exit_code, envelope
    except Exception as exc:
        envelope = _build_error_envelope(
            exit_code=_EXIT_INTERNAL,
            code="E_INTERNAL",
            message="internal single-analysis failure",
            started_at=started_at,
            finished_at=_now_utc_iso(),
            elapsed_ms=int((time.perf_counter() - run_start) * 1000),
            run_id=run_id,
            output_dir=output_dir,
            system_spec=system_spec,
            traj_spec=traj_spec,
            results=results,
            warnings=warnings,
            context={"analysis": plan_name},
            details=str(exc),
            debug_errors=debug_errors,
            exc=exc,
        )
        return _EXIT_INTERNAL, envelope

    envelope = _build_success_envelope(
        status="ok",
        run_id=run_id,
        output_dir=output_dir or ".",
        system_spec=system_spec,
        traj_spec=traj_spec,
        started_at=started_at,
        finished_at=_now_utc_iso(),
        elapsed_ms=int((time.perf_counter() - run_start) * 1000),
        results=results,
        warnings=warnings,
    )
    return _EXIT_OK, envelope


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="warp-md")
    parser.add_argument(
        "--version",
        action="version",
        version=f"warp-md {_resolve_cli_version()}",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run analyses from a JSON/YAML config")
    run.add_argument("config", help="path to config.json|yaml")
    run.add_argument("--dry-run", action="store_true", help="validate and show outputs")
    run.add_argument(
        "--stream",
        choices=["none", "ndjson"],
        default=None,
        help="output mode for agents (CLI flag overrides config)",
    )
    run.add_argument(
        "--debug-errors",
        action="store_true",
        help="include traceback in JSON error envelopes",
    )

    list_cmd = sub.add_parser("list-plans", help="list available analysis names")
    list_cmd.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="output format",
    )
    list_cmd.add_argument(
        "--json",
        action="store_true",
        help="alias for --format json",
    )
    list_cmd.add_argument(
        "--details",
        action="store_true",
        help="include argument metadata (json format)",
    )
    water_models_cmd = sub.add_parser(
        "water-models",
        help="list available bundled water templates",
    )
    water_models_cmd.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="output format",
    )
    water_models_cmd.add_argument(
        "--json",
        action="store_true",
        help="alias for --format json",
    )
    atlas_fetch = sub.add_parser(
        "atlas-fetch",
        help="download trajectory archive from the ATLAS API",
    )
    atlas_fetch.add_argument(
        "--dataset",
        choices=["ATLAS", "chameleon", "DPF"],
        default="ATLAS",
        help="ATLAS dataset family",
    )
    atlas_fetch.add_argument(
        "--kind",
        choices=["analysis", "protein", "total"],
        default="total",
        help="trajectory package type",
    )
    atlas_fetch.add_argument(
        "--pdb-chain",
        required=True,
        help="PDB chain identifier (example: 16pk_A)",
    )
    atlas_fetch.add_argument(
        "--out",
        default=None,
        help="output archive path (.zip)",
    )
    atlas_fetch.add_argument(
        "--base-url",
        default=DEFAULT_ATLAS_API_BASE_URL,
        help="ATLAS API base URL",
    )
    atlas_fetch.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds",
    )
    atlas_fetch.add_argument(
        "--retries",
        type=int,
        default=3,
        help="number of retry attempts for transient failures",
    )
    atlas_fetch.add_argument(
        "--retry-wait",
        type=float,
        default=2.0,
        help="seconds to wait between retries",
    )
    atlas_fetch.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume from partial .part file when possible",
    )
    sub.add_parser("example", help="print example config")
    schema = sub.add_parser("schema", help="print run-config JSON schema for agents")
    schema.add_argument(
        "--kind",
        choices=["request", "result", "event"],
        default="request",
        help="schema kind",
    )
    schema.add_argument(
        "--format",
        choices=["json", "yaml"],
        default="json",
        help="schema output format",
    )
    schema.add_argument(
        "--json",
        action="store_true",
        help="alias for --format json",
    )
    schema.add_argument(
        "--out",
        help="optional output file path for schema",
    )

    # MCP (Model Context Protocol) server for agent integration
    mcp_cmd = sub.add_parser(
        "mcp",
        help="start MCP server for AI agent integration (stdio transport)",
    )
    mcp_cmd.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="transport mode (default: stdio)",
    )

    for name, setup in REGISTRY.items():
        help_text = f"Run {name} analysis"
        analysis = sub.add_parser(name, help=help_text, description=help_text)
        add_shared_args(analysis)
        setup(analysis)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "run":
        exit_code, envelope, stream_mode = run_config(
            args.config,
            dry_run=args.dry_run,
            stream=args.stream,
            debug_errors=args.debug_errors,
        )
        if stream_mode != "ndjson":
            print(json.dumps(envelope, indent=2))
        return exit_code
    if args.cmd == "list-plans":
        fmt = "json" if args.json else args.format
        list_plans_with_details(fmt, details=args.details)
        return 0
    if args.cmd == "water-models":
        fmt = "json" if args.json else args.format
        list_water_models(fmt)
        return 0
    if args.cmd == "atlas-fetch":
        exit_code, payload = run_atlas_fetch(args)
        print(json.dumps(payload, indent=2))
        return exit_code
    if args.cmd == "example":
        example_config()
        return 0
    if args.cmd == "schema":
        fmt = "json" if args.json else args.format
        text = render_agent_schema(target=args.kind, fmt=fmt)
        if args.out:
            Path(args.out).write_text(text)
            print(args.out)
        else:
            print(text)
        return 0
    if args.cmd == "mcp":
        from .mcp_server import main as mcp_main
        mcp_main()
        return 0
    if args.cmd in REGISTRY:
        args.analysis = args.cmd
        exit_code, envelope = run_single_analysis(args)
        print(json.dumps(envelope, indent=2))
        return exit_code

    return 1
