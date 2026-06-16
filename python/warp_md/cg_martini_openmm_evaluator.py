from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .cg_martini_openmm import run_martini_openmm

OBJECTIVE_RESULT_SCHEMA = "warp-cg.objective-result.v1"


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must decode to a JSON object")
    return data


def _write_result(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _candidate_parameters(request: Dict[str, Any]) -> Dict[str, float]:
    candidate = request.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("candidate request is missing candidate object")
    values: Dict[str, float] = {}
    for item in candidate.get("named_parameters") or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        value = item.get("value")
        if isinstance(name, str) and isinstance(value, (int, float)):
            values[name] = float(value)
    return values


def _resolve_path(spec: Dict[str, Any], spec_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [
        spec_path.parent / path,
        Path(str(spec.get("base_dir", "."))) / path,
        Path(str(spec.get("out_dir", "."))) / path,
        path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


def _copy_file(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest


def _materialize_inputs(spec: Dict[str, Any], spec_path: Path, eval_dir: Path) -> tuple[Path, Path]:
    template_dir = spec.get("template_dir")
    if isinstance(template_dir, str) and template_dir.strip():
        source_template = _resolve_path(spec, spec_path, template_dir)
        if not source_template.exists():
            raise FileNotFoundError(f"runner template_dir not found: {source_template}")
        shutil.copytree(source_template, eval_dir, dirs_exist_ok=True)

    gro_value = str(spec.get("gro") or "").strip()
    top_value = str(spec.get("top") or "").strip()
    if not gro_value or not top_value:
        raise ValueError("runner spec requires gro and top")

    gro_path = eval_dir / gro_value
    if not gro_path.exists():
        gro_path = _copy_file(_resolve_path(spec, spec_path, gro_value), eval_dir / Path(gro_value).name)

    top_path = eval_dir / top_value
    if not top_path.exists():
        top_path = _copy_file(_resolve_path(spec, spec_path, top_value), eval_dir / Path(top_value).name)

    return gro_path, top_path


def _materialize_forcefield(spec: Dict[str, Any], eval_dir: Path) -> None:
    forcefield_dir = spec.get("forcefield_directory")
    if not isinstance(forcefield_dir, str) or not forcefield_dir.strip():
        return
    source = Path(forcefield_dir)
    if not source.exists():
        return
    target = eval_dir / "forcefields" / source.name
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)


def _format_parameter(value: float, fmt: Optional[str]) -> str:
    if not fmt:
        return f"{value:.10g}"
    if "{" in fmt:
        return fmt.format(value)
    return format(value, fmt)


def _apply_replacement(path: Path, placeholder: str, value: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if placeholder not in text:
        return False
    path.write_text(text.replace(placeholder, value), encoding="utf-8")
    return True


def _iter_template_text_files(eval_dir: Path) -> Iterable[Path]:
    suffixes = {".top", ".itp", ".gro", ".mdp", ".txt", ".inc"}
    for path in eval_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def _apply_parameter_replacements(
    spec: Dict[str, Any],
    spec_path: Path,
    eval_dir: Path,
    parameters: Dict[str, float],
) -> Dict[str, Any]:
    replacements = spec.get("replacements") or []
    applied: Dict[str, Any] = {"count": 0, "missing": [], "parameters": []}
    applied_parameters = set()
    if replacements:
        for replacement in replacements:
            if not isinstance(replacement, dict):
                continue
            name = str(replacement.get("parameter") or "")
            rel_path = str(replacement.get("path") or "")
            if not name or not rel_path:
                continue
            if name not in parameters:
                applied["missing"].append({"parameter": name, "reason": "candidate parameter not found"})
                continue
            target = eval_dir / rel_path
            if not target.exists():
                source = _resolve_path(spec, spec_path, rel_path)
                if source.exists():
                    target = _copy_file(source, target)
            if not target.exists():
                applied["missing"].append({"parameter": name, "path": rel_path, "reason": "file not found"})
                continue
            placeholder = str(replacement.get("placeholder") or f"{{{{{name}}}}}")
            rendered = _format_parameter(parameters[name], replacement.get("format"))
            if _apply_replacement(target, placeholder, rendered):
                applied["count"] += 1
                applied_parameters.add(name)
            else:
                applied["missing"].append({"parameter": name, "path": rel_path, "reason": "placeholder not found"})
        applied["parameters"] = sorted(applied_parameters)
        return applied

    for path in _iter_template_text_files(eval_dir):
        for name, value in parameters.items():
            if _apply_replacement(path, f"{{{{{name}}}}}", _format_parameter(value, None)):
                applied["count"] += 1
                applied_parameters.add(name)
    applied["parameters"] = sorted(applied_parameters)
    return applied


def _validate_required_replacements(spec: Dict[str, Any], parameters: Dict[str, float], applied: Dict[str, Any]) -> None:
    if not spec.get("require_parameter_replacements"):
        return
    missing = sorted(set(parameters) - set(applied.get("parameters") or []))
    if missing:
        raise RuntimeError(
            "simulation_fit requires every candidate parameter to be applied to the OpenMM template; "
            f"missing replacements for: {', '.join(missing)}"
        )


def _relative_to_eval(path: Path, eval_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(eval_dir.resolve()))
    except ValueError:
        return str(path)


def evaluate(spec_path: Path, candidate_path: Path, result_path: Path) -> int:
    spec = _load_json(spec_path)
    request = _load_json(candidate_path)
    eval_dir = candidate_path.parent
    try:
        parameters = _candidate_parameters(request)
        gro, top = _materialize_inputs(spec, spec_path, eval_dir)
        _materialize_forcefield(spec, eval_dir)
        applied = _apply_parameter_replacements(spec, spec_path, eval_dir, parameters)
        _validate_required_replacements(spec, parameters, applied)
        (eval_dir / "applied_parameters.json").write_text(
            json.dumps({"parameters": parameters, "replacements": applied}, indent=2) + "\n",
            encoding="utf-8",
        )
        run_dir = eval_dir / "run"
        run_result = run_martini_openmm(gro, top, run_dir, spec.get("protocol") or {})
        metrics = {
            "runner.parameter_count": float(len(parameters)),
            "runner.replacements": float(applied["count"]),
            "runner.simulation_fit": 1.0 if spec.get("simulation_fit") else 0.0,
            "runner.dry_run": 1.0 if run_result.get("dry_run") else 0.0,
        }
        payload: Dict[str, Any] = {
            "schema_version": OBJECTIVE_RESULT_SCHEMA,
            "status": "completed",
            "objective": 0.0,
            "metrics": metrics,
        }
        trajectory = run_result.get("trajectory")
        if trajectory:
            payload["candidate_trajectory"] = {
                "path": _relative_to_eval(Path(str(trajectory)), eval_dir)
            }
        _write_result(result_path, payload)
        return 0
    except Exception as exc:
        _write_result(
            result_path,
            {
                "schema_version": OBJECTIVE_RESULT_SCHEMA,
                "status": "failed_simulation",
                "reason": str(exc),
                "metrics": {},
            },
        )
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one warp-cg candidate with Martini/OpenMM.")
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--candidate-json", type=Path)
    parser.add_argument("--result-json", type=Path)
    return parser


def run_cli(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    candidate = args.candidate_json or Path(str(Path.cwd() / "candidate.json"))
    result = args.result_json or Path(str(Path.cwd() / "result.json"))
    env_candidate = Path(str(os.environ.get("WARP_CG_CANDIDATE_JSON", candidate)))
    env_result = Path(str(os.environ.get("WARP_CG_RESULT_JSON", result)))
    return evaluate(args.spec, env_candidate, env_result)


def main(argv: Optional[Iterable[str]] = None) -> int:
    try:
        return run_cli(argv)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
