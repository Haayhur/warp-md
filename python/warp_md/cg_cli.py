from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .cg_contract import (
    build_example_request,
    cg_capabilities,
    cg_build_capabilities,
    cg_backmap_capabilities,
    cg_forcefield_inspect,
    cg_forcefield_install,
    cg_simulate_capabilities,
    example_request,
    plan_cg_simulate_request,
    render_cg_build_schema,
    render_cg_backmap_schema,
    render_cg_simulate_schema,
    render_cg_schema,
    run_cg_build_request,
    run_cg_backmap_request,
    run_cg_request,
    cg_simulate_status,
    simulate_example_request,
    validate_build_request_payload,
    validate_backmap_request_payload,
    validate_request_payload,
    validate_simulate_request_payload,
)


def _load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "YAML config requires PyYAML. Install with `pip install warp-md[cli]`."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("request must decode to an object")
    return data


def _dump_payload(payload: Dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(payload, indent=2)
    if fmt == "yaml":
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("YAML output requires PyYAML installed") from exc
        return yaml.safe_dump(payload, sort_keys=False)
    raise ValueError("format must be json or yaml")


def _load_request(args: argparse.Namespace) -> Dict[str, Any]:
    if getattr(args, "stdin", False):
        data = json.loads(sys.stdin.read())
        if not isinstance(data, dict):
            raise ValueError("request must decode to an object")
        return data
    if not args.request:
        raise ValueError("request path is required unless --stdin is used")
    return _load_config(Path(args.request))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="warp-cg",
        description="Agent-safe Martini coarse-graining mapper",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_cmd = sub.add_parser("run", help="run coarse-graining request")
    run_cmd.add_argument("request", nargs="?", help="path to request.json")
    run_cmd.add_argument("--stdin", action="store_true", help="read request from stdin")
    run_cmd.add_argument(
        "--stream",
        choices=["none", "ndjson"],
        default="none",
        help="stream progress events to stderr",
    )

    validate_cmd = sub.add_parser("validate", help="validate request")
    validate_cmd.add_argument("request", nargs="?", help="path to request.json")
    validate_cmd.add_argument("--stdin", action="store_true", help="read request from stdin")
    validate_cmd.add_argument("--format", choices=["json", "yaml"], default="json")

    schema_cmd = sub.add_parser("schema", help="print request/result/event schema")
    schema_cmd.add_argument("--kind", choices=["request", "result", "event"], default="request")
    schema_cmd.add_argument("--format", choices=["json", "yaml"], default="json")
    schema_cmd.add_argument("--out", help="optional output path")

    example_cmd = sub.add_parser("example", help="print example request")
    example_cmd.add_argument("--format", choices=["json", "yaml"], default="json")

    caps_cmd = sub.add_parser("capabilities", help="print capabilities")
    caps_cmd.add_argument("--format", choices=["json", "yaml"], default="json")

    build_cmd = sub.add_parser("build", help="build coarse-grained biomolecular systems")
    build_sub = build_cmd.add_subparsers(dest="build_cmd", required=True)

    build_run_cmd = build_sub.add_parser("run", help="run build request")
    build_run_cmd.add_argument("request", nargs="?", help="path to request.json")
    build_run_cmd.add_argument("--stdin", action="store_true", help="read request from stdin")
    build_run_cmd.add_argument(
        "--stream",
        choices=["none", "ndjson"],
        default="none",
        help="stream progress events to stderr",
    )

    build_validate_cmd = build_sub.add_parser("validate", help="validate build request")
    build_validate_cmd.add_argument("request", nargs="?", help="path to request.json")
    build_validate_cmd.add_argument("--stdin", action="store_true", help="read request from stdin")
    build_validate_cmd.add_argument("--format", choices=["json", "yaml"], default="json")

    build_schema_cmd = build_sub.add_parser("schema", help="print build request/result/event schema")
    build_schema_cmd.add_argument("--kind", choices=["request", "result", "event"], default="request")
    build_schema_cmd.add_argument("--format", choices=["json", "yaml"], default="json")
    build_schema_cmd.add_argument("--out", help="optional output path")

    build_example_cmd = build_sub.add_parser("example", help="print example build request")
    build_example_cmd.add_argument("--format", choices=["json", "yaml"], default="json")

    build_caps_cmd = build_sub.add_parser("capabilities", help="print build capabilities")
    build_caps_cmd.add_argument("--format", choices=["json", "yaml"], default="json")

    backmap_cmd = sub.add_parser("backmap", help="reconstruct AA coordinates from CG frames")
    backmap_sub = backmap_cmd.add_subparsers(dest="backmap_cmd", required=True)
    backmap_run_cmd = backmap_sub.add_parser("run", help="run backmap request")
    backmap_run_cmd.add_argument("request", nargs="?", help="path to request.json")
    backmap_run_cmd.add_argument("--stdin", action="store_true", help="read request from stdin")
    backmap_validate_cmd = backmap_sub.add_parser("validate", help="validate backmap request")
    backmap_validate_cmd.add_argument("request", nargs="?", help="path to request.json")
    backmap_validate_cmd.add_argument("--stdin", action="store_true", help="read request from stdin")
    backmap_validate_cmd.add_argument("--format", choices=["json", "yaml"], default="json")
    backmap_schema_cmd = backmap_sub.add_parser("schema", help="print backmap schema")
    backmap_schema_cmd.add_argument("--kind", choices=["request", "result"], default="request")
    backmap_schema_cmd.add_argument("--format", choices=["json", "yaml"], default="json")
    backmap_schema_cmd.add_argument("--out")
    backmap_caps_cmd = backmap_sub.add_parser("capabilities", help="print backmap capabilities")
    backmap_caps_cmd.add_argument("--format", choices=["json", "yaml"], default="json")

    simulate_cmd = sub.add_parser("simulate", help="plan and inspect CG simulation handoffs")
    simulate_sub = simulate_cmd.add_subparsers(dest="simulate_cmd", required=True)

    simulate_schema_cmd = simulate_sub.add_parser("schema", help="print simulate schemas")
    simulate_schema_cmd.add_argument(
        "--kind",
        choices=["request", "plan", "result", "status", "manifest"],
        default="request",
    )
    simulate_schema_cmd.add_argument("--format", choices=["json", "yaml"], default="json")
    simulate_schema_cmd.add_argument("--out", help="optional output path")

    simulate_example_cmd = simulate_sub.add_parser("example", help="print simulate request example")
    simulate_example_cmd.add_argument("--engine", choices=["gromacs", "openmm"], default="gromacs")
    simulate_example_cmd.add_argument("--format", choices=["json", "yaml"], default="json")

    simulate_validate_cmd = simulate_sub.add_parser("validate", help="validate simulate request")
    simulate_validate_cmd.add_argument("request", nargs="?", help="path to request.json")
    simulate_validate_cmd.add_argument("--stdin", action="store_true", help="read request from stdin")
    simulate_validate_cmd.add_argument("--format", choices=["json", "yaml"], default="json")

    simulate_plan_cmd = simulate_sub.add_parser("plan", help="emit simulation command plan")
    simulate_plan_cmd.add_argument("request", nargs="?", help="path to request.json")
    simulate_plan_cmd.add_argument("--stdin", action="store_true", help="read request from stdin")
    simulate_plan_cmd.add_argument("--engine", choices=["gromacs", "openmm"])
    simulate_plan_cmd.add_argument("--format", choices=["json", "yaml"], default="json")

    simulate_status_cmd = simulate_sub.add_parser("status", help="inspect simulation run directory")
    simulate_status_cmd.add_argument("run_dir", help="run directory")
    simulate_status_cmd.add_argument("--format", choices=["json", "yaml"], default="json")

    simulate_caps_cmd = simulate_sub.add_parser("capabilities", help="print simulate capabilities")
    simulate_caps_cmd.add_argument("--format", choices=["json", "yaml"], default="json")

    forcefield_cmd = sub.add_parser("forcefield", help="inspect or install bundled forcefields")
    forcefield_sub = forcefield_cmd.add_subparsers(dest="forcefield_cmd", required=True)

    forcefield_inspect_cmd = forcefield_sub.add_parser("inspect", help="print bundled forcefield manifest")
    forcefield_inspect_cmd.add_argument("--kind", default="martini3")
    forcefield_inspect_cmd.add_argument("--format", choices=["json", "yaml"], default="json")

    forcefield_install_cmd = forcefield_sub.add_parser(
        "install", help="copy bundled forcefield into a project directory"
    )
    forcefield_install_cmd.add_argument("--kind", default="martini3")
    forcefield_install_cmd.add_argument("--dest", required=True)
    forcefield_install_cmd.add_argument("--overwrite", action="store_true")
    forcefield_install_cmd.add_argument("--format", choices=["json", "yaml"], default="json")

    runner_cmd = sub.add_parser("runner", help="run managed CG simulation helpers")
    runner_sub = runner_cmd.add_subparsers(dest="runner_cmd", required=True)

    martini_cmd = runner_sub.add_parser(
        "martini-openmm",
        help="run Martini/OpenMM minimization, equilibration, and optional production",
    )
    martini_cmd.add_argument("runner_args", nargs=argparse.REMAINDER)

    return parser


def run_cli(argv: Optional[list[str]] = None) -> int:
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if (
        len(effective_argv) >= 2
        and effective_argv[0] == "runner"
        and effective_argv[1] == "martini-openmm"
    ):
        from .cg_martini_openmm import run_cli as run_martini_openmm_cli

        runner_args = effective_argv[2:]
        if runner_args and runner_args[0] == "--":
            runner_args = runner_args[1:]
        return run_martini_openmm_cli(runner_args)

    args = build_parser().parse_args(effective_argv)

    if args.cmd == "run":
        exit_code, result = run_cg_request(_load_request(args), stream=args.stream)
        print(json.dumps(result, indent=2))
        return exit_code
    if args.cmd == "validate":
        result = validate_request_payload(_load_request(args))
        print(_dump_payload(result, args.format))
        return 0 if result.get("valid", False) else 2
    if args.cmd == "schema":
        rendered = render_cg_schema(args.kind, args.format)
        if args.out:
            Path(args.out).write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
        return 0
    if args.cmd == "example":
        print(_dump_payload(example_request(), args.format))
        return 0
    if args.cmd == "capabilities":
        print(_dump_payload(cg_capabilities(), args.format))
        return 0
    if args.cmd == "build":
        if args.build_cmd == "run":
            exit_code, result = run_cg_build_request(_load_request(args), stream=args.stream)
            print(json.dumps(result, indent=2))
            return exit_code
        if args.build_cmd == "validate":
            result = validate_build_request_payload(_load_request(args))
            print(_dump_payload(result, args.format))
            return 0 if result.get("valid", False) else 2
        if args.build_cmd == "schema":
            rendered = render_cg_build_schema(args.kind, args.format)
            if args.out:
                Path(args.out).write_text(rendered + "\n", encoding="utf-8")
            else:
                print(rendered)
            return 0
        if args.build_cmd == "example":
            print(_dump_payload(build_example_request(), args.format))
            return 0
        if args.build_cmd == "capabilities":
            print(_dump_payload(cg_build_capabilities(), args.format))
            return 0
    if args.cmd == "backmap":
        if args.backmap_cmd == "run":
            exit_code, result = run_cg_backmap_request(_load_request(args))
            print(json.dumps(result, indent=2))
            return exit_code
        if args.backmap_cmd == "validate":
            result = validate_backmap_request_payload(_load_request(args))
            print(_dump_payload(result, args.format))
            return 0 if result.get("valid", False) else 2
        if args.backmap_cmd == "schema":
            rendered = render_cg_backmap_schema(args.kind, args.format)
            if args.out:
                Path(args.out).write_text(rendered + "\n", encoding="utf-8")
            else:
                print(rendered)
            return 0
        if args.backmap_cmd == "capabilities":
            print(_dump_payload(cg_backmap_capabilities(), args.format))
            return 0
    if args.cmd == "simulate":
        if args.simulate_cmd == "schema":
            rendered = render_cg_simulate_schema(args.kind, args.format)
            if args.out:
                Path(args.out).write_text(rendered + "\n", encoding="utf-8")
            else:
                print(rendered)
            return 0
        if args.simulate_cmd == "example":
            print(_dump_payload(simulate_example_request(args.engine), args.format))
            return 0
        if args.simulate_cmd == "validate":
            result = validate_simulate_request_payload(_load_request(args))
            print(_dump_payload(result, args.format))
            return 0 if result.get("valid", False) else 2
        if args.simulate_cmd == "plan":
            exit_code, result = plan_cg_simulate_request(_load_request(args), engine=args.engine)
            print(_dump_payload(result, args.format))
            return exit_code
        if args.simulate_cmd == "status":
            exit_code, result = cg_simulate_status(args.run_dir)
            print(_dump_payload(result, args.format))
            return exit_code
        if args.simulate_cmd == "capabilities":
            print(_dump_payload(cg_simulate_capabilities(), args.format))
            return 0
    if args.cmd == "forcefield":
        if args.forcefield_cmd == "inspect":
            print(_dump_payload(cg_forcefield_inspect(args.kind), args.format))
            return 0
        if args.forcefield_cmd == "install":
            print(
                _dump_payload(
                    cg_forcefield_install(
                        args.dest,
                        kind=args.kind,
                        overwrite=args.overwrite,
                    ),
                    args.format,
                )
            )
            return 0
    if args.cmd == "runner":
        if args.runner_cmd == "martini-openmm":
            from .cg_martini_openmm import run_cli as run_martini_openmm_cli

            runner_args = args.runner_args
            if runner_args and runner_args[0] == "--":
                runner_args = runner_args[1:]
            return run_martini_openmm_cli(runner_args)
    raise RuntimeError(f"unknown command: {args.cmd}")


def main(argv: Optional[list[str]] = None) -> int:
    try:
        return run_cli(argv)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
