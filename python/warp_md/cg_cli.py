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
    example_request,
    render_cg_build_schema,
    render_cg_schema,
    run_cg_build_request,
    run_cg_request,
    validate_build_request_payload,
    validate_request_payload,
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

    return parser


def run_cli(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

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
    raise RuntimeError(f"unknown command: {args.cmd}")


def main(argv: Optional[list[str]] = None) -> int:
    try:
        return run_cli(argv)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
