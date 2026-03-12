from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from . import build as build_contract


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


def _load_request(path: str) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("request must decode to an object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="warp-build", description="Agent-safe polymer builder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    schema_cmd = sub.add_parser("schema", help="print request/result/event/source schemas")
    schema_cmd.add_argument(
        "--kind",
        choices=[
            "request",
            "result",
            "event",
            "source_bundle",
            "build_manifest",
            "charge_manifest",
            "topology_graph",
        ],
        default="request",
        help="schema kind",
    )
    schema_cmd.add_argument("--format", choices=["json", "yaml"], default="json", help="output format")
    schema_cmd.add_argument("--json", action="store_true", help="alias for --format json")
    schema_cmd.add_argument("--out", help="optional output file path")

    example_cmd = sub.add_parser("example", help="print example build request")
    example_cmd.add_argument("--mode", default="random_walk", help="example mode")
    example_cmd.add_argument("--format", choices=["json", "yaml"], default="json", help="output format")
    example_cmd.add_argument("--json", action="store_true", help="alias for --format json")

    example_bundle_cmd = sub.add_parser("example-bundle", help="print example source bundle")
    example_bundle_cmd.add_argument("--format", choices=["json", "yaml"], default="json", help="output format")
    example_bundle_cmd.add_argument("--json", action="store_true", help="alias for --format json")

    caps_cmd = sub.add_parser("capabilities", help="print build capabilities")
    caps_cmd.add_argument("--format", choices=["json", "yaml"], default="json", help="output format")
    caps_cmd.add_argument("--json", action="store_true", help="alias for --format json")

    inspect_cmd = sub.add_parser("inspect-source", help="inspect a source bundle")
    inspect_cmd.add_argument("source", help="path to bundle.json")
    inspect_cmd.add_argument("--format", choices=["json", "yaml"], default="json", help="output format")
    inspect_cmd.add_argument("--json", action="store_true", help="alias for --format json")

    validate_cmd = sub.add_parser("validate", help="validate high-level build request")
    validate_cmd.add_argument("request", help="path to request.json")
    validate_cmd.add_argument("--format", choices=["json", "yaml"], default="json", help="output format")
    validate_cmd.add_argument("--json", action="store_true", help="alias for --format json")

    run_cmd = sub.add_parser("run", help="run high-level build request")
    run_cmd.add_argument("request", help="path to request.json")
    run_cmd.add_argument("--stream", action="store_true", help="stream progress events to stderr")

    return parser


def run_cli(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "schema":
        fmt = "json" if args.json else args.format
        text = _dump_payload(build_contract.schema_json(args.kind), fmt)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
            print(args.out)
        else:
            print(text)
        return 0

    if args.cmd == "example":
        fmt = "json" if args.json else args.format
        print(_dump_payload(build_contract.example_request(args.mode), fmt))
        return 0

    if args.cmd == "example-bundle":
        fmt = "json" if args.json else args.format
        print(_dump_payload(build_contract.example_bundle(), fmt))
        return 0

    if args.cmd == "capabilities":
        fmt = "json" if args.json else args.format
        print(_dump_payload(build_contract.capabilities(), fmt))
        return 0

    if args.cmd == "inspect-source":
        fmt = "json" if args.json else args.format
        payload = build_contract.inspect_source(args.source)
        print(_dump_payload(payload, fmt))
        return 0 if payload.get("valid", True) else 2

    if args.cmd == "validate":
        fmt = "json" if args.json else args.format
        payload = build_contract.validate(_load_request(args.request))
        print(_dump_payload(payload, fmt))
        return 0 if payload.get("valid") else 2

    if args.cmd == "run":
        exit_code, payload = build_contract.run(_load_request(args.request), stream=args.stream)
        if not args.stream:
            print(json.dumps(payload, indent=2))
        return exit_code

    return 1


def main(argv: Optional[list[str]] = None) -> int:
    try:
        return run_cli(argv)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
