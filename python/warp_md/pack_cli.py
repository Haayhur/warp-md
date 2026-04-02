from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .pack import PackConfig, export, parse_inp, run
from .pack_contract import example_request, pack_capabilities, render_pack_schema, run_build_request, validate_request_payload


def _infer_output_format(path: str) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix or "pdb"


def _load_config(path: Path) -> Dict[str, Any]:
    ext = path.suffix.lower()
    if ext == ".inp":
        return parse_inp(str(path))

    text = path.read_text(encoding="utf-8")
    if ext in {".yaml", ".yml"}:
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
        raise ValueError("config must decode to a JSON/YAML object")
    return data


def build_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="warp-pack", description="CPU packing utility")
    parser.add_argument("-c", "--config", required=True, help="path to config (.json|.yaml|.inp)")
    parser.add_argument("-o", "--output", help="optional output path override")
    parser.add_argument("-f", "--format", help="optional output format override (default: pdb)")
    return parser


def build_contract_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="warp-pack", description="Agent-safe production system builder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_cmd = sub.add_parser("run", help="run high-level build request")
    run_cmd.add_argument("request", nargs="?", help="path to request.json")
    run_cmd.add_argument("--stdin", action="store_true", help="read request from stdin")
    run_cmd.add_argument("--stream", choices=["none", "ndjson"], default="none", help="stream progress events to stderr")

    validate_cmd = sub.add_parser("validate", help="validate high-level build request")
    validate_cmd.add_argument("request", nargs="?", help="path to request.json")
    validate_cmd.add_argument("--stdin", action="store_true", help="read request from stdin")
    validate_cmd.add_argument("--format", choices=["json", "yaml"], default="json", help="output format")
    validate_cmd.add_argument("--json", action="store_true", help="alias for --format json")

    schema_cmd = sub.add_parser("schema", help="print request/result/event schema")
    schema_cmd.add_argument("--kind", choices=["request", "result", "event"], default="request", help="schema kind")
    schema_cmd.add_argument("--format", choices=["json", "yaml"], default="json", help="output format")
    schema_cmd.add_argument("--json", action="store_true", help="alias for --format json")
    schema_cmd.add_argument("--out", help="optional output file path")

    example_cmd = sub.add_parser("example", help="print example request")
    example_cmd.add_argument(
        "--mode",
        choices=[
            "solute_solvate",
            "polymer_build_handoff",
            "components_amorphous_bulk",
            "components_backbone_aligned_bulk",
        ],
        default="solute_solvate",
    )
    example_cmd.add_argument("--format", choices=["json", "yaml"], default="json", help="output format")
    example_cmd.add_argument("--json", action="store_true", help="alias for --format json")

    caps_cmd = sub.add_parser("capabilities", help="print pack capabilities")
    caps_cmd.add_argument("--format", choices=["json", "yaml"], default="json", help="output format")
    caps_cmd.add_argument("--json", action="store_true", help="alias for --format json")

    return parser


def _build_output_override(cfg: Dict[str, Any], output: str, fmt: str | None) -> None:
    previous = cfg.get("output")
    scale = None
    if isinstance(previous, dict):
        scale = previous.get("scale")
    merged: Dict[str, Any] = {"path": output, "format": fmt or _infer_output_format(output)}
    if scale is not None:
        merged["scale"] = scale
    cfg["output"] = merged


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
        raw = sys.stdin.read()
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("request must decode to an object")
        return data
    request_path = getattr(args, "request", None)
    if not request_path:
        raise ValueError("request path is required unless --stdin is used")
    return _load_config(Path(request_path))


def run_legacy_cli(argv: list[str] | None = None) -> int:
    args = build_legacy_parser().parse_args(argv)
    cfg_dict = _load_config(Path(args.config))
    if args.output:
        _build_output_override(cfg_dict, args.output, args.format)

    cfg = PackConfig.from_dict(cfg_dict)
    cfg.validate()
    result = run(cfg)

    if cfg.output is not None:
        add_box_sides = cfg.add_box_sides or cfg.pbc
        box_sides_fix = cfg.add_box_sides_fix if cfg.add_box_sides else 0.0
        write_conect = not cfg.ignore_conect
        written = export(
            result,
            cfg.output.format,
            cfg.output.path,
            cfg.output.scale,
            add_box_sides=add_box_sides,
            box_sides_fix=box_sides_fix,
            write_conect=write_conect,
            hexadecimal_indices=cfg.hexadecimal_indices,
        )
        if isinstance(written, dict) and written.get("fallback_applied"):
            print(
                f"requested output required mmcif; wrote '{written['path']}' instead",
                file=sys.stderr,
            )

    if cfg.write_crd:
        export(
            result,
            "crd",
            cfg.write_crd,
            cfg.output.scale if cfg.output is not None else 1.0,
            box_sides_fix=cfg.add_box_sides_fix or 0.0,
        )
    return 0


def run_contract_cli(argv: list[str] | None = None) -> int:
    args = build_contract_parser().parse_args(argv)

    if args.cmd == "run":
        try:
            payload = _load_request(args)
        except Exception as exc:
            envelope = {
                "schema_version": "warp-pack.agent.v1",
                "status": "error",
                "run_id": None,
                "exit_code": 2,
                "error": {
                    "code": "E_CONFIG_LOAD",
                    "path": getattr(args, "request", None),
                    "message": str(exc),
                },
                "errors": [
                    {
                        "code": "E_CONFIG_LOAD",
                        "path": getattr(args, "request", None),
                        "message": str(exc),
                    }
                ],
            }
            if args.stream == "ndjson":
                sys.stderr.write(
                    json.dumps(
                        {
                            "event": "run_failed",
                            "schema_version": "warp-pack.agent.v1",
                            "run_id": "warp-pack-run",
                            "elapsed_ms": 0,
                            "final_envelope": envelope,
                        }
                    )
                    + "\n"
                )
                sys.stderr.flush()
            else:
                print(json.dumps(envelope, indent=2))
            return 2
        exit_code, envelope = run_build_request(payload, stream=args.stream)
        if args.stream != "ndjson":
            print(json.dumps(envelope, indent=2))
        return exit_code

    if args.cmd == "validate":
        try:
            payload = _load_request(args)
        except Exception as exc:
            fmt = "json" if args.json else args.format
            result = {
                "status": "error",
                "valid": False,
                "errors": [
                    {
                        "code": "E_CONFIG_LOAD",
                        "path": getattr(args, "request", None),
                        "message": str(exc),
                    }
                ],
            }
            print(_dump_payload(result, fmt))
            return 2
        fmt = "json" if args.json else args.format
        result = validate_request_payload(payload)
        print(_dump_payload(result, fmt))
        return 0 if result.get("valid") else 2

    if args.cmd == "schema":
        fmt = "json" if args.json else args.format
        text = render_pack_schema(target=args.kind, fmt=fmt)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(args.out)
        else:
            print(text)
        return 0

    if args.cmd == "example":
        fmt = "json" if args.json else args.format
        print(_dump_payload(example_request(args.mode), fmt))
        return 0

    if args.cmd == "capabilities":
        fmt = "json" if args.json else args.format
        print(_dump_payload(pack_capabilities(), fmt))
        return 0

    return 1


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    contract_commands = {"run", "validate", "schema", "example", "capabilities"}
    try:
        if argv and argv[0] in contract_commands:
            return run_contract_cli(argv)
        return run_legacy_cli(argv)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
