from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from .pack import PackConfig, export, parse_inp, run


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
        raise ValueError("pack config must decode to a JSON/YAML object")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="warp-pack", description="CPU packing utility")
    parser.add_argument("-c", "--config", required=True, help="path to config (.json|.yaml|.inp)")
    parser.add_argument("-o", "--output", help="optional output path override")
    parser.add_argument("-f", "--format", help="optional output format override (default: pdb)")
    return parser


def _build_output_override(cfg: Dict[str, Any], output: str, fmt: str | None) -> None:
    previous = cfg.get("output")
    scale = None
    if isinstance(previous, dict):
        scale = previous.get("scale")
    merged: Dict[str, Any] = {"path": output, "format": fmt or "pdb"}
    if scale is not None:
        merged["scale"] = scale
    cfg["output"] = merged


def run_cli(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        export(
            result,
            cfg.output.format,
            cfg.output.path,
            cfg.output.scale,
            add_box_sides=add_box_sides,
            box_sides_fix=box_sides_fix,
            write_conect=write_conect,
            hexadecimal_indices=cfg.hexadecimal_indices,
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


def main(argv: list[str] | None = None) -> int:
    try:
        return run_cli(argv)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
