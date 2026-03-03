from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List, Optional


def _parse_angles_csv(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None:
        return None
    values = [s.strip() for s in raw.split(",") if s.strip()]
    if not values:
        return []
    try:
        return [float(v) for v in values]
    except ValueError as exc:
        raise ValueError(f"invalid angle list '{raw}'") from exc


def _parse_mutations(raw: str) -> List[str]:
    values = [s.strip() for s in raw.split(",") if s.strip()]
    if not values:
        raise ValueError("mutations cannot be empty")
    return values


def _load_native():
    try:
        from .traj_py import pep_build, pep_mutate  # type: ignore

        return pep_build, pep_mutate
    except Exception as exc:  # pragma: no cover - import fallback
        raise RuntimeError(
            "warp-pep native bindings unavailable. Reinstall warp-md with compiled extensions."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="warp-pep",
        description="Peptide builder and mutation CLI (pip-installed wrapper)",
    )
    parser.add_argument(
        "--stream",
        dest="_global_stream",
        action="store_true",
        help="emit lightweight NDJSON progress events to stderr",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="Build a peptide")
    b.add_argument("-s", "--sequence")
    b.add_argument("-t", "--three-letter")
    b.add_argument("-j", "--json")
    b.add_argument("-o", "--output")
    b.add_argument("-f", "--format")
    b.add_argument("--oxt", action="store_true")
    b.add_argument("--preset")
    b.add_argument("--phi")
    b.add_argument("--psi")
    b.add_argument("--omega")
    b.add_argument("--detect-ss", action="store_true")
    b.add_argument(
        "--stream",
        action="store_true",
        help="emit lightweight NDJSON progress events to stderr",
    )

    m = sub.add_parser("mutate", help="Mutate residue(s)")
    m.add_argument("-i", "--input")
    m.add_argument("-S", "--sequence")
    m.add_argument("-t", "--three-letter")
    m.add_argument("-m", "--mutations", required=True)
    m.add_argument("-o", "--output")
    m.add_argument("-f", "--format")
    m.add_argument("--oxt", action="store_true")
    m.add_argument("--preset")
    m.add_argument("--detect-ss", action="store_true")
    m.add_argument(
        "--stream",
        action="store_true",
        help="emit lightweight NDJSON progress events to stderr",
    )

    return parser


def run_cli(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    pep_build, pep_mutate = _load_native()

    stream_enabled = bool(getattr(args, "_global_stream", False) or getattr(args, "stream", False))
    def emit_stream(payload: dict) -> None:
        if stream_enabled:
            print(json.dumps(payload), file=sys.stderr, flush=True)

    if args.command == "build":
        total_residues = (
            len(args.sequence)
            if args.sequence
            else len(args.three_letter.split("-")) if args.three_letter else 0
        )
        emit_stream(
            {
                "event": "operation_started",
                "operation": "build",
                "input_path": args.json or args.sequence or args.three_letter,
                "total_chains": 1,
                "total_residues": total_residues,
                "total_mutations": None,
            }
        )
        t0 = time.perf_counter()
        try:
            pep_build(
                sequence=args.sequence,
                three_letter=args.three_letter,
                json_path=args.json,
                output=args.output,
                format=args.format,
                oxt=bool(args.oxt),
                preset=args.preset,
                phi=_parse_angles_csv(args.phi),
                psi=_parse_angles_csv(args.psi),
                omega=_parse_angles_csv(args.omega),
                detect_ss=bool(args.detect_ss),
            )
        except Exception as exc:
            emit_stream({"event": "error", "message": str(exc)})
            raise
        elapsed_ms = int((time.perf_counter() - t0) * 1000.0)
        emit_stream(
            {
                "event": "operation_complete",
                "operation": "build",
                "total_atoms": 0,
                "total_residues": total_residues,
                "total_chains": 1,
                "output_path": args.output,
                "elapsed_ms": elapsed_ms,
            }
        )
        return 0

    if args.command == "mutate":
        mutation_specs = _parse_mutations(args.mutations)
        emit_stream(
            {
                "event": "operation_started",
                "operation": "mutate",
                "input_path": args.input or args.sequence or args.three_letter,
                "total_chains": 1,
                "total_residues": 0,
                "total_mutations": len(mutation_specs),
            }
        )
        t0 = time.perf_counter()
        try:
            pep_mutate(
                input_path=args.input,
                sequence=args.sequence,
                three_letter=args.three_letter,
                mutations=mutation_specs,
                output=args.output,
                format=args.format,
                oxt=bool(args.oxt),
                preset=args.preset,
                detect_ss=bool(args.detect_ss),
            )
        except Exception as exc:
            emit_stream({"event": "error", "message": str(exc)})
            raise
        elapsed_ms = int((time.perf_counter() - t0) * 1000.0)
        emit_stream(
            {
                "event": "operation_complete",
                "operation": "mutate",
                "total_atoms": 0,
                "total_residues": 0,
                "total_chains": 1,
                "output_path": args.output,
                "elapsed_ms": elapsed_ms,
            }
        )
        return 0

    raise RuntimeError(f"unknown command: {args.command}")


def main(argv: Optional[List[str]] = None) -> int:
    try:
        return run_cli(argv)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
