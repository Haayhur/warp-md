"""
Python wrapper for warp-pep peptide building and mutation.

This module provides a Python interface to warp-pep, enabling
agent-friendly peptide construction operations.

Usage:
    from warp_pep_wrapper import build_peptide, mutate_peptide

    # Build a peptide
    result = build_peptide(sequence="ACDEFG", preset="alpha-helix")
    print(f"Built {result.total_atoms} atoms")

    # Mutate a residue
    result = mutate_peptide(
        input="peptide.pdb",
        mutations=["A5G", "L10W"],
        output="mutated.pdb"
    )
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class PepResult:
    """Result from a peptide operation."""
    total_atoms: int
    total_residues: int
    total_chains: int
    output_path: Optional[str]
    elapsed_ms: int
    success: bool
    error: Optional[str] = None


def build_peptide(
    sequence: Optional[str] = None,
    three_letter: Optional[str] = None,
    json_spec: Optional[str] = None,
    output: str = "peptide.pdb",
    format: str = "pdb",
    preset: str = "extended",
    oxt: bool = False,
    detect_ss: bool = False,
    stream: bool = False,
) -> PepResult:
    """
    Build a peptide structure.

    Args:
        sequence: One-letter amino acid sequence (e.g., "ACDEFG")
        three_letter: Dash-separated three-letter codes (e.g., "ALA-CYS-GLU")
        json_spec: Path to JSON specification file
        output: Output file path
        format: Output format (pdb, pdbx, xyz, gro, mol2, crd, lammps)
        preset: Ramachandran preset (extended, alpha-helix, beta-sheet, polyproline)
        oxt: Add terminal OXT oxygen
        detect_ss: Detect disulfide bonds
        stream: Enable NDJSON streaming

    Returns:
        PepResult with statistics

    Example:
        >>> result = build_peptide(
        ...     sequence="ACDEFG",
        ...     preset="alpha-helix",
        ...     output="helix.pdb"
        ... )
        >>> print(f"Built {result.total_atoms} atoms")
    """
    cmd = ["warp-pep", "build"]

    if json_spec:
        cmd.extend(["--json", json_spec])
    elif sequence:
        cmd.extend(["--sequence", sequence])
    elif three_letter:
        cmd.extend(["--three-letter", three_letter])
    else:
        raise ValueError("Must provide sequence, three_letter, or json_spec")

    cmd.extend([
        "--output", output,
        "--format", format,
        "--preset", preset,
    ])

    if oxt:
        cmd.append("--oxt")
    if detect_ss:
        cmd.append("--detect-ss")
    if stream:
        cmd.append("--stream")

    return _run_pep_command(cmd)


def mutate_peptide(
    input: str,
    mutations: List[str],
    output: str = "mutated.pdb",
    format: str = "pdb",
    oxt: bool = False,
    detect_ss: bool = False,
    stream: bool = False,
) -> PepResult:
    """
    Mutate residue(s) in a peptide structure.

    Args:
        input: Input structure file
        mutations: List of mutation specs (e.g., ["A5G", "L10W"])
        output: Output file path
        format: Output format
        oxt: Add terminal OXT oxygen
        detect_ss: Detect disulfide bonds
        stream: Enable NDJSON streaming

    Returns:
        PepResult with statistics

    Example:
        >>> result = mutate_peptide(
        ...     input="peptide.pdb",
        ...     mutations=["A5G", "L10W"],
        ...     output="mutated.pdb"
        ... )
        >>> print(f"Mutated {result.total_residues} residues")
    """
    if not mutations:
        raise ValueError("Must provide at least one mutation")

    cmd = [
        "warp-pep", "mutate",
        "--input", input,
        "--mutations", ",".join(mutations),
        "--output", output,
        "--format", format,
    ]

    if oxt:
        cmd.append("--oxt")
    if detect_ss:
        cmd.append("--detect-ss")
    if stream:
        cmd.append("--stream")

    return _run_pep_command(cmd)


def _run_pep_command(cmd: List[str]) -> PepResult:
    """Run a warp-pep command and parse the result."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Parse streaming events
    final_event = None
    while True:
        line = proc.stderr.readline()
        if not line:
            break
        try:
            event = json.loads(line.strip())
            if event.get("event") == "operation_complete":
                final_event = event
        except (json.JSONDecodeError, TypeError):
            pass

    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read()
        return PepResult(
            total_atoms=0,
            total_residues=0,
            total_chains=0,
            output_path=None,
            elapsed_ms=0,
            success=False,
            error=stderr,
        )

    if final_event:
        return PepResult(
            total_atoms=final_event.get("total_atoms", 0),
            total_residues=final_event.get("total_residues", 0),
            total_chains=final_event.get("total_chains", 0),
            output_path=final_event.get("output_path"),
            elapsed_ms=final_event.get("elapsed_ms", 0),
            success=True,
        )

    # Fallback: check output file
    output_path = None
    for arg in cmd:
        if arg == "--output" or arg == "-o":
            idx = cmd.index(arg)
            if idx + 1 < len(cmd):
                output_path = cmd[idx + 1]
                break

    if output_path and Path(output_path).exists():
        return PepResult(
            total_atoms=0,  # Would need to parse file
            total_residues=0,
            total_chains=0,
            output_path=output_path,
            elapsed_ms=0,
            success=True,
        )

    return PepResult(
        total_atoms=0,
        total_residues=0,
        total_chains=0,
        output_path=None,
        elapsed_ms=0,
        success=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Python wrapper for warp-pep")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build a peptide")
    build_parser.add_argument("--sequence", "-s", help="One-letter sequence")
    build_parser.add_argument("--three-letter", "-t", help="Three-letter codes")
    build_parser.add_argument("--json", "-j", help="JSON spec file")
    build_parser.add_argument("--output", "-o", default="peptide.pdb", help="Output file")
    build_parser.add_argument("--preset", "-p", default="extended", help="Ramachandran preset")
    build_parser.add_argument("--oxt", action="store_true", help="Add terminal OXT")
    build_parser.add_argument("--stream", action="store_true", help="Enable streaming")

    # Mutate command
    mutate_parser = subparsers.add_parser("mutate", help="Mutate residues")
    mutate_parser.add_argument("--input", "-i", required=True, help="Input file")
    mutate_parser.add_argument("--mutations", "-m", required=True, help="Mutation specs (comma-separated)")
    mutate_parser.add_argument("--output", "-o", default="mutated.pdb", help="Output file")

    args = parser.parse_args()

    if args.command == "build":
        result = build_peptide(
            sequence=args.sequence,
            three_letter=args.three_letter,
            json_spec=args.json,
            output=args.output,
            preset=args.preset,
            oxt=args.oxt,
            stream=args.stream,
        )
    elif args.command == "mutate":
        mutations = args.mutations.split(",") if args.mutations else []
        result = mutate_peptide(
            input=args.input,
            mutations=mutations,
            output=args.output,
            stream=args.stream,
        )
    else:
        parser.print_help()
        exit(1)

    if result.success:
        print(f"✓ {args.command} complete: {result.total_atoms} atoms in {result.elapsed_ms}ms")
    else:
        print(f"✗ Error: {result.error}")
        exit(1)
