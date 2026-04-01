from __future__ import annotations

import argparse
from pathlib import Path

from . import FrameEditor

_INPUT_TRAJECTORY_FORMATS = ["dcd", "xtc", "trr", "pdb", "pdbqt"]


def _validate_frame_edit_paths(args: argparse.Namespace) -> None:
    topology_path = Path(args.topology).expanduser()
    if not topology_path.is_file():
        raise FileNotFoundError(f"topology file not found: {topology_path}")

    traj_path = Path(args.traj).expanduser()
    if not traj_path.is_file():
        raise FileNotFoundError(f"trajectory file not found: {traj_path}")

    out_path = Path(args.out).expanduser()
    parent = out_path.parent
    if parent and not parent.exists():
        raise FileNotFoundError(f"output directory not found: {parent}")


def add_frame_edit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-p",
        "--topology",
        "--pdbfile",
        dest="topology",
        required=True,
        help="Topology file (.pdb or .gro)",
    )
    parser.add_argument(
        "-t",
        "--traj",
        required=True,
        help="Input trajectory file (.dcd, .xtc, .trr, .pdb, .pdbqt)",
    )
    parser.add_argument(
        "-o",
        "--out",
        "--outfile",
        dest="out",
        required=True,
        help="Output file (.pdb, .gro, .dcd, .xtc, .trr)",
    )
    parser.add_argument("-b", "--begin", type=int, default=0, help="Starting frame index")
    parser.add_argument("-e", "--end", type=int, help="Ending frame index (exclusive)")
    parser.add_argument("-s", "--step", type=int, default=1, help="Step size between frames")
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        help="Single frame index to extract; overrides begin/end/step",
    )
    parser.add_argument("--topology-format", choices=["pdb", "gro"], help="Override topology format")
    parser.add_argument(
        "--traj-format",
        choices=_INPUT_TRAJECTORY_FORMATS,
        help="Override trajectory format",
    )
    parser.add_argument(
        "--traj-length-scale",
        type=float,
        help="DCD length scale (for example 10.0 for nm->A)",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        help="Frames per read chunk while scanning and writing",
    )


def run_frame_edit(args: argparse.Namespace) -> tuple[int, dict]:
    _validate_frame_edit_paths(args)
    payload = FrameEditor.run(
        topology_path=args.topology,
        traj_path=args.traj,
        out_path=args.out,
        begin=args.begin,
        end=args.end,
        step=args.step,
        index=args.index,
        topology_format=args.topology_format,
        traj_format=args.traj_format,
        traj_length_scale=args.traj_length_scale,
        chunk_frames=args.chunk_frames,
    )
    return 0, payload
