"""
Python wrapper for warp-pack molecular packing.

This module provides a Python interface to warp-pack, enabling
agent-friendly packing operations with structured output.

Usage:
    from warp_pack_wrapper import run_pack, PackConfig

    result = run_pack(
        config="pack.yaml",
        output="packed.pdb",
        stream=True
    )
    print(f"Packed {result.total_atoms} atoms")
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class PackResult:
    """Result from a molecular packing operation."""
    total_atoms: int
    total_molecules: int
    final_box_size: tuple[float, float, float]
    output_path: Optional[str]
    elapsed_ms: int
    profile_ms: Dict[str, int]
    success: bool
    error: Optional[str] = None


class PackConfig:
    """
    Configuration for molecular packing.

    Supports YAML, JSON, and Packmol INP formats.
    """

    def __init__(
        self,
        filepath: Optional[str] = None,
        structures: Optional[List[Dict[str, Any]]] = None,
        box_size: Optional[tuple[float, float, float]] = None,
        **kwargs
    ):
        """
        Create a packing configuration.

        Args:
            filepath: Path to existing config file (YAML/JSON/INP)
            structures: List of structure specifications
            box_size: Simulation box dimensions (x, y, z)
            **kwargs: Additional parameters
        """
        self.filepath = filepath
        self.structures = structures or []
        self.box_size = box_size
        self.params = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {}
        if self.filepath:
            d["file"] = self.filepath
        if self.structures:
            d["structures"] = self.structures
        if self.box_size:
            d["box_"] = {"size": list(self.box_size)}
        d.update(self.params)
        return d


def run_pack(
    config: Union[str, PackConfig, Dict[str, Any]],
    output: str,
    format: str = "pdb",
    stream: bool = False,
) -> PackResult:
    """
    Run molecular packing using warp-pack.

    Args:
        config: Path to config file (YAML/JSON/INP) or PackConfig object
        output: Output file path
        format: Output format (pdb, gro, mol2, etc.)
        stream: Enable NDJSON streaming progress

    Returns:
        PackResult with statistics and status

    Example:
        >>> result = run_pack(
        ...     config="pack.yaml",
        ...     output="packed.pdb",
        ...     stream=True
        ... )
        >>> print(f"Packed {result.total_atoms} atoms")
    """
    config_path: Optional[str] = None
    temp_config_path: Optional[Path] = None

    if isinstance(config, str):
        config_path = config
    elif isinstance(config, PackConfig):
        if config.filepath:
            config_path = config.filepath
        else:
            payload = config.to_dict()
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
                json.dump(payload, fh)
                temp_config_path = Path(fh.name)
            config_path = str(temp_config_path)
    elif isinstance(config, dict):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(config, fh)
            temp_config_path = Path(fh.name)
        config_path = str(temp_config_path)
    else:
        raise TypeError("config must be str, PackConfig, or dict")

    # Build command
    cmd = [
        "warp-pack",
        "--config", config_path,
        "--output", output,
        "--format", format,
    ]
    if stream:
        cmd.append("--stream")

    try:
        # Execute
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
                if event.get("event") == "pack_complete":
                    final_event = event
            except (json.JSONDecodeError, TypeError):
                pass

        proc.wait()

        if proc.returncode != 0:
            _stdout, stderr = proc.communicate()
            return PackResult(
                total_atoms=0,
                total_molecules=0,
                final_box_size=(0, 0, 0),
                output_path=None,
                elapsed_ms=0,
                profile_ms={},
                success=False,
                error=stderr or "Unknown error",
            )

        if final_event:
            return PackResult(
                total_atoms=final_event.get("total_atoms", 0),
                total_molecules=final_event.get("total_molecules", 0),
                final_box_size=tuple(final_event.get("final_box_size", [0, 0, 0])),
                output_path=final_event.get("output_path"),
                elapsed_ms=final_event.get("elapsed_ms", 0),
                profile_ms=final_event.get("profile_ms", {}),
                success=True,
            )

        # Fallback: check if output file exists
        if Path(output).exists():
            return PackResult(
                total_atoms=0,  # Would need to parse file
                total_molecules=0,
                final_box_size=(0, 0, 0),
                output_path=output,
                elapsed_ms=0,
                profile_ms={},
                success=True,
            )

        return PackResult(
            total_atoms=0,
            total_molecules=0,
            final_box_size=(0, 0, 0),
            output_path=output,
            elapsed_ms=0,
            profile_ms={},
            success=True,
        )
    finally:
        if temp_config_path:
            temp_config_path.unlink(missing_ok=True)


def create_simple_pack_config(
    molecules: List[Dict[str, Any]],
    box_size: tuple[float, float, float],
    output: str,
) -> PackConfig:
    """
    Create a simple packing configuration.

    Args:
        molecules: List of molecule specifications
            [{"path": "water.pdb", "count": 100}, ...]
        box_size: Box dimensions (x, y, z)
        output: Output path

    Example:
        >>> config = create_simple_pack_config(
        ...     molecules=[{"path": "water.pdb", "count": 100}],
        ...     box_size=(50, 50, 50),
        ...     output="packed.pdb"
        ... )
        >>> result = run_pack(config, "packed.pdb")
    """
    return PackConfig(
        structures=molecules,
        box_size=box_size,
        output={"path": output, "format": "pdb"},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Python wrapper for warp-pack")
    parser.add_argument("--config", "-c", required=True, help="Config file")
    parser.add_argument("--output", "-o", required=True, help="Output file")
    parser.add_argument("--format", "-f", default="pdb", help="Output format")
    parser.add_argument("--stream", "-s", action="store_true", help="Enable streaming")

    args = parser.parse_args()

    result = run_pack(
        config=args.config,
        output=args.output,
        format=args.format,
        stream=args.stream,
    )

    if result.success:
        print(f"✓ Packed {result.total_atoms} atoms in {result.elapsed_ms/1000:.2f}s")
    else:
        print(f"✗ Error: {result.error}")
        exit(1)
