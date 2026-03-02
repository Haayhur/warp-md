"""
warp-pack CrewAI Tool with Streaming Support

This example demonstrates CrewAI integration with warp-pack streaming,
enabling agents to monitor and report on packing progress.

Usage:
    python warp_pack_streaming.py --config pack.yaml --output packed.pdb
"""

import json
import subprocess
from pathlib import Path
from typing import Type

from crewai_tools import ToolBase
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from warp_utils import (
    parse_stream_events,
    WarpPackEventHandler,
    ProgressTracker,
    PackCompleteEvent,
    GencanIterationEvent,
    PhaseCompleteEvent,
)


class WarpPackStreamingInput(BaseModel):
    """Input schema for warp-pack streaming."""

    config: str = Field(
        ...,
        description="Path to packing configuration file",
    )
    output: str = Field(
        ...,
        description="Output file path",
    )
    format: str = Field(
        default="pdb",
        description="Output format (default: pdb)",
    )


class StreamingWarpPackTool(ToolBase):
    """
    CrewAI tool for warp-pack with streaming progress.

    This tool provides real-time progress updates during molecular packing,
    allowing agents to report status to users.

    Example:
        >>> tool = StreamingWarpPackTool()
        >>> result = tool.run(
        ...     config="pack.yaml",
        ...     output="packed.pdb"
        ... )
    """

    name: str = "warp_pack_streaming"
    description: str = """
    Perform molecular packing using warp-pack with real-time progress.

    Supports:
    - Multiple molecule types
    - Box and spherical constraints
    - Various output formats (PDB, GRO, MOL2, etc.)

    Progress events track:
    - Template loading
    - Core molecule placement
    - GenCan optimization iterations
    - Movebad refinement passes
    - Final relaxation

    Returns: JSON with output path and statistics
    """
    args_schema: Type[BaseModel] = WarpPackStreamingInput

    def _run(
        self,
        config: str,
        output: str,
        format: str = "pdb",
    ) -> str:
        """Execute warp-pack with streaming."""
        cmd = [
            "warp-pack",
            "--config", config,
            "--output", output,
            "--format", format,
            "--stream",
        ]

        handler = CrewaiPackHandler()
        final_event = self._run_with_stream(cmd, handler)

        if final_event:
            # Build summary report
            report = {
                "status": "success",
                "output_path": output,
                "total_atoms": final_event.get("total_atoms"),
                "total_molecules": final_event.get("total_molecules"),
                "elapsed_seconds": final_event.get("elapsed_ms", 0) / 1000.0,
                "iterations": len(handler.iteration_history),
                "phases_completed": handler.phases_completed,
            }
            return json.dumps(report, indent=2)

        return json.dumps({"status": "success", "output_path": output})

    def _run_with_stream(
        self,
        cmd: list,
        handler: WarpPackEventHandler,
    ) -> dict:
        """Run command and parse streaming events."""
        proc = subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        return parse_stream_events(proc, handler)


class CrewaiPackHandler(WarpPackEventHandler):
    """
    Event handler that tracks packing progress for CrewAI agents.

    Accumulates progress information that can be reported to the user.
    """

    def __init__(self):
        self.iteration_history = []
        self.phases_completed = []
        self.current_phase = None
        self.molecules_placed = 0
        self.start_time = None

    def on_pack_started(self, event) -> None:
        self.molecules_placed = 0

    def on_phase_started(self, event) -> None:
        self.current_phase = event.phase

    def on_gencan_iteration(self, event: GencanIterationEvent) -> None:
        # Store every 10th iteration to avoid excessive memory
        if event.iteration % 10 == 0:
            self.iteration_history.append({
                "iteration": event.iteration,
                "obj_value": event.obj_value,
                "pg_sup": event.pg_sup,
                "progress_pct": event.progress_pct,
            })

    def on_phase_complete(self, event: PhaseCompleteEvent) -> None:
        self.phases_completed.append(event.phase)
        self.current_phase = None

    def get_progress_summary(self) -> str:
        """Get a human-readable progress summary."""
        parts = []
        if self.phases_completed:
            parts.append(f"Completed phases: {', '.join(self.phases_completed)}")
        if self.iteration_history:
            latest = self.iteration_history[-1]
            parts.append(f"Latest iteration: {latest['iteration']}, progress: {latest['progress_pct']:.1f}%")
        if self.current_phase:
            parts.append(f"Current phase: {self.current_phase}")
        return ". ".join(parts) if parts else "No progress yet"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="warp-pack CrewAI streaming example"
    )
    parser.add_argument("--config", "-c", required=True, help="Packing config file")
    parser.add_argument("--output", "-o", required=True, help="Output file")
    parser.add_argument("--format", "-f", default="pdb", help="Output format")

    args = parser.parse_args()

    tool = StreamingWarpPackTool()
    result = tool._run(
        config=args.config,
        output=args.output,
        format=args.format,
    )
    print(result)
