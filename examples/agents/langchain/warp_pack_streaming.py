"""
warp-pack LangChain Tool with Streaming Support

This example demonstrates how to create a LangChain tool that uses
warp-pack with streaming progress events.

Usage:
    python warp_pack_streaming.py --config pack.yaml --output packed.pdb
"""

import json
import subprocess
from pathlib import Path
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from warp_utils import (
    ProgressTracker,
    parse_stream_events,
    WarpPackEventHandler,
    PackCompleteEvent,
    GencanIterationEvent,
)


class WarpPackInput(BaseModel):
    """Input schema for warp-pack with streaming."""

    config: str = Field(
        ...,
        description="Path to packing configuration file (YAML, JSON, or Packmol INP)",
    )
    output: str = Field(
        ...,
        description="Output file path for the packed structure",
    )
    format: Optional[str] = Field(
        default="pdb",
        description="Output format (default: pdb)",
    )
    stream: bool = Field(
        default=True,
        description="Enable NDJSON streaming progress events",
    )


class StreamingWarpPackTool(BaseTool):
    """
    LangChain tool for warp-pack with streaming progress.

    This tool runs molecular packing and provides real-time progress
    updates through streaming events.

    Example:
        >>> tool = StreamingWarpPackTool()
        >>> result = tool.run(
        ...     config="pack.yaml",
        ...     output="packed.pdb"
        ... )
    """

    name: str = "warp_pack_streaming"
    description: str = """
    Perform molecular packing using warp-pack with streaming progress.

    Supports various file formats (PDB, GRO, MOL2, etc.) and constraints
    (spheres, boxes, custom geometries).

    Input requirements:
    - config: Path to configuration file (YAML, JSON, or INP)
    - output: Output file path
    - format: Optional output format (default: pdb)

    Emits real-time progress events via stderr when --stream is enabled.
    """
    args_schema: Type[BaseModel] = WarpPackInput

    def _run(
        self,
        config: str,
        output: str,
        format: str = "pdb",
        stream: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute warp-pack with streaming."""
        cmd = [
            "warp-pack",
            "--config", config,
            "--output", output,
            "--format", format,
        ]
        if stream:
            cmd.append("--stream")

        handler = ProgressTracker(verbose=False)
        final_event = self._run_with_stream(cmd, handler)

        if final_event:
            return json.dumps(final_event, indent=2)
        return f"Successfully packed structure to {output}"

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


def create_streaming_warp_pack_tool() -> "StreamingWarpPackTool":
    """Factory for creating a configured streaming warp-pack tool."""
    return StreamingWarpPackTool()


class CallbackWarpPackEventHandler(WarpPackEventHandler):
    """
    Event handler that calls LangChain callbacks.

    This allows agents to receive structured progress updates during packing.
    """

    def __init__(self, callback_manager: Optional[CallbackManagerForToolRun] = None):
        self.callback_manager = callback_manager
        self.iterations = []
        self.phases = []

    def on_pack_started(self, event) -> None:
        if self.callback_manager:
            self.callback_manager.on_text(
                f"Starting pack: {event.total_molecules} molecules\n",
            )

    def on_phase_started(self, event) -> None:
        self.phases.append(event.phase)
        if self.callback_manager:
            self.callback_manager.on_text(f"Phase: {event.phase}\n")

    def on_gencan_iteration(self, event: GencanIterationEvent) -> None:
        self.iterations.append(event)
        if event.iteration % 50 == 0:
            if self.callback_manager:
                self.callback_manager.on_text(
                    f"Progress: {event.progress_pct:.1f}% (f={event.obj_value:.2e})\n",
                )

    def on_pack_complete(self, event: PackCompleteEvent) -> None:
        if self.callback_manager:
            sec = event.elapsed_ms / 1000.0
            self.callback_manager.on_text(
                f"Complete: {event.total_atoms} atoms in {sec:.1f}s\n",
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="warp-pack streaming example"
    )
    parser.add_argument("--config", "-c", required=True, help="Packing config file")
    parser.add_argument("--output", "-o", required=True, help="Output file")
    parser.add_argument("--format", "-f", default="pdb", help="Output format")

    args = parser.parse_args()

    tool = create_streaming_warp_pack_tool()
    result = tool._run(
        config=args.config,
        output=args.output,
        format=args.format,
        stream=True,
    )
    print(result)
