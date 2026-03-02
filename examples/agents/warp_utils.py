"""
Utilities for parsing warp-md NDJSON streaming events.

This module provides parsers and event handlers for warp-md tool streaming,
enabling agents to monitor progress in real-time.

Usage:
    from warp_utils import parse_stream_events, WarpPackEventHandler

    class MyHandler(WarpPackEventHandler):
        def on_gencan_iteration(self, event):
            print(f"Iteration {event.iteration}: {event.progress_pct:.1f}%")

    handler = MyHandler()
    parse_stream_events(process, handler)
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PackStartedEvent:
    total_molecules: int
    box_size: List[float]
    box_origin: List[float]
    output_path: Optional[str]


@dataclass
class PhaseStartedEvent:
    phase: str
    total_molecules: Optional[int]
    max_iterations: Optional[int]


@dataclass
class MoleculePlacedEvent:
    molecule_index: int
    total_molecules: int
    molecule_name: str
    successful: bool
    progress_pct: float


@dataclass
class GencanIterationEvent:
    iteration: int
    max_iterations: int
    obj_value: float
    obj_overlap: float
    obj_constraint: float
    pg_sup: float
    pg_norm: float
    elapsed_ms: int
    progress_pct: float
    eta_ms: int


@dataclass
class PhaseCompleteEvent:
    phase: str
    elapsed_ms: int
    iterations: Optional[int]
    final_obj_value: Optional[float]


@dataclass
class PackCompleteEvent:
    total_atoms: int
    total_molecules: int
    final_box_size: List[float]
    output_path: Optional[str]
    elapsed_ms: int
    profile_ms: Dict[str, int]


@dataclass
class OperationStartedEvent:
    operation: str
    input_path: Optional[str]
    total_chains: int
    total_residues: int
    total_mutations: Optional[int]


@dataclass
class MutationCompleteEvent:
    mutation_index: int
    total_mutations: int
    mutation_spec: str
    successful: bool
    elapsed_ms: int
    progress_pct: float


@dataclass
class OperationCompleteEvent:
    operation: str
    total_atoms: int
    total_residues: int
    total_chains: int
    output_path: Optional[str]
    elapsed_ms: int


class WarpPackEventHandler:
    """Base handler for warp-pack streaming events."""

    def on_pack_started(self, event: PackStartedEvent) -> None:
        pass

    def on_phase_started(self, event: PhaseStartedEvent) -> None:
        pass

    def on_molecule_placed(self, event: MoleculePlacedEvent) -> None:
        pass

    def on_gencan_iteration(self, event: GencanIterationEvent) -> None:
        pass

    def on_phase_complete(self, event: PhaseCompleteEvent) -> None:
        pass

    def on_pack_complete(self, event: PackCompleteEvent) -> None:
        pass

    def on_error(self, code: str, message: str, context: Any) -> None:
        pass


class WarpPepEventHandler:
    """Base handler for warp-pep streaming events."""

    def on_operation_started(self, event: OperationStartedEvent) -> None:
        pass

    def on_mutation_complete(self, event: MutationCompleteEvent) -> None:
        pass

    def on_operation_complete(self, event: OperationCompleteEvent) -> None:
        pass

    def on_error(self, code: str, message: str) -> None:
        pass


def parse_stream_events(
    process: subprocess.Popen,
    handler: WarpPackEventHandler | WarpPepEventHandler,
) -> Optional[Dict[str, Any]]:
    """
    Parse NDJSON streaming events from a subprocess.

    Args:
        process: Subprocess running with --stream flag
        handler: Event handler instance

    Returns:
        Final result envelope (if any)

    Example:
        proc = subprocess.Popen(
            ["warp-pack", "--config", "pack.yaml", "--stream"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        result = parse_stream_events(proc, my_handler)
    """
    final_result = None

    while True:
        line = process.stderr.readline()
        if not line:
            break

        try:
            raw = line.decode().strip() if isinstance(line, bytes) else line.strip()
            if not raw:
                continue

            event = json.loads(raw)
            event_type = event.get("event")
            payload = {k: v for k, v in event.items() if k != "event"}

            if event_type == "pack_started":
                if isinstance(handler, WarpPackEventHandler):
                    handler.on_pack_started(PackStartedEvent(**payload))
            elif event_type == "phase_started":
                if isinstance(handler, WarpPackEventHandler):
                    handler.on_phase_started(PhaseStartedEvent(**payload))
            elif event_type == "molecule_placed":
                if isinstance(handler, WarpPackEventHandler):
                    handler.on_molecule_placed(MoleculePlacedEvent(**payload))
            elif event_type == "gencan_iteration":
                if isinstance(handler, WarpPackEventHandler):
                    handler.on_gencan_iteration(GencanIterationEvent(**payload))
            elif event_type == "phase_complete":
                if isinstance(handler, WarpPackEventHandler):
                    handler.on_phase_complete(PhaseCompleteEvent(**payload))
            elif event_type == "pack_complete":
                final_result = event
                if isinstance(handler, WarpPackEventHandler):
                    handler.on_pack_complete(PackCompleteEvent(**payload))
            elif event_type == "operation_started":
                if isinstance(handler, WarpPepEventHandler):
                    handler.on_operation_started(OperationStartedEvent(**payload))
            elif event_type == "mutation_complete":
                if isinstance(handler, WarpPepEventHandler):
                    handler.on_mutation_complete(MutationCompleteEvent(**payload))
            elif event_type == "operation_complete":
                final_result = event
                if isinstance(handler, WarpPepEventHandler):
                    handler.on_operation_complete(OperationCompleteEvent(**payload))
            elif event_type == "error":
                code = event.get("code", "UNKNOWN")
                message = event.get("message", "")
                context = event.get("context")
                if isinstance(handler, WarpPackEventHandler):
                    handler.on_error(code, message, context)
                elif isinstance(handler, WarpPepEventHandler):
                    handler.on_error(code, message)

        except (json.JSONDecodeError, TypeError):
            # Skip malformed lines
            continue

    process.wait()
    return final_result


class ProgressTracker(WarpPackEventHandler, WarpPepEventHandler):
    """
    Progress tracker that prints formatted progress updates.

    Example:
        tracker = ProgressTracker()
        proc = subprocess.Popen(["warp-pack", "--stream", ...], ...)
        parse_stream_events(proc, tracker)
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _print(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # Warp-pack events
    def on_pack_started(self, event: PackStartedEvent) -> None:
        self._print(f"📦 Packing {event.total_molecules} molecules...")

    def on_phase_started(self, event: PhaseStartedEvent) -> None:
        phase_names = {
            "template_load": "Loading templates",
            "core_placement": "Placing molecules",
            "movebad": "Refining (movebad)",
            "gencan": "Optimizing (GenCan)",
            "relax": "Relaxing overlaps",
        }
        name = phase_names.get(event.phase, event.phase)
        if event.max_iterations:
            self._print(f"  → {name} (max {event.max_iterations} iterations)")
        else:
            self._print(f"  → {name}...")

    def on_molecule_placed(self, event: MoleculePlacedEvent) -> None:
        if event.molecule_index % 10 == 0 or event.molecule_index == event.total_molecules:
            self._print(f"    Placed {event.molecule_index}/{event.total_molecules} ({event.progress_pct:.1f}%)")

    def on_gencan_iteration(self, event: GencanIterationEvent) -> None:
        if event.iteration % 10 == 0 or event.iteration == event.max_iterations:
            obj_str = f"{event.obj_value:.2e}"
            self._print(f"    Iter {event.iteration}: f={obj_str}, pg={event.pg_sup:.2e} ({event.progress_pct:.1f}%)")

    def on_phase_complete(self, event: PhaseCompleteEvent) -> None:
        sec = event.elapsed_ms / 1000.0
        if event.final_obj_value:
            self._print(f"  ✓ {event.phase} complete in {sec:.2f}s (f={event.final_obj_value:.2e})")
        else:
            self._print(f"  ✓ {event.phase} complete in {sec:.2f}s")

    def on_pack_complete(self, event: PackCompleteEvent) -> None:
        sec = event.elapsed_ms / 1000.0
        self._print(f"✅ Pack complete: {event.total_atoms} atoms, {event.total_molecules} molecules in {sec:.2f}s")

    # Warp-pep events
    def on_operation_started(self, event: OperationStartedEvent) -> None:
        if event.total_mutations:
            self._print(f"🧬 {event.operation}: {event.total_mutations} mutations...")
        else:
            self._print(f"🧬 {event.operation}...")

    def on_mutation_complete(self, event: MutationCompleteEvent) -> None:
        self._print(f"    Mutation {event.mutation_index}/{event.total_mutations}: {event.mutation_spec}")

    def on_operation_complete(self, event: OperationCompleteEvent) -> None:
        sec = event.elapsed_ms / 1000.0
        self._print(f"✅ {event.operation} complete: {event.total_atoms} atoms in {sec:.2f}s")

    # Error handling
    def on_error(self, code: str, message: str, context: Any = None) -> None:
        self._print(f"❌ Error [{code}]: {message}")


def run_with_progress(
    cmd: List[str],
    handler: Optional[WarpPackEventHandler | WarpPepEventHandler] = None,
) -> Optional[Dict[str, Any]]:
    """
    Run a warp command with streaming progress.

    Args:
        cmd: Command list with --stream flag included
        handler: Event handler (uses ProgressTracker if None)

    Returns:
        Final result envelope

    Example:
        result = run_with_progress([
            "warp-pack", "--config", "pack.yaml", "--stream"
        ])
    """
    if handler is None:
        handler = ProgressTracker()

    proc = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    return parse_stream_events(proc, handler)
