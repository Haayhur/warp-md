"""
warp-md OpenAI Agents SDK Integration

This module provides OpenAI Agents SDK tools for warp-md,
enabling modern OpenAI agents to perform molecular dynamics analysis.

Installation:
    pip install openai-agents warp-md

Example:
    from openai import OpenAI
    from openai_agents import Agent, Runner
    from warp_md_openai_tool import WarpMDTool

    # Create agent with warp-md tool
    agent = Agent(
        name="md_analyst",
        instructions="Analyze MD trajectories using warp-md",
        tools=[WarpMDTool()],
    )

    # Run analysis
    result = await Runner.run(
        agent,
        input="Calculate Rg for the protein in protein.pdb and traj.xtc"
    )
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional

from openai.types import FunctionDefinition, FunctionParameters


class WarpMDTool:
    """
    OpenAI Agents SDK tool for warp-md trajectory analysis.

    This class implements the OpenAI function calling interface,
    allowing agents to invoke warp-md analyses directly.

    Example:
        >>> tool = WarpMDTool()
        >>> result = tool.call(
        ...     topology="protein.pdb",
        ...     trajectory="traj.xtc",
        ...     analyses=[{"name": "rg", "selection": "protein"}]
        ... )
    """

    name: str = "warp_md_analysis"

    description: str = """
    Perform molecular dynamics trajectory analysis using warp-md.

    Supports 50+ analyses including:
    - Structural: radius of gyration (rg), RMSD, RMSF, end-to-end distance
    - Dynamics: mean squared displacement (msd), rotational autocorrelation
    - Transport: diffusion, conductivity, dielectric properties
    - Solvation: radial distribution functions (rdf), hydrogen bonds (hbond)
    - Polymer: persistence length, contour length, chain Rg

    Returns a JSON envelope with result file paths and metadata.
    """

    def to_function_definition(self) -> FunctionDefinition:
        """Convert to OpenAI function definition format."""
        return FunctionDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "topology": {
                        "type": "string",
                        "description": "Path to topology file (PDB, GRO, PDBQT)",
                    },
                    "trajectory": {
                        "type": "string",
                        "description": "Path to trajectory file (DCD, XTC, PDBQT)",
                    },
                    "analyses": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Analysis name (e.g., 'rg', 'rmsd', 'rdf')",
                                },
                                "selection": {
                                    "type": "string",
                                    "description": "Atom selection mask (for most analyses)",
                                },
                                "sel_a": {
                                    "type": "string",
                                    "description": "First selection for RDF",
                                },
                                "sel_b": {
                                    "type": "string",
                                    "description": "Second selection for RDF",
                                },
                                "bins": {
                                    "type": "integer",
                                    "description": "Number of histogram bins",
                                },
                                "r_max": {
                                    "type": "number",
                                    "description": "Maximum distance for histograms",
                                },
                            },
                        },
                        "description": "Array of analysis specifications",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Output directory (default: current directory)",
                    },
                    "device": {
                        "type": "string",
                        "description": "Compute device: 'auto', 'cpu', 'cuda' (default: auto)",
                    },
                },
                "required": ["topology", "trajectory", "analyses"],
            },
        )

    def call(
        self,
        topology: str,
        trajectory: str,
        analyses: list[dict[str, Any]],
        output_dir: str = ".",
        device: str = "auto",
    ) -> dict[str, Any]:
        """
        Execute warp-md analysis.

        Args:
            topology: Path to topology file
            trajectory: Path to trajectory file
            analyses: List of analysis specifications
            output_dir: Output directory
            device: Compute device

        Returns:
            Parsed result envelope as dictionary
        """
        # Build request
        run_request = {
            "version": "warp-md.agent.v1",
            "system": topology,
            "trajectory": trajectory,
            "device": device,
            "output_dir": output_dir,
            "analyses": analyses,
        }

        # Write to temp file
        config_path = Path(output_dir) / "_warp_md_openai_request.json"
        try:
            with open(config_path, "w") as f:
                json.dump(run_request, f, indent=2)

            # Execute
            result = subprocess.run(
                ["warp-md", "run", str(config_path)],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                return {
                    "status": "error",
                    "exit_code": result.returncode,
                    "stderr": result.stderr,
                    "stdout": result.stdout,
                }

            # Parse output
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return {"raw_output": result.stdout}

        finally:
            if config_path.exists():
                config_path.unlink()


# For async compatibility
async def acall(
    self,
    topology: str,
    trajectory: str,
    analyses: list[dict[str, Any]],
    output_dir: str = ".",
    device: str = "auto",
) -> dict[str, Any]:
    """Async version of call."""
    # Subprocess is synchronous, so we just wrap it
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: self.call(topology, trajectory, analyses, output_dir, device),
    )


# Monkey-patch async method
WarpMDTool.acall = acall


if __name__ == "__main__":
    tool = WarpMDTool()
    print(f"Tool: {tool.name}")
    print(f"Definition:\n{json.dumps(tool.to_function_definition(), indent=2)}")
