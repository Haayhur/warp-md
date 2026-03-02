"""
warp-md CrewAI Tool Integration

This module provides CrewAI tools for warp-md, enabling agent crews
to perform molecular dynamics trajectory analysis.

Installation:
    pip install crewai crewai-tools warp-md

Example:
    from crewai import Agent, Task, Crew, Process
    from warp_md_crewai_tool import WarpMDAnalysisTool

    # Create the MD analyst agent
    md_analyst = Agent(
        role="Molecular Dynamics Analyst",
        goal="Analyze MD trajectories to extract structural insights",
        backstory="""You are an expert in molecular dynamics simulation
        analysis with deep knowledge of protein structure, dynamics, and
        statistical mechanics.""",
        tools=[WarpMDAnalysisTool()],
        verbose=True
    )

    # Define analysis task
    analysis_task = Task(
        description="Calculate radius of gyration for the protein",
        expected_output="Path to the results file",
        agent=md_analyst
    )

    # Execute
    crew = Crew(agents=[md_analyst], tasks=[analysis_task])
    result = crew.kickoff()
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional, Type

from crewai_tools import ToolBase
from pydantic import BaseModel, Field


class WarpMDInput(BaseModel):
    """Input schema for warp-md analysis in CrewAI."""

    topology: str = Field(
        ...,
        description="Path to topology file (PDB, GRO, or PDBQT)",
    )
    trajectory: str = Field(
        ...,
        description="Path to trajectory file (DCD, XTC, or PDB/PDBQT)",
    )
    analyses: str = Field(
        ...,
        description=(
            "JSON array of analysis specifications. "
            "Example: [{\"name\": \"rg\", \"selection\": \"protein\"}]"
        ),
    )
    output_dir: str = Field(
        default=".",
        description="Output directory for results",
    )
    device: str = Field(
        default="auto",
        description="Compute device: 'auto', 'cpu', 'cuda'",
    )


class WarpMDAnalysisTool(ToolBase):
    """
    CrewAI tool for warp-md trajectory analysis.

    This tool integrates warp-md's analysis capabilities into CrewAI
    workflows, enabling multi-agent collaboration on molecular dynamics
    analysis tasks.

    Attributes:
        name: Tool identifier
        description: Tool description for agents
        args_schema: Input validation schema

    Example:
        >>> tool = WarpMDAnalysisTool()
        >>> result = tool.run(
        ...     topology="protein.pdb",
        ...     trajectory="traj.xtc",
        ...     analyses='[{"name": "rg", "selection": "protein"}]'
        ... )
    """

    name: str = "warp_md_analysis"
    description: str = """
    Perform molecular dynamics trajectory analysis using warp-md.

    Supports 50+ analyses:
    - Structural: radius of gyration (rg), RMSD, RMSF, end-to-end distance
    - Dynamics: mean squared displacement (msd), rotational autocorrelation
    - Transport: diffusion, conductivity, dielectric properties
    - Solvation: radial distribution functions (rdf), hydrogen bonds
    - Polymer: persistence length, contour length, chain Rg

    Input: topology path, trajectory path, JSON analysis specifications
    Output: JSON result envelope with artifact paths
    """
    args_schema: Type[BaseModel] = WarpMDInput

    def _run(
        self,
        topology: str,
        trajectory: str,
        analyses: str,
        output_dir: str = ".",
        device: str = "auto",
    ) -> str:
        """
        Execute warp-md analysis.

        Args:
            topology: Path to topology file
            trajectory: Path to trajectory file
            analyses: JSON string of analysis specs
            output_dir: Output directory
            device: Compute device

        Returns:
            JSON result envelope as string
        """
        # Parse analyses
        try:
            analyses_list = json.loads(analyses)
            if not isinstance(analyses_list, list):
                raise ValueError("analyses must be a JSON array")
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON: {e}"})

        # Build request
        run_request = {
            "version": "warp-md.agent.v1",
            "system": topology,
            "trajectory": trajectory,
            "device": device,
            "output_dir": output_dir,
            "analyses": analyses_list,
        }

        # Write to temp file
        config_path = Path(output_dir) / "_warp_md_crewai_request.json"
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
                return json.dumps({
                    "status": "error",
                    "exit_code": result.returncode,
                    "stderr": result.stderr,
                })

            return result.stdout

        finally:
            if config_path.exists():
                config_path.unlink()


# Convenience wrapper for single analyses
class QuickWarpMDTool(ToolBase):
    """
    Simplified CrewAI tool for single warp-md analyses.

    This tool provides a simpler interface for common analyses
    without requiring full JSON specification.

    Example:
        >>> tool = QuickWarpMDTool()
        >>> result = tool.run(
        ...     topology="protein.pdb",
        ...     trajectory="traj.xtc",
        ...     analysis="rg",
        ...     selection="protein"
        ... )
    """

    name: str = "quick_warp_md_analysis"
    description: str = """
    Perform a single MD trajectory analysis.

    Analyses: rg, rmsd, msd, rdf, end_to_end, persistence_length, etc.
    For rdf: provide sel_a, sel_b, bins, r_max
    For conductivity: provide charges, temperature
    """
    args_schema: Type[BaseModel] = BaseModel

    def _run(
        self,
        topology: str,
        trajectory: str,
        analysis: str,
        **params,
    ) -> str:
        """Run a single analysis."""
        analyses_spec = [{"name": analysis, **params}]
        tool = WarpMDAnalysisTool()
        return tool._run(
            topology=topology,
            trajectory=trajectory,
            analyses=json.dumps(analyses_spec),
        )


if __name__ == "__main__":
    print("warp-md CrewAI Tools")
    print("=" * 50)
    print(f"Main tool: {WarpMDAnalysisTool.name}")
    print(f"Quick tool: {QuickWarpMDTool.name}")
