"""
warp-md LangChain Tool Integration

This module provides a LangChain tool wrapper for warp-md, enabling
agents to perform molecular dynamics trajectory analysis through
structured tool calls.

Installation:
    pip install langchain langchain-core warp-md

Example:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_openai import ChatOpenAI
    from langchain import hub

    from warp_md_tool import WarpMDTool

    # Create the tool
    warp_tool = WarpMDTool()

    # Set up agent
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agent = create_tool_calling_agent(llm, [warp_tool], hub.pull("hwchase17/openai-tools"))
    executor = AgentExecutor(agent=agent, tools=[warp_tool], verbose=True)

    # Run analysis
    result = executor.invoke({
        "input": "Analyze the radius of gyration for protein in my_trajectory.xtc"
    })
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field


class WarpMDInput(BaseModel):
    """Input schema for warp-md analysis."""

    topology: str = Field(
        ...,
        description="Path to topology file (PDB, GRO, or PDBQT)",
    )
    trajectory: str = Field(
        ...,
        description="Path to trajectory file (DCD, XTC, or PDB/PDBQT for docking poses)",
    )
    analyses: str = Field(
        ...,
        description=(
            "JSON array of analysis specifications. Each analysis must have 'name' "
            "and analysis-specific parameters. "
            "Common analyses: 'rg', 'rmsd', 'msd', 'rdf', 'conductivity'. "
            "Example: [{\"name\": \"rg\", \"selection\": \"protein\"}]"
        ),
    )
    output_dir: str = Field(
        default=".",
        description="Output directory for results (default: current directory)",
    )
    device: str = Field(
        default="auto",
        description="Compute device: 'auto', 'cpu', 'cuda', or 'cuda:N' (default: auto)",
    )
    stream: bool = Field(
        default=False,
        description="Enable NDJSON streaming for real-time progress updates",
    )


class WarpMDTool(BaseTool):
    """
    LangChain tool for warp-md molecular dynamics trajectory analysis.

    This tool provides structured access to warp-md's trajectory analysis
    capabilities through LangChain's tool interface. It validates inputs
    using Pydantic models and returns structured JSON results.

    Attributes:
        name: Tool identifier for LangChain
        description: Tool description for LLMs
        args_schema: Pydantic schema for input validation

    Example:
        >>> tool = WarpMDTool()
        >>> result = tool.run({
        ...     "topology": "protein.pdb",
        ...     "trajectory": "traj.xtc",
        ...     "analyses": '[{"name": "rg", "selection": "protein"}]'
        ... })
    """

    name: str = "warp_md_analysis"
    description: str = """
    Perform molecular dynamics trajectory analysis using warp-md.

    Supports 50+ analyses including:
    - Structural: radius of gyration (rg), RMSD, RMSF, end-to-end distance
    - Dynamics: mean squared displacement (msd), rotational autocorrelation (rotacf)
    - Transport: diffusion, conductivity, dielectric properties
    - Solvation: radial distribution functions (rdf), hydrogen bonds (hbond)
    - Polymer: persistence length, contour length, chain Rg

    Input requirements:
    - topology: PDB/GRO/PDBQT file path
    - trajectory: DCD/XTC/PDBQT file path
    - analyses: JSON array with analysis specifications

    Each analysis requires specific parameters:
    - rg, rmsd, msd, end_to_end: "selection" (atom selection mask)
    - rdf: "sel_a", "sel_b", "bins", "r_max"
    - conductivity: "selection", "charges", "temperature"
    - hbond: "donors", "acceptors", "dist_cutoff"
    """
    args_schema: Type[BaseModel] = WarpMDInput

    def _run(
        self,
        topology: str,
        trajectory: str,
        analyses: str,
        output_dir: str = ".",
        device: str = "auto",
        stream: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Execute warp-md analysis with the provided parameters.

        Args:
            topology: Path to topology file
            trajectory: Path to trajectory file
            analyses: JSON string of analysis specifications
            output_dir: Output directory for results
            device: Compute device selection
            stream: Enable streaming progress
            run_manager: LangChain callback manager

        Returns:
            JSON string containing the analysis result envelope

        Raises:
            ValueError: If analyses JSON is invalid
            subprocess.CalledProcessError: If warp-md CLI fails
        """
        # Parse and validate analyses JSON
        try:
            analyses_list = json.loads(analyses)
            if not isinstance(analyses_list, list):
                raise ValueError("analyses must be a JSON array")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid analyses JSON: {e}")

        # Build the run request
        run_request = {
            "version": "warp-md.agent.v1",
            "system": topology,
            "trajectory": trajectory,
            "device": device,
            "stream": "ndjson" if stream else "none",
            "output_dir": output_dir,
            "analyses": analyses_list,
        }

        # Write request to temporary file
        config_path = Path(output_dir) / "_warp_md_langchain_request.json"
        try:
            with open(config_path, "w") as f:
                json.dump(run_request, f, indent=2)

            # Build command
            cmd = ["warp-md", "run", str(config_path)]
            if stream:
                cmd.append("--stream")
                cmd.append("ndjson")

            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # Parse output
            if result.returncode != 0:
                # Return error information
                return json.dumps({
                    "status": "error",
                    "exit_code": result.returncode,
                    "stderr": result.stderr,
                    "stdout": result.stdout,
                })

            # Return stdout as result (contains final envelope)
            return result.stdout

        finally:
            # Clean up temporary config
            if config_path.exists():
                config_path.unlink()


def create_warp_md_tool() -> StructuredTool:
    """
    Factory function to create a configured warp-md tool.

    Returns:
        Configured StructuredTool for LangChain

    Example:
        >>> from langchain.agents import initialize_agent, AgentType
        >>> from langchain_openai import ChatOpenAI
        >>> tool = create_warp_md_tool()
        >>> llm = ChatOpenAI()
        >>> agent = initialize_agent(
        ...     [tool],
        ...     llm,
        ...     agent=AgentType.OPENAI_FUNCTIONS
        ... )
    """
    return StructuredTool.from_function(
        func=WarpMDTool()._run,
        name="warp_md_analysis",
        description=WarpMDTool.description,
        args_schema=WarpMDInput,
    )


# Convenience function for quick analyses
def quick_analysis(
    topology: str,
    trajectory: str,
    analysis_name: str,
    **analysis_params,
) -> dict:
    """
    Perform a single analysis without full agent setup.

    Args:
        topology: Path to topology file
        trajectory: Path to trajectory file
        analysis_name: Name of the analysis (e.g., "rg", "rmsd")
        **analysis_params: Analysis-specific parameters

    Returns:
        Parsed result envelope as dictionary

    Example:
        >>> result = quick_analysis(
        ...     "protein.pdb",
        ...     "traj.xtc",
        ...     "rg",
        ...     selection="protein"
        ... )
        >>> print(result["results"][0]["out"])
    """
    tool = WarpMDTool()
    analyses = [{"name": analysis_name, **analysis_params}]
    result = tool._run(
        topology=topology,
        trajectory=trajectory,
        analyses=json.dumps(analyses),
    )

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {"raw_output": result}


if __name__ == "__main__":
    # Quick test
    print("warp-md LangChain Tool")
    print("=" * 50)
    print(f"Tool name: {WarpMDTool.name}")
    print(f"Schema: {WarpMDInput.schema()}")
