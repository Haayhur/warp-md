"""
warp-md AutoGen Integration

This module provides AutoGen tools for warp-md, enabling multi-agent
conversations for molecular dynamics analysis.

Installation:
    pip install pyautogen warp-md

Example:
    from autogen import AssistantAgent, UserProxyAgent
    from warp_md_autogen_tool import warp_md_function

    # Create agents
    assistant = AssistantAgent(
        name="analyst",
        llm_config={"config_list": [...]},
    )

    user_proxy = UserProxyAgent(
        name="user",
        function_map={"warp_md_analysis": warp_md_function},
    )

    # Start conversation
    user_proxy.initiate_chat(
        assistant,
        message="Calculate Rg for the protein"
    )
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def warp_md_function(
    topology: str,
    trajectory: str,
    analyses: str | list[dict],
    output_dir: str = ".",
    device: str = "auto",
) -> str:
    """
    AutoGen-compatible function for warp-md analysis.

    This function can be registered in an AutoGen agent's function_map
    to enable tool calling during agent conversations.

    Args:
        topology: Path to topology file
        trajectory: Path to trajectory file
        analyses: JSON string or list of analysis specs
        output_dir: Output directory
        device: Compute device

    Returns:
        JSON result envelope as string

    Example:
        >>> result = warp_md_function(
        ...     topology="protein.pdb",
        ...     trajectory="traj.xtc",
        ...     analyses='[{"name": "rg", "selection": "protein"}]'
        ... )
    """
    # Parse analyses
    if isinstance(analyses, str):
        try:
            analyses_list = json.loads(analyses)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON: {e}"})
    else:
        analyses_list = analyses

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
    config_path = Path(output_dir) / "_warp_md_autogen_request.json"
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


# AutoGen function schema for registration
WARP_MD_FUNCTION_SCHEMA = {
    "name": "warp_md_analysis",
    "description": """
    Perform molecular dynamics trajectory analysis using warp-md.

    Supports 50+ analyses: rg, rmsd, msd, rdf, conductivity, hbond, etc.
    Returns JSON envelope with result file paths.

    Input:
    - topology: Path to topology file (PDB/GRO/PDBQT)
    - trajectory: Path to trajectory file (DCD/XTC/PDBQT)
    - analyses: JSON string of analysis specifications
    - output_dir: Output directory (default: .)
    - device: Compute device (default: auto)
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "topology": {"type": "string"},
            "trajectory": {"type": "string"},
            "analyses": {"type": "string"},
            "output_dir": {"type": "string"},
            "device": {"type": "string"},
        },
        "required": ["topology", "trajectory", "analyses"],
    },
}


def create_warp_md_tool_decorator(
    topology: str,
    trajectory: str,
    output_dir: str = ".",
) -> callable:
    """
    Create a pre-configured warp-md function for AutoGen.

    This factory function creates a partially-applied version of
    warp_md_function with fixed file paths.

    Args:
        topology: Path to topology file
        trajectory: Path to trajectory file
        output_dir: Output directory

    Returns:
        Configured function for AutoGen

    Example:
        >>> md_tool = create_warp_md_tool_decorator("p.pdb", "t.xtc")
        >>> result = md_tool(analyses='[{"name": "rg", "selection": "protein"}]')
    """

    def configured_warp_md(
        analyses: str | list[dict],
        device: str = "auto",
    ) -> str:
        return warp_md_function(
            topology=topology,
            trajectory=trajectory,
            analyses=analyses,
            output_dir=output_dir,
            device=device,
        )

    configured_warp_md.__name__ = "warp_md_analysis"
    configured_warp_md.__doc__ = f"""
    Analyze trajectory: {trajectory} with topology {topology}

    Args:
        analyses: JSON string or list of analysis specs
        device: Compute device (default: auto)
    """

    return configured_warp_md


if __name__ == "__main__":
    print("warp-md AutoGen Integration")
    print("=" * 50)
    print(f"Function: {warp_md_function.__name__}")
    print(f"Schema:\n{json.dumps(WARP_MD_FUNCTION_SCHEMA, indent=2)}")
