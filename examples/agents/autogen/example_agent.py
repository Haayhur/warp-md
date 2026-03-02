"""
warp-md AutoGen Agent Example

This example demonstrates multi-agent MD analysis using AutoGen.

Prerequisites:
    pip install pyautogen warp-md python-dotenv

Usage:
    python example_agent.py --topology protein.pdb --trajectory traj.xtc

Reference:
    https://microsoft.github.io/autogen/
"""

import os
from pathlib import Path

from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).parent))
from warp_md_tool import (
    warp_md_function,
    create_warp_md_tool_decorator,
    WARP_MD_FUNCTION_SCHEMA,
)

load_dotenv()


def create_autogen_agents(
    topology: str,
    trajectory: str,
    output_dir: str = "results/autogen_analysis",
):
    """
    Create AutoGen agents for MD analysis workflow.

    Args:
        topology: Path to topology file
        trajectory: Path to trajectory file
        output_dir: Output directory

    Returns:
        Tuple of (user_proxy, analyst, interpreter) agents

    Example:
        >>> user, analyst, interpreter = create_autogen_agents("p.pdb", "t.xtc")
        >>> user.initiate_chat(analyst, message="Analyze the protein structure")
    """
    try:
        from autogen import AssistantAgent, UserProxyAgent
    except ImportError:
        raise ImportError(
            "AutoGen not installed. Install with: pip install pyautogen"
        )

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY or AZURE_OPENAI_API_KEY"
        )

    # LLM config
    llm_config = {
        "config_list": [
            {
                "model": "gpt-4o",
                "api_key": api_key,
            }
        ],
        "temperature": 0,
    }

    # Create pre-configured warp-md tool
    md_tool = create_warp_md_tool_decorator(topology, trajectory, output_dir)

    # User proxy - executes tools
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config=False,
        function_map={
            "warp_md_analysis": md_tool,
        },
    )

    # MD Analyst - performs analyses
    analyst = AssistantAgent(
        name="md_analyst",
        system_message="""You are a molecular dynamics analyst.
        You have access to the warp_md_analysis tool to perform trajectory analyses.

        Available files:
        - Topology: {topology}
        - Trajectory: {trajectory}

        Common analyses:
        - rg: radius of gyration (requires: selection)
        - rmsd: root mean square deviation (requires: selection)
        - rdf: radial distribution function (requires: sel_a, sel_b, bins, r_max)
        - msd: mean squared displacement (requires: selection)

        Always use the tool to perform analyses, then summarize the results.
        Return the output file paths so the interpreter can examine them.""".format(
            topology=topology, trajectory=trajectory
        ),
        llm_config=llm_config,
    )

    # Data Interpreter - interprets results
    interpreter = AssistantAgent(
        name="data_interpreter",
        system_message="""You are a biophysicist who interprets MD analysis results.

        When the analyst provides result file paths or analysis outputs:
        1. Examine the numerical values
        2. Explain what they mean physically
        3. Identify trends, patterns, or anomalies
        4. Connect to relevant scientific principles

        Provide clear, scientifically accurate interpretations.
        Do not perform analyses yourself - focus on interpreting results.""",
        llm_config=llm_config,
    )

    return user_proxy, analyst, interpreter


def run_autogen_analysis(
    topology: str,
    trajectory: str,
    query: str,
    output_dir: str = "results/autogen_analysis",
):
    """
    Run an AutoGen-based MD analysis.

    Args:
        topology: Path to topology file
        trajectory: Path to trajectory file
        query: Analysis request
        output_dir: Output directory

    Example:
        >>> run_autogen_analysis(
        ...     "protein.pdb",
        ...     "traj.xtc",
        ...     "Analyze the protein structure stability"
        ... )
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create agents
    user_proxy, analyst, interpreter = create_autogen_agents(
        topology, trajectory, output_dir
    )

    # Start conversation
    print("Starting AutoGen MD Analysis")
    print("=" * 60)

    user_proxy.initiate_chat(
        analyst,
        message=f"""Please analyze the following molecular dynamics system:

{query}

After performing the analysis, work with the interpreter to understand the results.""",
        clear_history=True,
    )

    print("\n" + "=" * 60)
    print("Analysis Complete")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="warp-md AutoGen Agent Example"
    )
    parser.add_argument(
        "--topology", "-t",
        required=True,
        help="Path to topology file",
    )
    parser.add_argument(
        "--trajectory", "-tr",
        required=True,
        help="Path to trajectory file",
    )
    parser.add_argument(
        "--query", "-q",
        default="Analyze the protein structure. Calculate Rg, RMSD, and end-to-end distance.",
        help="Analysis query",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="results/autogen_analysis",
        help="Output directory",
    )

    args = parser.parse_args()

    run_autogen_analysis(
        topology=args.topology,
        trajectory=args.trajectory,
        query=args.query,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
