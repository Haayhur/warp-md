"""
warp-md LangChain Agent Example

This example demonstrates how to create a LangChain agent that uses
warp-md to perform molecular dynamics analysis tasks.

Prerequisites:
    pip install langchain langchain-core langchain-openai python-dotenv

Setup:
    1. Copy this file to your project directory
    2. Set OPENAI_API_KEY environment variable or create .env file
    3. Ensure warp-md is installed: pip install warp-md
    4. Have topology and trajectory files ready

Usage:
    python example_agent.py

The agent will:
    1. Load your molecular system
    2. Perform multiple analyses (Rg, RMSD, RDF)
    3. Return structured results with file artifacts
"""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub

# Import the warp-md tool
import sys
sys.path.insert(0, str(Path(__file__).parent))
from warp_md_tool import WarpMDTool, create_warp_md_tool

# Load environment variables
load_dotenv()


def create_md_analysis_agent(
    model: str = "gpt-4o",
    temperature: float = 0,
    verbose: bool = True,
) -> AgentExecutor:
    """
    Create a LangChain agent for molecular dynamics analysis.

    Args:
        model: OpenAI model name
        temperature: Sampling temperature
        verbose: Enable verbose logging

    Returns:
        Configured AgentExecutor

    Example:
        >>> agent = create_md_analysis_agent()
        >>> result = agent.invoke({
        ...     "input": "Analyze the radius of gyration for the protein"
        ... })
    """
    # Initialize LLM
    llm = ChatOpenAI(model=model, temperature=temperature)

    # Create tools
    tools = [create_warp_md_tool()]

    # Get prompt from hub (or create custom)
    prompt = hub.pull("hwchase17/openai-tools")

    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Create executor
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=5,
    )

    return executor


def run_example_analysis(
    topology: str,
    trajectory: str,
    queries: List[str],
) -> None:
    """
    Run a set of example analysis queries.

    Args:
        topology: Path to topology file
        trajectory: Path to trajectory file
        queries: List of natural language queries to execute

    Example:
        >>> run_example_analysis(
        ...     "protein.pdb",
        ...     "traj.xtc",
        ...     [
        ...         "Calculate the radius of gyration for the protein",
        ...         "Compute RMSD relative to the first frame",
        ...         "Calculate the water-water radial distribution function"
        ...     ]
        ... )
    """
    # Create agent with file context
    agent = create_md_analysis_agent()

    # Add file context to each query
    system_context = f"""
You have access to these molecular dynamics files:
- Topology: {topology}
- Trajectory: {trajectory}

When performing analyses, always specify these file paths exactly.
Available analyses include: rg, rmsd, msd, rdf, conductivity, hbond, and many more.
"""

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print(f"{'='*60}")

        try:
            result = agent.invoke({
                "input": system_context + "\n\n" + query,
            })

            print(f"\nResult: {result.get('output', 'No output')}")

        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Main entry point for the example agent."""
    import argparse

    parser = argparse.ArgumentParser(
        description="warp-md LangChain Agent Example"
    )
    parser.add_argument(
        "--topology", "-t",
        required=True,
        help="Path to topology file (PDB/GRO/PDBQT)",
    )
    parser.add_argument(
        "--trajectory", "-tr",
        required=True,
        help="Path to trajectory file (DCD/XTC/PDBQT)",
    )
    parser.add_argument(
        "--query", "-q",
        action="append",
        help="Analysis query (can be specified multiple times)",
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4o",
        help="OpenAI model to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="results/langchain_agent",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Default queries if none provided
    if not args.query:
        args.query = [
            "Calculate the radius of gyration for the protein backbone",
            "Compute RMSD for the protein, aligning to the backbone",
            "Calculate the water oxygen-oxygen radial distribution function",
        ]

    # Run analyses
    run_example_analysis(
        topology=args.topology,
        trajectory=args.trajectory,
        queries=args.query,
    )


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set")
        print("Set it via: export OPENAI_API_KEY='your-key'")
        print("Or create a .env file with: OPENAI_API_KEY=your-key")
        exit(1)

    main()
