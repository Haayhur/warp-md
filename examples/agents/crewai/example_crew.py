"""
warp-md CrewAI Crew Example

This example demonstrates a multi-agent crew for molecular dynamics
analysis using warp-md tools.

Prerequisites:
    pip install crewai crewai-tools warp-md python-dotenv

Usage:
    python example_crew.py --topology protein.pdb --trajectory traj.xtc

The crew consists of:
    1. MD Analyst: Performs trajectory analyses
    2. Data Interpreter: Interprets and summarizes results
    3. Report Writer: Compiles findings into a report
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM

import sys
sys.path.insert(0, str(Path(__file__).parent))
from warp_md_tool import WarpMDAnalysisTool, QuickWarpMDTool

load_dotenv()


def create_md_analyst(
    model: str = "gpt-4o",
    verbose: bool = True,
) -> Agent:
    """
    Create an agent specialized in MD trajectory analysis.

    Args:
        model: LLM model name
        verbose: Enable verbose output

    Returns:
        Configured CrewAI agent

    Example:
        >>> analyst = create_md_analyst()
    """
    llm = LLM(model=model, temperature=0)

    return Agent(
        role="Molecular Dynamics Analyst",
        goal="Execute accurate trajectory analyses to extract structural and dynamic properties",
        backstory="""You are a computational chemist with 15 years of experience
        in molecular dynamics simulations. You understand protein structure,
        statistical mechanics, and the proper application of analysis methods.
        You carefully select appropriate atom selections and analysis parameters.""",
        tools=[WarpMDAnalysisTool(), QuickWarpMDTool()],
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
    )


def create_data_interpreter(
    model: str = "gpt-4o",
    verbose: bool = True,
) -> Agent:
    """
    Create an agent for interpreting analysis results.

    Args:
        model: LLM model name
        verbose: Enable verbose output

    Returns:
        Configured CrewAI agent
    """
    llm = LLM(model=model, temperature=0)

    return Agent(
        role="Data Interpreter",
        goal="Analyze trajectory results and extract meaningful scientific insights",
        backstory="""You are a biophysicist specializing in interpreting
        molecular dynamics data. You understand the physical meaning of
        radius of gyration, RMSD, diffusion coefficients, and distribution
        functions. You can identify patterns and anomalies in trajectory data.""",
        tools=[],  # No tools - analyzes text data
        llm=llm,
        verbose=verbose,
    )


def create_report_writer(
    model: str = "gpt-4o",
    verbose: bool = True,
) -> Agent:
    """
    Create an agent for compiling analysis reports.

    Args:
        model: LLM model name
        verbose: Enable verbose output

    Returns:
        Configured CrewAI agent
    """
    llm = LLM(model=model, temperature=0.3)

    return Agent(
        role="Scientific Report Writer",
        goal="Compile findings into clear, well-structured scientific summaries",
        backstory="""You are a scientific writer with expertise in
        computational chemistry. You can synthesize complex analysis
        results into clear, concise summaries that highlight key findings
        and their scientific implications.""",
        tools=[],
        llm=llm,
        verbose=verbose,
    )


def create_md_analysis_crew(
    model: str = "gpt-4o",
    verbose: bool = True,
) -> Crew:
    """
    Create a complete crew for MD analysis workflows.

    Args:
        model: LLM model name for all agents
        verbose: Enable verbose output

    Returns:
        Configured CrewAI crew

    Example:
        >>> crew = create_md_analysis_crew()
        >>> result = crew.kickoff(inputs={"topology": "p.pdb", "trajectory": "t.xtc"})
    """
    # Create agents
    analyst = create_md_analyst(model=model, verbose=verbose)
    interpreter = create_data_interpreter(model=model, verbose=verbose)
    writer = create_report_writer(model=model, verbose=verbose)

    return Crew(
        agents=[analyst, interpreter, writer],
        process=Process.sequential,
        verbose=verbose,
        memory=True,
    )


def run_comprehensive_analysis(
    topology: str,
    trajectory: str,
    model: str = "gpt-4o",
    output_dir: str = "results/crewai_analysis",
) -> str:
    """
    Run a comprehensive MD analysis with a multi-agent crew.

    This workflow:
    1. Analyzes protein structure (Rg, RMSD, end-to-end)
    2. Examines solvent structure (RDF)
    3. Interprets the results
    4. Compiles a summary report

    Args:
        topology: Path to topology file
        trajectory: Path to trajectory file
        model: LLM model name
        output_dir: Output directory

    Returns:
        Final crew output

    Example:
        >>> result = run_comprehensive_analysis("p.pdb", "t.xtc")
        >>> print(result)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create crew
    crew = create_md_analysis_crew(model=model)

    # Define analysis tasks
    analysis_task = Task(
        description=f"""
        Perform a comprehensive structural analysis of the protein trajectory.

        File paths:
        - Topology: {topology}
        - Trajectory: {trajectory}

        Required analyses:
        1. Radius of gyration (rg) for the protein backbone
        2. RMSD for protein atoms, aligned to backbone
        3. End-to-end distance for the protein
        4. Water oxygen-oxygen radial distribution function (rdf)

        Set output directory to: {output_dir}

        Return the paths to all generated result files.
        """,
        expected_output="Paths to result files (npz format) for each analysis",
        agent=crew.agents[0],  # MD Analyst
    )

    interpretation_task = Task(
        description="""
        Interpret the trajectory analysis results provided by the analyst.

        Examine the result file paths and provide:
        1. Summary of structural properties (Rg, RMSD trends)
        2. Assessment of protein stability
        3. Solvent structure observations from RDF
        4. Any notable patterns or anomalies

        Be specific about what the values indicate about the system.
        """,
        expected_output="Scientific interpretation of the analysis results",
        agent=crew.agents[1],  # Data Interpreter
    )

    report_task = Task(
        description="""
        Compile the analysis and interpretation into a concise report.

        Include:
        1. System information (file paths, analyses performed)
        2. Key findings from the interpretation
        3. Scientific conclusions

        Format as a clear markdown-style summary.
        """,
        expected_output="Final analysis report in markdown format",
        agent=crew.agents[2],  # Report Writer
    )

    # Execute crew
    result = crew.kickoff(
        inputs={
            "topology": topology,
            "trajectory": trajectory,
        }
    )

    return result


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="warp-md CrewAI Crew Example"
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
        "--model", "-m",
        default="gpt-4o",
        help="OpenAI model (default: gpt-4o)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="results/crewai_analysis",
        help="Output directory",
    )

    args = parser.parse_args()

    # Run analysis
    print("Starting warp-md CrewAI Analysis Crew")
    print("=" * 60)

    result = run_comprehensive_analysis(
        topology=args.topology,
        trajectory=args.trajectory,
        model=args.model,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("Crew Analysis Complete")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
