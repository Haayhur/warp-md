"""MCP (Model Context Protocol) server for warp-md.

This enables AI agents like Claude to directly call warp-md tools.

Usage:
    warp-md mcp  # starts stdio MCP server
    
Configuration (claude_desktop_config.json):
    {
        "mcpServers": {
            "warp-md": {
                "command": "warp-md",
                "args": ["mcp"]
            }
        }
    }
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None  # type: ignore

from .agent_schema import AnalysisName, _ANALYSIS_REQUIRED_FIELDS


def _get_mcp() -> "FastMCP":
    """Get or create the MCP server instance."""
    if FastMCP is None:
        raise ImportError(
            "MCP SDK not installed. Install with: pip install mcp"
        )
    return FastMCP("warp-md", dependencies=["warp-md"])


mcp = None


def _ensure_mcp():
    global mcp
    if mcp is None:
        mcp = _get_mcp()
    return mcp


def _consume_stderr_events(
    proc: Any,
    completion_event: str,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    """Read stderr NDJSON stream and retain plain-text diagnostics."""
    import json

    final_event: Optional[Dict[str, Any]] = None
    error_event: Optional[Dict[str, Any]] = None
    diagnostics: List[str] = []

    if proc.stderr is None:
        return None, None, ""

    while True:
        line = proc.stderr.readline()
        if not line:
            break
        text = line.strip()
        if not text:
            continue
        try:
            event = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            diagnostics.append(text)
            continue

        event_name = event.get("event")
        if event_name == completion_event:
            final_event = event
        elif event_name == "error":
            error_event = event
        else:
            diagnostics.append(text)

    return final_event, error_event, "\n".join(diagnostics).strip()


def register_tools():
    """Register all warp-md tools with the MCP server."""
    server = _ensure_mcp()
    
    @server.tool()
    def run_analysis(
        system_path: str,
        trajectory_path: str,
        analyses: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
        device: str = "auto",
        fail_fast: bool = True,
    ) -> Dict[str, Any]:
        """Run molecular dynamics analyses on a trajectory.
        
        Args:
            system_path: Path to topology/structure file (PDB, PSF, etc.)
            trajectory_path: Path to trajectory file (XTC, DCD, etc.)
            analyses: List of analysis specifications, each with:
                - name: Analysis type (e.g., "rg", "rmsd", "dssp")
                - Additional parameters specific to the analysis
            output_dir: Directory for output files (default: temp dir)
            device: Compute device ("auto", "cpu", "cuda")
            fail_fast: Stop on first error (default True)
        
        Returns:
            Result envelope with status, timing, and analysis results
        
        Example:
            run_analysis(
                system_path="structure.pdb",
                trajectory_path="trajectory.xtc",
                analyses=[
                    {"name": "rg", "selection": "protein"},
                    {"name": "rmsd", "selection": "backbone"},
                ]
            )
        """
        from .runner import run_analyses
        
        result = run_analyses(
            {
                "version": "warp-md.agent.v1",
                "system": {"path": system_path},
                "trajectory": {"path": trajectory_path},
                "analyses": analyses,
                "fail_fast": fail_fast,
            },
            output_dir=output_dir,
            device=device,
        )
        return result.model_dump(mode="json")

    @server.tool()
    def list_analyses() -> List[str]:
        """List all available analysis types.
        
        Returns:
            List of analysis names that can be used in run_analysis.
            
        Example analyses:
            - rg: Radius of gyration
            - rmsd: Root mean square deviation
            - msd: Mean square displacement
            - rdf: Radial distribution function
            - dssp: Secondary structure
            - diffusion: Diffusion coefficient
            - pca: Principal component analysis
            - rmsf: Root mean square fluctuation
            - density: Spatial density
            - native_contacts: Native contact fraction
            - docking: Molecular docking interactions
        """
        return list(AnalysisName.__args__)

    @server.tool()
    def get_analysis_schema(name: str) -> Dict[str, Any]:
        """Get the parameter schema for a specific analysis.
        
        Args:
            name: Analysis name (e.g., "rg", "rmsd", "dssp")
            
        Returns:
            Schema with required and optional parameters.
            
        Example:
            get_analysis_schema("rg")
            # Returns: {"name": "rg", "required": ["selection"], "optional": [...]}
        """
        name_normalized = name.strip().replace("-", "_")
        if name_normalized not in AnalysisName.__args__:
            return {
                "error": f"Unknown analysis: {name}",
                "available": list(AnalysisName.__args__)[:10],
            }
        
        required = _ANALYSIS_REQUIRED_FIELDS.get(name_normalized, ())
        
        # Common optional fields
        optional = ["out", "device", "chunk_frames"]
        
        return {
            "name": name_normalized,
            "required": list(required),
            "optional": optional,
        }

    @server.tool()
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an analysis configuration without running it.

        Args:
            config: Configuration matching warp-md.agent.v1 schema

        Returns:
            Validation result with status and any errors.

        Example:
            validate_config({
                "version": "warp-md.agent.v1",
                "system": {"path": "structure.pdb"},
                "trajectory": {"path": "traj.xtc"},
                "analyses": [{"name": "rg", "selection": "protein"}]
            })
        """
        from .agent_schema import validate_run_request
        from pydantic import ValidationError

        try:
            cfg = validate_run_request(config)
            return {
                "valid": True,
                "analysis_count": len(cfg.analyses),
                "analyses": [a.name for a in cfg.analyses],
            }
        except ValidationError as exc:
            return {
                "valid": False,
                "errors": exc.errors(),
            }
        except Exception as exc:
            return {
                "valid": False,
                "errors": [{"msg": str(exc)}],
            }

    # ============ warp-pack tools ============

    @server.tool()
    def pack_molecules(
        config_path: str,
        output: str,
        format: str = "pdb",
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Pack molecules into a simulation box using warp-pack.

        Efficiently places molecules into a simulation box, supporting
        various constraints (box, sphere) and output formats.

        Args:
            config_path: Path to packing configuration (YAML, JSON, or Packmol INP)
            output: Output file path for the packed structure
            format: Output format (pdb, gro, mol2, crd, lammps, pdbx, xyz)
            stream: Enable NDJSON streaming progress events

        Returns:
            Result with total atoms, molecules, timing, and output path

        Example:
            pack_molecules(
                config_path="pack.yaml",
                output="packed.pdb",
                format="pdb",
                stream=True
            )
        """
        import subprocess
        from pathlib import Path

        cmd = [
            "warp-pack",
            "--config", config_path,
            "--output", output,
            "--format", format,
        ]
        if stream:
            cmd.append("--stream")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        final_event, error_event, diagnostics = _consume_stderr_events(
            proc, "pack_complete"
        )

        returncode = proc.wait()

        if error_event:
            return {
                "success": False,
                "error": error_event.get("message", diagnostics or "Unknown error"),
                "code": error_event.get("code"),
            }

        if returncode != 0:
            return {
                "success": False,
                "error": diagnostics or "Packing failed",
            }

        if final_event:
            return {
                "success": True,
                "total_atoms": final_event.get("total_atoms", 0),
                "total_molecules": final_event.get("total_molecules", 0),
                "final_box_size": final_event.get("final_box_size"),
                "output_path": output,
                "elapsed_ms": final_event.get("elapsed_ms", 0),
            }

        # Fallback: check output file
        if Path(output).exists():
            return {
                "success": True,
                "output_path": output,
                "total_atoms": 0,  # Would need to parse file
            }

        return {
            "success": True,
            "output_path": output,
        }

    # ============ warp-pep tools ============

    @server.tool()
    def build_peptide(
        sequence: Optional[str] = None,
        three_letter: Optional[str] = None,
        output: str = "peptide.pdb",
        format: str = "pdb",
        preset: str = "extended",
        oxt: bool = False,
        detect_ss: bool = False,
    ) -> Dict[str, Any]:
        """Build a peptide structure from amino acid sequence.

        Supports all 20 standard amino acids plus Amber force field variants
        (CYX, HID, HIE, HIP, ASH, GLH, LYN) and D-amino acids.

        Args:
            sequence: One-letter amino acid sequence (e.g., "ACDEFG")
            three_letter: Dash-separated three-letter codes (e.g., "ALA-CYS-GLU")
            output: Output file path
            format: Output format (pdb, pdbx, xyz, gro, mol2, crd, lammps)
            preset: Ramachandran preset (extended, alpha-helix, beta-sheet, polyproline)
            oxt: Add terminal OXT oxygen
            detect_ss: Detect disulfide bonds (CYS pairs within 2.5 Å)

        Returns:
            Result with total atoms, residues, chains, and output path

        Example:
            build_peptide(
                sequence="ACDEFG",
                preset="alpha-helix",
                output="helix.pdb"
            )
        """
        if not sequence and not three_letter:
            return {
                "success": False,
                "error": "Must provide sequence or three_letter",
            }

        cmd = ["warp-pep", "build", "--stream"]

        if sequence:
            cmd.extend(["--sequence", sequence])
        elif three_letter:
            cmd.extend(["--three-letter", three_letter])

        cmd.extend([
            "--output", output,
            "--format", format,
            "--preset", preset,
        ])

        if oxt:
            cmd.append("--oxt")
        if detect_ss:
            cmd.append("--detect-ss")

        return _run_pep_command(cmd, output)

    @server.tool()
    def mutate_peptide(
        input: str,
        mutations: List[str],
        output: str = "mutated.pdb",
        format: str = "pdb",
        oxt: bool = False,
        detect_ss: bool = False,
    ) -> Dict[str, Any]:
        """Mutate residue(s) in a peptide structure.

        Args:
            input: Input structure file path
            mutations: List of mutation specs (e.g., ["A5G", "L10W"])
            output: Output file path
            format: Output format
            oxt: Add terminal OXT oxygen
            detect_ss: Detect disulfide bonds after mutation

        Returns:
            Result with total atoms and output path

        Example:
            mutate_peptide(
                input="peptide.pdb",
                mutations=["A5G", "L10W"],
                output="mutated.pdb"
            )
        """
        if not mutations:
            return {
                "success": False,
                "error": "Must provide at least one mutation",
            }

        cmd = [
            "warp-pep", "mutate",
            "--input", input,
            "--mutations", ",".join(mutations),
            "--output", output,
            "--format", format,
            "--stream",
        ]

        if oxt:
            cmd.append("--oxt")
        if detect_ss:
            cmd.append("--detect-ss")

        return _run_pep_command(cmd, output)

    return server


def _run_pep_command(cmd: List[str], output: str) -> Dict[str, Any]:
    """Helper to run a warp-pep command and parse results."""
    import subprocess
    from pathlib import Path

    if "--stream" not in cmd:
        cmd = [*cmd, "--stream"]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    final_event, error_event, diagnostics = _consume_stderr_events(
        proc, "operation_complete"
    )

    returncode = proc.wait()

    if error_event:
        return {
            "success": False,
            "error": error_event.get("message", diagnostics or "Unknown error"),
        }

    if returncode != 0:
        return {
            "success": False,
            "error": diagnostics or "Operation failed",
        }

    if final_event:
        return {
            "success": True,
            "total_atoms": final_event.get("total_atoms", 0),
            "total_residues": final_event.get("total_residues", 0),
            "total_chains": final_event.get("total_chains", 0),
            "output_path": output,
            "elapsed_ms": final_event.get("elapsed_ms", 0),
        }

    # Fallback: check output file
    if Path(output).exists():
        return {
            "success": True,
            "output_path": output,
            "total_atoms": 0,  # Would need to parse file
        }

    return {
        "success": True,
        "output_path": output,
    }


def main():
    """Run the MCP server (stdio transport)."""
    server = register_tools()
    server.run()


if __name__ == "__main__":
    main()
