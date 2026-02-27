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

    return server


def main():
    """Run the MCP server (stdio transport)."""
    server = register_tools()
    server.run()


if __name__ == "__main__":
    main()
