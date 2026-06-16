from __future__ import annotations

import argparse

from .cli_analysis_registry import ANALYSIS_REGISTRY


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--topology", required=True, help="Topology file (.pdb or .gro)")
    parser.add_argument("--traj", required=True, help="Trajectory file (.dcd, .xtc, or .trr)")
    parser.add_argument(
        "--topology-format",
        choices=["pdb", "gro"],
        help="Override topology format",
    )
    parser.add_argument(
        "--traj-format",
        choices=["dcd", "xtc", "trr"],
        help="Override trajectory format",
    )
    parser.add_argument(
        "--traj-length-scale",
        type=float,
        help="DCD length scale (e.g., 10.0 for nm->A)",
    )
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:0")
    parser.add_argument("--chunk-frames", type=int, help="Frames per chunk")
    parser.add_argument("--out", help="Output path (.npz/.npy/.csv/.json)")
    summary_group = parser.add_mutually_exclusive_group()
    summary_group.add_argument(
        "--print-summary",
        dest="print_summary",
        action="store_true",
        help="Deprecated: single-analysis commands now always emit JSON envelopes",
    )
    summary_group.add_argument(
        "--no-summary",
        dest="print_summary",
        action="store_false",
        help="Deprecated: single-analysis commands now always emit JSON envelopes",
    )
    parser.set_defaults(print_summary=True)
    parser.add_argument(
        "--summary-format",
        choices=["json", "text"],
        default="json",
        help="Deprecated: summary format is ignored (JSON envelope always emitted)",
    )
    parser.add_argument(
        "--debug-errors",
        action="store_true",
        help="Include traceback in JSON error envelopes",
    )


def add_dynamics_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--frame-decimation",
        help="start,stride (e.g., 0,10)",
    )
    parser.add_argument(
        "--dt-decimation",
        help="cut1,stride1,cut2,stride2",
    )
    parser.add_argument(
        "--time-binning",
        help="eps_num,eps_add",
    )
    parser.add_argument(
        "--lag-mode",
        choices=["auto", "multi_tau", "ring", "fft"],
        help="Lag mode (auto/multi_tau/ring/fft)",
    )
    parser.add_argument("--max-lag", type=int, help="Max lag (ring mode)")
    parser.add_argument("--memory-budget-bytes", type=int, help="Memory budget")
    parser.add_argument("--multi-tau-m", type=int, help="Multi-tau m")
    parser.add_argument("--multi-tau-levels", type=int, help="Multi-tau levels")


def add_group_types_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--group-types",
        help=(
            "JSON list or selections:<sel1,sel2>. "
            "Example: --group-types '[0,1,1]' or --group-types 'selections:resname NA,resname CL'"
        ),
    )


def setup_rg_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Selection string")
    parser.add_argument("--mass-weighted", action="store_true", help="Mass-weighted Rg")


def setup_rmsd_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Selection string")
    parser.add_argument(
        "--reference",
        choices=["topology", "frame0"],
        default="topology",
        help="Reference frame",
    )
    parser.add_argument(
        "--align",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align before RMSD",
    )


def setup_msd_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Selection string")
    parser.add_argument(
        "--group-by",
        choices=["resid", "chain", "resid_chain"],
        default="resid",
        help="Group-by mode",
    )
    parser.add_argument("--axis", help="x,y,z axis components")
    parser.add_argument("--length-scale", type=float, help="Length scale")
    add_group_types_args(parser)
    add_dynamics_args(parser)


def setup_rotacf_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Selection string")
    parser.add_argument(
        "--group-by",
        choices=["resid", "chain", "resid_chain"],
        default="resid",
        help="Group-by mode",
    )
    parser.add_argument("--orientation", required=True, help="Indices (2 or 3) within group")
    parser.add_argument(
        "--p2-legendre",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use P2 Legendre",
    )
    parser.add_argument("--length-scale", type=float, help="Length scale")
    add_group_types_args(parser)
    add_dynamics_args(parser)


def setup_conductivity_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Selection string")
    parser.add_argument(
        "--charges",
        required=True,
        help=(
            "Charges: JSON list, table:path, or selections:[{selection,charge},...]"
        ),
    )
    parser.add_argument("--temperature", type=float, required=True, help="Temperature (K)")
    parser.add_argument(
        "--group-by",
        choices=["resid", "chain", "resid_chain"],
        default="resid",
        help="Group-by mode",
    )
    parser.add_argument(
        "--transference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute transference matrix",
    )
    parser.add_argument("--length-scale", type=float, help="Length scale")
    add_group_types_args(parser)
    add_dynamics_args(parser)


def setup_dielectric_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Selection string")
    parser.add_argument(
        "--charges",
        required=True,
        help="Charges: JSON list, table:path, or selections:[{selection,charge},...]",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature (K) for dielectric fluctuation scaling",
    )
    parser.add_argument(
        "--group-by",
        choices=["resid", "chain", "resid_chain"],
        default="resid",
        help="Group-by mode",
    )
    parser.add_argument(
        "--make-whole",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reconstruct grouped molecules across periodic boundaries before dipole evaluation",
    )
    parser.add_argument("--length-scale", type=float, help="Length scale")
    add_group_types_args(parser)


def setup_dipole_alignment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Selection string")
    parser.add_argument(
        "--charges",
        required=True,
        help="Charges: JSON list, table:path, or selections:[{selection,charge},...]",
    )
    parser.add_argument(
        "--group-by",
        choices=["resid", "chain", "resid_chain"],
        default="resid",
        help="Group-by mode",
    )
    parser.add_argument("--length-scale", type=float, help="Length scale")
    add_group_types_args(parser)


def setup_ion_pair_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Selection string")
    parser.add_argument("--rclust-cat", type=float, required=True, help="Cation cutoff")
    parser.add_argument("--rclust-ani", type=float, required=True, help="Anion cutoff")
    parser.add_argument(
        "--group-by",
        choices=["resid", "chain", "resid_chain"],
        default="resid",
        help="Group-by mode",
    )
    parser.add_argument("--cation-type", type=int, default=0, help="Cation type index")
    parser.add_argument("--anion-type", type=int, default=1, help="Anion type index")
    parser.add_argument("--max-cluster", type=int, default=10, help="Max cluster size")
    parser.add_argument("--length-scale", type=float, help="Length scale")
    add_group_types_args(parser)
    add_dynamics_args(parser)


def setup_structure_factor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Selection string")
    parser.add_argument("--bins", type=int, required=True, help="r-space bins")
    parser.add_argument("--r-max", type=float, required=True, help="r-space max (A)")
    parser.add_argument("--q-bins", type=int, required=True, help="q-space bins")
    parser.add_argument("--q-max", type=float, required=True, help="q-space max (1/A)")
    parser.add_argument(
        "--pbc",
        choices=["orthorhombic", "none"],
        default="orthorhombic",
        help="PBC mode",
    )
    parser.add_argument("--length-scale", type=float, help="Length scale")


def setup_water_count_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--water-selection", required=True, help="Water selection")
    parser.add_argument("--center-selection", required=True, help="Center selection")
    parser.add_argument("--box-unit", required=True, help="Box unit (x,y,z)")
    parser.add_argument("--region-size", required=True, help="Region size (x,y,z)")
    parser.add_argument("--shift", help="Shift (x,y,z)")
    parser.add_argument("--length-scale", type=float, help="Length scale")


def setup_free_volume_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Selection to mark occupied space")
    parser.add_argument("--center-selection", required=True, help="Center selection")
    parser.add_argument("--box-unit", help="Voxel size (x,y,z) - defaults to 1.0,1.0,1.0 if not specified")
    parser.add_argument("--region-size", help="Region size (x,y,z) - auto-detected from selection bounding box if not specified")
    parser.add_argument("--probe-radius", type=float, help="Probe radius in coordinate units")
    parser.add_argument("--shift", help="Shift (x,y,z)")
    parser.add_argument("--length-scale", type=float, help="Length scale")


def setup_hydrophobic_defects_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lipid-selection", required=True, help="Membrane/lipid atoms that occupy space")
    parser.add_argument("--reference-selection", required=True, help="Reference atoms defining defect columns")
    parser.add_argument("--midplane-selection", help="Selection used to define the leaflet midplane")
    parser.add_argument(
        "--leaflet",
        choices=["both", "upper", "lower"],
        default="both",
        help="Automatically filter lipid/reference atoms to one leaflet",
    )
    parser.add_argument("--leaflet-bins", type=int, default=1, help="XY bins per dimension for local leaflet midplanes")
    parser.add_argument("--voxel-size", type=float, default=1.0, help="Voxel size in coordinate units")
    parser.add_argument("--z-bounds", help="Lower,upper z bounds for the defect slab")
    parser.add_argument("--probe-radius", type=float, help="Water probe radius in coordinate units")
    parser.add_argument("--defect-radius", type=float, help="XY search radius around reference atoms")
    parser.add_argument(
        "--grid-mode",
        choices=["voxel_centers", "lattice_nodes"],
        default="voxel_centers",
        help="Voxel-center grid or reference-compatible lattice-node grid",
    )
    parser.add_argument("--length-scale", type=float, help="Length scale applied before analysis")


def setup_bondi_ffv_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Selection used for Bondi van der Waals volume")
    parser.add_argument("--bondi-scale", type=float, help="Homogeneity scale factor applied to Bondi vdW volume (default 1.3)")
    parser.add_argument("--probe-radius", type=float, help="Probe radius in Angstroms (default 0.0)")
    parser.add_argument("--seed", type=int, help="Deterministic Monte Carlo seed")
    parser.add_argument("--ninsert-per-nm3", type=int, help="Probe insertions per cubic nm (default 1000)")
    parser.add_argument("--length-scale", type=float, help="Length scale")


def setup_equipartition_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Selection string")
    parser.add_argument(
        "--group-by",
        choices=["resid", "chain", "resid_chain"],
        default="resid",
        help="Group-by mode",
    )
    parser.add_argument("--velocity-scale", type=float, help="Velocity scale")
    parser.add_argument("--length-scale", type=float, help="Length scale")
    add_group_types_args(parser)


def setup_hbond_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--donors", required=True, help="Donor selection")
    parser.add_argument("--acceptors", required=True, help="Acceptor selection")
    parser.add_argument("--dist-cutoff", type=float, required=True, help="Distance cutoff (A)")
    parser.add_argument("--hydrogens", help="Hydrogen selection")
    parser.add_argument("--angle-cutoff", type=float, help="Angle cutoff (deg)")


def setup_rdf_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sel-a", required=True, help="Selection A")
    parser.add_argument("--sel-b", required=True, help="Selection B")
    parser.add_argument("--bins", type=int, required=True, help="Number of bins")
    parser.add_argument("--r-max", type=float, required=True, help="Max distance (A)")
    parser.add_argument(
        "--pbc",
        choices=["orthorhombic", "none"],
        default="orthorhombic",
        help="PBC mode",
    )


def setup_end_to_end_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Selection string")


def setup_contour_length_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Selection string")


def setup_chain_rg_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Selection string")


def setup_bond_length_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Selection string")
    parser.add_argument("--bins", type=int, required=True, help="Number of bins")
    parser.add_argument("--r-max", type=float, required=True, help="Max distance (A)")


def setup_bond_angle_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Selection string")
    parser.add_argument("--bins", type=int, required=True, help="Number of bins")
    parser.add_argument(
        "--degrees",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return degrees (default true)",
    )


def setup_persistence_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Selection string")


def add_lipid_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--length-scale", type=float, help="Coordinate length scale")
    parser.add_argument("--frame-indices", help="Comma-separated frame indices")


def add_lipid_leaflet_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--midplane-selection", help="Selection used to define membrane midplane")
    parser.add_argument("--midplane-cutoff", type=float, default=0.0, help="Midplane exclusion cutoff")
    parser.add_argument("--bins", type=int, default=1, help="XY bins per dimension")


def add_lipid_cutoff_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cutoff", type=float, default=10.0, help="Neighbour cutoff")


def add_lipid_leaflets_file_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--leaflets", required=True, help="Leaflet array path (.npy/.npz/.csv/.json)")


def setup_lipid_leaflets_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Lipid headgroup selection")
    add_lipid_leaflet_args(parser)
    add_lipid_common_args(parser)


def setup_lipid_curved_leaflets_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Lipid headgroup selection")
    parser.add_argument("--cutoff", type=float, default=15.0, help="Graph cutoff")
    parser.add_argument("--midplane-selection", help="Selection used to define membrane midplane")
    parser.add_argument("--midplane-cutoff", type=float, default=0.0, help="Midplane exclusion cutoff")
    add_lipid_common_args(parser)


def setup_lipid_z_positions_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--membrane-selection", required=True, help="Membrane reference selection")
    parser.add_argument("--height-selection", required=True, help="Height selection")
    parser.add_argument("--bins", type=int, default=1, help="XY bins per dimension")
    add_lipid_common_args(parser)


def setup_lipid_z_thickness_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Lipid selection")
    add_lipid_common_args(parser)


def setup_lipid_z_angles_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--atom-a", required=True, help="First atom selection")
    parser.add_argument("--atom-b", required=True, help="Second atom selection")
    parser.add_argument("--degrees", action=argparse.BooleanOptionalAction, default=True)
    add_lipid_common_args(parser)


def setup_lipid_area_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Lipid selection")
    add_lipid_leaflets_file_arg(parser)
    add_lipid_common_args(parser)


def setup_lipid_flip_flop_args(parser: argparse.ArgumentParser) -> None:
    add_lipid_leaflets_file_arg(parser)
    parser.add_argument("--residue-ids", help="Residue ids path or comma-separated ids")
    parser.add_argument("--frame-cutoff", type=int, default=1)


def setup_lipid_neighbours_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Lipid selection")
    add_lipid_cutoff_args(parser)
    add_lipid_common_args(parser)


def setup_lipid_neighbour_matrix_args(parser: argparse.ArgumentParser) -> None:
    setup_lipid_neighbours_args(parser)


def setup_lipid_largest_cluster_args(parser: argparse.ArgumentParser) -> None:
    setup_lipid_neighbours_args(parser)


def setup_lipid_membrane_thickness_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Lipid selection")
    add_lipid_leaflets_file_arg(parser)
    parser.add_argument("--bins", type=int, default=1, help="XY bins per dimension")
    add_lipid_common_args(parser)


def setup_lipid_registration_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--upper-selection", required=True, help="Upper leaflet selection")
    parser.add_argument("--lower-selection", required=True, help="Lower leaflet selection")
    add_lipid_leaflets_file_arg(parser)
    parser.add_argument("--bins", type=int, default=1, help="XY bins per dimension")
    parser.add_argument("--gaussian-sd", type=float, default=0.0)
    add_lipid_common_args(parser)


def setup_lipid_msd_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--selection", required=True, help="Lipid selection")
    parser.add_argument("--com-removal-selection", help="COM removal selection")
    add_lipid_common_args(parser)


def setup_lipid_scc_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tail-selection", required=True, help="Tail atom selection")
    parser.add_argument("--normals", help="Normal vector array path (.npy/.npz/.csv/.json)")
    add_lipid_common_args(parser)


REGISTRY = {
    entry.cli_name: globals()[entry.setup_fn]
    for entry in ANALYSIS_REGISTRY
    if entry.setup_fn is not None
}
