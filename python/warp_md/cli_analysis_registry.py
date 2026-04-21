from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisEntry:
    cli_name: str
    plan_name: str
    setup_fn: str | None
    spec_fn: str | None
    build_fn: str
    legacy_cli_aliases: tuple[str, ...] = ()


ANALYSIS_REGISTRY: tuple[AnalysisEntry, ...] = (
    AnalysisEntry("rg", "rg", "setup_rg_args", "_spec_rg", "_build_rg"),
    AnalysisEntry("rmsd", "rmsd", "setup_rmsd_args", "_spec_rmsd", "_build_rmsd"),
    AnalysisEntry("msd", "msd", "setup_msd_args", "_spec_msd", "_build_msd"),
    AnalysisEntry("rotacf", "rotacf", "setup_rotacf_args", "_spec_rotacf", "_build_rotacf"),
    AnalysisEntry(
        "conductivity",
        "conductivity",
        "setup_conductivity_args",
        "_spec_conductivity",
        "_build_conductivity",
    ),
    AnalysisEntry(
        "dielectric",
        "dielectric",
        "setup_dielectric_args",
        "_spec_dielectric",
        "_build_dielectric",
    ),
    AnalysisEntry(
        "dipole-alignment",
        "dipole_alignment",
        "setup_dipole_alignment_args",
        "_spec_dipole_alignment",
        "_build_dipole_alignment",
    ),
    AnalysisEntry(
        "ion-pair-correlation",
        "ion_pair_correlation",
        "setup_ion_pair_args",
        "_spec_ion_pair",
        "_build_ion_pair",
    ),
    AnalysisEntry(
        "structure-factor",
        "structure_factor",
        "setup_structure_factor_args",
        "_spec_structure_factor",
        "_build_structure_factor",
    ),
    AnalysisEntry(
        "water-count",
        "water_count",
        "setup_water_count_args",
        "_spec_water_count",
        "_build_water_count",
    ),
    AnalysisEntry(
        "free-volume",
        "free_volume",
        "setup_free_volume_args",
        "_spec_free_volume",
        "_build_free_volume",
    ),
    AnalysisEntry(
        "bondi-ffv",
        "bondi_ffv",
        "setup_bondi_ffv_args",
        "_spec_bondi_ffv",
        "_build_bondi_ffv",
        legacy_cli_aliases=("ffv",),
    ),
    AnalysisEntry(
        "equipartition",
        "equipartition",
        "setup_equipartition_args",
        "_spec_equipartition",
        "_build_equipartition",
    ),
    AnalysisEntry("hbond", "hbond", "setup_hbond_args", "_spec_hbond", "_build_hbond"),
    AnalysisEntry("rdf", "rdf", "setup_rdf_args", "_spec_rdf", "_build_rdf"),
    AnalysisEntry(
        "end-to-end",
        "end_to_end",
        "setup_end_to_end_args",
        "_spec_end_to_end",
        "_build_end_to_end",
    ),
    AnalysisEntry(
        "contour-length",
        "contour_length",
        "setup_contour_length_args",
        "_spec_contour_length",
        "_build_contour_length",
    ),
    AnalysisEntry(
        "chain-rg",
        "chain_rg",
        "setup_chain_rg_args",
        "_spec_chain_rg",
        "_build_chain_rg",
    ),
    AnalysisEntry(
        "bond-length-distribution",
        "bond_length_distribution",
        "setup_bond_length_args",
        "_spec_bond_length",
        "_build_bond_length",
    ),
    AnalysisEntry(
        "bond-angle-distribution",
        "bond_angle_distribution",
        "setup_bond_angle_args",
        "_spec_bond_angle",
        "_build_bond_angle",
    ),
    AnalysisEntry(
        "persistence-length",
        "persistence_length",
        "setup_persistence_args",
        "_spec_persistence",
        "_build_persistence",
    ),
    AnalysisEntry("docking", "docking", None, None, "_build_docking"),
    AnalysisEntry("dssp", "dssp", None, None, "_build_dssp"),
    AnalysisEntry("diffusion", "diffusion", None, None, "_build_diffusion"),
    AnalysisEntry("pca", "pca", None, None, "_build_pca"),
    AnalysisEntry("rmsf", "rmsf", None, None, "_build_rmsf"),
    AnalysisEntry("density", "density", None, None, "_build_density"),
    AnalysisEntry("native-contacts", "native_contacts", None, None, "_build_native_contacts"),
    AnalysisEntry("volmap", "volmap", None, None, "_build_volmap"),
    AnalysisEntry("surf", "surf", None, None, "_build_surf"),
    AnalysisEntry("molsurf", "molsurf", None, None, "_build_molsurf"),
    AnalysisEntry("watershell", "watershell", None, None, "_build_watershell"),
    AnalysisEntry("tordiff", "tordiff", None, None, "_build_tordiff"),
    AnalysisEntry("projection", "projection", None, None, "_build_projection"),
    AnalysisEntry("nmr", "nmr", None, None, "_build_nmr"),
    AnalysisEntry("jcoupling", "jcoupling", None, None, "_build_jcoupling"),
    AnalysisEntry("gist", "gist", None, None, "_build_gist"),
)
