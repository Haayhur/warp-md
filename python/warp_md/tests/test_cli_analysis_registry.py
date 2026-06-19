import argparse

from warp_md import cli_args, cli_builders, cli_specs
import numpy as np
import warp_md


def test_cli_arg_and_spec_registries_are_synced() -> None:
    assert set(cli_args.REGISTRY) == set(cli_specs.SPEC_BUILDERS)


def test_cli_to_plan_contains_all_cli_commands() -> None:
    for cmd_name in cli_args.REGISTRY:
        assert cmd_name in cli_builders.CLI_TO_PLAN
        plan_name = cli_builders.CLI_TO_PLAN[cmd_name]
        assert plan_name in cli_builders.PLAN_BUILDERS


def test_cli_to_plan_legacy_aliases_present() -> None:
    assert cli_builders.CLI_TO_PLAN["ffv"] == "bondi_ffv"
    assert cli_builders.CLI_TO_PLAN["native-contacts"] == "native_contacts"


def test_lipid_cli_plans_are_registered() -> None:
    expected = {
        "lipid-leaflets": "lipid_leaflets",
        "lipid-area": "lipid_area",
        "lipid-neighbour-matrix": "lipid_neighbour_matrix",
        "lipid-membrane-thickness": "lipid_membrane_thickness",
        "lipid-scc": "lipid_scc",
    }
    for cli_name, plan_name in expected.items():
        assert cli_builders.CLI_TO_PLAN[cli_name] == plan_name
        assert plan_name in cli_builders.PLAN_BUILDERS


def test_native_easy_medium_cli_plans_are_registered() -> None:
    expected = {
        "bundle": "bundle",
        "current": "current",
        "density": "density",
        "dipole-moments": "dipole_moments",
        "docking": "docking",
        "drid": "drid",
        "dssp": "dssp",
        "diffusion": "diffusion",
        "gyrate": "gyrate",
        "gist": "gist",
        "h2order": "h2order",
        "helix": "helix",
        "helixorient": "helixorient",
        "hydorder": "hydorder",
        "kabsch-sander": "kabsch_sander",
        "mdmat": "mdmat",
        "native-contacts": "native_contacts",
        "pca": "pca",
        "projection": "projection",
        "rama": "rama",
        "rmsf": "rmsf",
        "saltbr": "saltbr",
        "shape-descriptors": "shape_descriptors",
        "sorient": "sorient",
        "spol": "spol",
        "surf": "surf",
        "molsurf": "molsurf",
        "runningavg": "runningavg",
        "lineardensity": "lineardensity",
        "nematic-order": "nematic_order",
        "nmr": "nmr",
        "jcoupling": "jcoupling",
        "tordiff": "tordiff",
        "volmap": "volmap",
        "watershell": "watershell",
    }
    for cli_name, plan_name in expected.items():
        assert cli_builders.CLI_TO_PLAN[cli_name] == plan_name
        assert cli_name in cli_args.REGISTRY
        assert cli_name in cli_specs.SPEC_BUILDERS
        assert plan_name in cli_builders.PLAN_BUILDERS


def test_pairdist_cli_plan_is_registered() -> None:
    assert cli_args.REGISTRY["hbond"] is cli_args.setup_hbond_args
    assert cli_specs.SPEC_BUILDERS["hbond"] is cli_specs._spec_hbond
    assert cli_builders.CLI_TO_PLAN["hbond"] == "hbond"
    assert "hbond" in cli_builders.PLAN_BUILDERS
    assert cli_args.REGISTRY["pairdist"] is cli_args.setup_pairdist_args
    assert cli_specs.SPEC_BUILDERS["pairdist"] is cli_specs._spec_pairdist
    assert cli_builders.CLI_TO_PLAN["pairdist"] == "pairdist"
    assert "pairdist" in cli_builders.PLAN_BUILDERS
    assert cli_args.REGISTRY["mindist"] is cli_args.setup_mindist_args
    assert cli_specs.SPEC_BUILDERS["mindist"] is cli_specs._spec_mindist
    assert cli_builders.CLI_TO_PLAN["mindist"] == "mindist"
    assert "mindist" in cli_builders.PLAN_BUILDERS
    assert cli_args.REGISTRY["maxdist"] is cli_args.setup_maxdist_args
    assert cli_specs.SPEC_BUILDERS["maxdist"] is cli_specs._spec_maxdist
    assert cli_builders.CLI_TO_PLAN["maxdist"] == "maxdist"
    assert "maxdist" in cli_builders.PLAN_BUILDERS


def test_pairdist_cli_spec_parses_native_options() -> None:
    parser = argparse.ArgumentParser()
    cli_args.setup_pairdist_args(parser)
    args = parser.parse_args(
        [
            "--mask",
            "@1",
            "--mask2",
            "@2",
            "--delta",
            "0.25",
            "--maxdist",
            "8.0",
            "--mode",
            "min",
            "--image",
            "--frame-indices",
            "0,2,4",
        ]
    )

    spec = cli_specs._spec_pairdist(args, None)

    assert spec == {
        "mask": "@1",
        "mask2": "@2",
        "delta": 0.25,
        "maxdist": 8.0,
        "mode": "min",
        "image": True,
        "frame_indices": [0, 2, 4],
    }


def test_pairdist_builder_uses_callable_analysis_wrapper() -> None:
    plan = cli_builders.PLAN_BUILDERS["pairdist"](
        None,
        {
            "mask": "@1",
            "mask2": "@2",
            "delta": 0.25,
            "maxdist": 8.0,
            "mode": "min",
            "image": True,
            "frame_indices": [0, 2],
            "chunk_frames": 16,
        },
    )

    assert plan._kwargs == {
        "mask": "@1",
        "mask2": "@2",
        "delta": 0.25,
        "maxdist": 8.0,
        "mode": "min",
        "frame_indices": [0, 2],
        "chunk_frames": 16,
        "image": True,
    }


def test_dist_extrema_cli_specs_and_builders_use_callable_wrappers() -> None:
    parser = argparse.ArgumentParser()
    cli_args.setup_mindist_args(parser)
    args = parser.parse_args(
        [
            "--mask",
            "@1",
            "--mask2",
            "@2",
            "--maxdist",
            "5.0",
            "--image",
            "--frame-indices",
            "0,2",
        ]
    )
    assert cli_specs._spec_mindist(args, None) == {
        "mask": "@1",
        "mode": "min",
        "image": True,
        "mask2": "@2",
        "maxdist": 5.0,
        "frame_indices": [0, 2],
    }

    parser = argparse.ArgumentParser()
    cli_args.setup_maxdist_args(parser)
    args = parser.parse_args(["--mask", "all", "--no-image"])
    assert cli_specs._spec_maxdist(args, None) == {
        "mask": "all",
        "mode": "max",
        "image": False,
    }

    plan = cli_builders.PLAN_BUILDERS["mindist"](
        None,
        {
            "mask": "@1",
            "mask2": "@2",
            "maxdist": 5.0,
            "image": True,
            "frame_indices": [0, 2],
            "chunk_frames": 8,
        },
    )
    assert plan._kwargs == {
        "mask": "@1",
        "mask2": "@2",
        "maxdist": 5.0,
        "frame_indices": [0, 2],
        "chunk_frames": 8,
        "image": True,
    }

    plan = cli_builders.PLAN_BUILDERS["maxdist"](
        None,
        {"mask": "all", "image": False},
    )
    assert plan._kwargs == {"mask": "all", "image": False}


def test_hbond_cli_spec_and_builder_use_callable_wrapper() -> None:
    parser = argparse.ArgumentParser()
    cli_args.setup_hbond_args(parser)
    args = parser.parse_args(
        [
            "--donors",
            "@1",
            "--acceptors",
            "@3",
            "--dist-cutoff",
            "3.0",
            "--hydrogens",
            "@2",
            "--angle-cutoff",
            "120.0",
            "--frame-indices",
            "1,0",
        ]
    )

    assert cli_specs._spec_hbond(args, None) == {
        "donors": "@1",
        "acceptors": "@3",
        "dist_cutoff": 3.0,
        "hydrogens": "@2",
        "angle_cutoff": 120.0,
        "frame_indices": [1, 0],
    }

    plan = cli_builders.PLAN_BUILDERS["hbond"](
        None,
        {
            "donors": "@1",
            "acceptors": "@3",
            "dist_cutoff": 3.0,
            "hydrogens": "@2",
            "angle_cutoff": 120.0,
            "frame_indices": [1, 0],
            "chunk_frames": 8,
        },
    )
    assert plan._kwargs == {
        "donors": "@1",
        "acceptors": "@3",
        "dist_cutoff": 3.0,
        "hydrogens": "@2",
        "angle_cutoff": 120.0,
        "frame_indices": [1, 0],
        "chunk_frames": 8,
    }


def test_saltbr_cli_spec_and_builder_use_callable_wrapper() -> None:
    parser = argparse.ArgumentParser()
    cli_args.setup_saltbr_args(parser)
    args = parser.parse_args(
        [
            "--selection",
            "all",
            "--charges",
            "[1.0, -1.0]",
            "--group-by",
            "atom",
            "--truncate",
            "4.0",
            "--contact-cutoff",
            "3.2",
            "--length-scale",
            "0.1",
            "--frame-indices",
            "0,2",
        ]
    )

    assert cli_specs._spec_saltbr(args, None) == {
        "selection": "all",
        "group_by": "atom",
        "charges": [1.0, -1.0],
        "truncate": 4.0,
        "contact_cutoff": 3.2,
        "length_scale": 0.1,
        "frame_indices": [0, 2],
    }

    plan = cli_builders.PLAN_BUILDERS["saltbr"](
        None,
        {
            "selection": "all",
            "charges": [1.0, -1.0],
            "group_by": "atom",
            "truncate": 4.0,
            "contact_cutoff": 3.2,
            "length_scale": 0.1,
            "frame_indices": [0, 2],
            "chunk_frames": 8,
        },
    )
    assert cli_builders.CLI_TO_PLAN["salt-bridge"] == "saltbr"
    assert plan._kwargs == {
        "selection": "all",
        "charges": [1.0, -1.0],
        "group_by": "atom",
        "truncate": 4.0,
        "contact_cutoff": 3.2,
        "length_scale": 0.1,
        "frame_indices": [0, 2],
        "chunk_frames": 8,
    }


def test_water_orientation_cli_specs_and_builders_use_callable_wrappers() -> None:
    parser = argparse.ArgumentParser()
    cli_args.setup_h2order_args(parser)
    args = parser.parse_args(
        [
            "--selection",
            "resname SOL",
            "--charges",
            "[-0.834, 0.417, 0.417]",
            "--axis",
            "z",
            "--bin",
            "0.2",
            "--n-slices",
            "8",
            "--length-scale",
            "0.1",
            "--water-resnames",
            "SOL,WAT",
            "--frame-indices",
            "0,2",
        ]
    )
    assert cli_specs._spec_h2order(args, None) == {
        "selection": "resname SOL",
        "charges": [-0.834, 0.417, 0.417],
        "axis": "z",
        "bin": 0.2,
        "n_slices": 8,
        "length_scale": 0.1,
        "water_resnames": ["SOL", "WAT"],
        "frame_indices": [0, 2],
    }
    plan = cli_builders.PLAN_BUILDERS["h2order"](None, cli_specs._spec_h2order(args, None))
    assert cli_builders.CLI_TO_PLAN["water-order"] == "h2order"
    assert plan._kwargs["water_resnames"] == ["SOL", "WAT"]
    assert plan._kwargs["frame_indices"] == [0, 2]

    parser = argparse.ArgumentParser()
    cli_args.setup_hydorder_args(parser)
    args = parser.parse_args(
        [
            "--selection",
            "name OW",
            "--axis",
            "x",
            "--bin",
            "0.5",
            "--tblock",
            "4",
            "--sgang1",
            "0.1",
            "--sgang2",
            "0.9",
            "--length-scale",
            "0.1",
            "--frame-indices",
            "1,3",
        ]
    )
    spec = cli_specs._spec_hydorder(args, None)
    assert spec == {
        "selection": "name OW",
        "axis": "x",
        "bin": 0.5,
        "tblock": 4,
        "sgang1": 0.1,
        "sgang2": 0.9,
        "length_scale": 0.1,
        "frame_indices": [1, 3],
    }
    plan = cli_builders.PLAN_BUILDERS["hydorder"](None, spec)
    assert cli_builders.CLI_TO_PLAN["hydration-order"] == "hydorder"
    assert plan._kwargs == spec


def test_solvent_orientation_cli_specs_and_builders_use_callable_wrappers() -> None:
    parser = argparse.ArgumentParser()
    cli_args.setup_sorient_args(parser)
    args = parser.parse_args(
        [
            "--solute-selection",
            "protein",
            "--solvent-selection",
            "resname SOL",
            "--atom1-indices",
            "1,4",
            "--atom2-indices",
            "2,5",
            "--atom3-indices",
            "3,6",
            "--r-min",
            "0.1",
            "--r-max",
            "1.5",
            "--cbin",
            "0.5",
            "--rbin",
            "1.0",
            "--use-com",
            "--use-vector23",
            "--r-profile-max",
            "2.0",
            "--length-scale",
            "0.1",
            "--water-resnames",
            "SOL,WAT",
            "--frame-indices",
            "0,2",
        ]
    )
    spec = cli_specs._spec_sorient(args, None)
    assert spec == {
        "solute_selection": "protein",
        "solvent_selection": "resname SOL",
        "atom1_indices": [1, 4],
        "atom2_indices": [2, 5],
        "atom3_indices": [3, 6],
        "r_min": 0.1,
        "r_max": 1.5,
        "cbin": 0.5,
        "rbin": 1.0,
        "use_com": True,
        "use_vector23": True,
        "r_profile_max": 2.0,
        "length_scale": 0.1,
        "water_resnames": ["SOL", "WAT"],
        "frame_indices": [0, 2],
    }
    plan = cli_builders.PLAN_BUILDERS["sorient"](None, spec)
    assert cli_builders.CLI_TO_PLAN["solvent-orientation"] == "sorient"
    assert plan._kwargs == spec

    parser = argparse.ArgumentParser()
    cli_args.setup_spol_args(parser)
    args = parser.parse_args(
        [
            "--solute-selection",
            "protein",
            "--solvent-selection",
            "resname SOL",
            "--charges",
            "[-0.834, 0.417, 0.417]",
            "--atom1-indices",
            "1",
            "--atom2-indices",
            "2",
            "--atom3-indices",
            "3",
            "--r-min",
            "0.1",
            "--r-max",
            "1.5",
            "--bin",
            "0.5",
            "--use-com",
            "--reference-atom",
            "1",
            "--direction-atom-offsets",
            "0,1,2",
            "--refdip",
            "1.0",
            "--r-hist-max",
            "2.0",
            "--length-scale",
            "0.1",
            "--water-resnames",
            "SOL,WAT",
            "--frame-indices",
            "0,2",
        ]
    )
    spec = cli_specs._spec_spol(args, None)
    assert spec == {
        "solute_selection": "protein",
        "solvent_selection": "resname SOL",
        "charges": [-0.834, 0.417, 0.417],
        "atom1_indices": [1],
        "atom2_indices": [2],
        "atom3_indices": [3],
        "r_min": 0.1,
        "r_max": 1.5,
        "bin": 0.5,
        "use_com": True,
        "reference_atom": 1,
        "direction_atom_offsets": (0, 1, 2),
        "refdip": 1.0,
        "r_hist_max": 2.0,
        "length_scale": 0.1,
        "water_resnames": ["SOL", "WAT"],
        "frame_indices": [0, 2],
    }
    plan = cli_builders.PLAN_BUILDERS["spol"](None, spec)
    assert cli_builders.CLI_TO_PLAN["solvent-polarization"] == "spol"
    assert plan._kwargs == spec


def test_rama_cli_spec_and_builder_use_callable_wrapper() -> None:
    parser = argparse.ArgumentParser()
    cli_args.setup_rama_args(parser)
    args = parser.parse_args(
        [
            "--selection",
            "protein",
            "--range360",
            "--frame-indices",
            "0,2",
        ]
    )
    spec = cli_specs._spec_rama(args, None)
    assert spec == {
        "selection": "protein",
        "range360": True,
        "frame_indices": [0, 2],
    }
    plan = cli_builders.PLAN_BUILDERS["rama"](None, spec)
    assert cli_builders.CLI_TO_PLAN["ramachandran"] == "rama"
    assert plan._kwargs == spec


def test_current_cli_spec_and_builder_use_callable_wrapper() -> None:
    parser = argparse.ArgumentParser()
    cli_args.setup_current_args(parser)
    args = parser.parse_args(
        [
            "--selection",
            "all",
            "--charges",
            "[1.0, -1.0]",
            "--temperature",
            "310.0",
            "--group-by",
            "atom",
            "--length-scale",
            "0.1",
            "--no-make-whole",
            "--frame-decimation",
            "0,2",
            "--dt-decimation",
            "8,2,16,4",
            "--time-binning",
            "1e-6,1e-8",
            "--lag-mode",
            "ring",
            "--max-lag",
            "8",
            "--memory-budget-bytes",
            "4096",
            "--multi-tau-m",
            "8",
            "--multi-tau-levels",
            "6",
            "--frame-indices",
            "0,2",
        ]
    )
    spec = cli_specs._spec_current(args, None)
    assert spec == {
        "selection": "all",
        "charges": [1.0, -1.0],
        "temperature": 310.0,
        "group_by": "atom",
        "length_scale": 0.1,
        "make_whole": False,
        "frame_decimation": (0, 2),
        "dt_decimation": (8, 2, 16, 4),
        "time_binning": (1e-6, 1e-8),
        "lag_mode": "ring",
        "max_lag": 8,
        "memory_budget_bytes": 4096,
        "multi_tau_m": 8,
        "multi_tau_levels": 6,
        "frame_indices": [0, 2],
    }
    plan = cli_builders.PLAN_BUILDERS["current"](None, spec)
    assert plan._kwargs == spec


def test_bundle_and_helix_cli_specs_and_builders_use_callable_wrappers() -> None:
    parser = argparse.ArgumentParser()
    cli_args.setup_bundle_args(parser)
    args = parser.parse_args(
        [
            "--top-selection",
            "top",
            "--bottom-selection",
            "bottom",
            "--n-axes",
            "2",
            "--kink-selection",
            "kink",
            "--use-z-reference",
            "--no-mass-weighted",
            "--length-scale",
            "0.1",
            "--frame-indices",
            "0,2",
        ]
    )
    spec = cli_specs._spec_bundle(args, None)
    assert spec == {
        "top_selection": "top",
        "bottom_selection": "bottom",
        "n_axes": 2,
        "kink_selection": "kink",
        "use_z_reference": True,
        "mass_weighted": False,
        "length_scale": 0.1,
        "frame_indices": [0, 2],
    }
    plan = cli_builders.PLAN_BUILDERS["bundle"](None, spec)
    assert plan._kwargs == spec

    parser = argparse.ArgumentParser()
    cli_args.setup_helix_args(parser)
    args = parser.parse_args(
        [
            "--selection",
            "name CA",
            "--no-fit",
            "--check-each-frame",
            "--residue-start",
            "2",
            "--residue-end",
            "5",
            "--length-scale",
            "0.1",
            "--frame-indices",
            "0,3",
        ]
    )
    spec = cli_specs._spec_helix(args, None)
    assert spec == {
        "selection": "name CA",
        "fit": False,
        "check_each_frame": True,
        "residue_start": 2,
        "residue_end": 5,
        "length_scale": 0.1,
        "frame_indices": [0, 3],
    }
    plan = cli_builders.PLAN_BUILDERS["helix"](None, spec)
    assert plan._kwargs == spec

    parser = argparse.ArgumentParser()
    cli_args.setup_helixorient_args(parser)
    args = parser.parse_args(
        [
            "--ca-selection",
            "name CA",
            "--sidechain-selection",
            "name CB",
            "--incremental",
            "--length-scale",
            "0.1",
            "--frame-indices",
            "0,2",
        ]
    )
    spec = cli_specs._spec_helixorient(args, None)
    assert spec == {
        "ca_selection": "name CA",
        "sidechain_selection": "name CB",
        "incremental": True,
        "length_scale": 0.1,
        "frame_indices": [0, 2],
    }
    plan = cli_builders.PLAN_BUILDERS["helixorient"](None, spec)
    assert cli_builders.CLI_TO_PLAN["helix-orientation"] == "helixorient"
    assert plan._kwargs == spec


def test_mdmat_cli_spec_and_builder_use_callable_wrapper() -> None:
    parser = argparse.ArgumentParser()
    cli_args.setup_mdmat_args(parser)
    args = parser.parse_args(
        [
            "--selection",
            "resid 1:2",
            "--truncate",
            "1.5",
            "--include-contacts",
            "--include-frames",
            "--frames-mode",
            "artifact",
            "--frames-out",
            "frames.npz",
            "--memory-budget-bytes",
            "4096",
            "--length-scale",
            "0.1",
            "--frame-indices",
            "0,2",
        ]
    )
    spec = cli_specs._spec_mdmat(args, None)
    assert spec == {
        "selection": "resid 1:2",
        "truncate": 1.5,
        "include_contacts": True,
        "include_frames": True,
        "frames_mode": "artifact",
        "frames_out": "frames.npz",
        "memory_budget_bytes": 4096,
        "length_scale": 0.1,
        "frame_indices": [0, 2],
    }
    plan = cli_builders.PLAN_BUILDERS["mdmat"](None, spec)
    assert cli_builders.CLI_TO_PLAN["distance-matrix"] == "mdmat"
    assert plan._kwargs == spec


def test_grid_map_cli_specs_and_builders_use_callable_wrappers() -> None:
    parser = argparse.ArgumentParser()
    cli_args.setup_density_args(parser)
    args = parser.parse_args(
        [
            "--selection",
            "name O",
            "--center-selection",
            "protein",
            "--box-unit",
            "0.25,0.25,0.25",
            "--region-size",
            "4,4,4",
            "--shift",
            "1,2,3",
            "--length-scale",
            "0.1",
            "--frame-indices",
            "0,2",
        ]
    )
    spec = cli_specs._spec_density(args, None)
    assert spec == {
        "selection": "name O",
        "center_selection": "protein",
        "box_unit": (0.25, 0.25, 0.25),
        "region_size": (4.0, 4.0, 4.0),
        "shift": (1.0, 2.0, 3.0),
        "length_scale": 0.1,
        "frame_indices": [0, 2],
    }
    plan = cli_builders.PLAN_BUILDERS["density"](None, spec)
    assert plan._kwargs == spec

    parser = argparse.ArgumentParser()
    cli_args.setup_volmap_args(parser)
    args = parser.parse_args(
        [
            "--selection",
            "all",
            "--center-selection",
            "resid 1",
            "--box-unit",
            "1,1,1",
            "--region-size",
            "2,2,2",
        ]
    )
    spec = cli_specs._spec_volmap(args, None)
    assert spec == {
        "selection": "all",
        "center_selection": "resid 1",
        "box_unit": (1.0, 1.0, 1.0),
        "region_size": (2.0, 2.0, 2.0),
    }
    plan = cli_builders.PLAN_BUILDERS["volmap"](None, spec)
    assert plan._kwargs == spec


def test_surface_and_watershell_cli_specs_and_builders_use_callable_wrappers() -> None:
    parser = argparse.ArgumentParser()
    cli_args.setup_surf_args(parser)
    args = parser.parse_args(
        [
            "--mask",
            "protein",
            "--algorithm",
            "sasa",
            "--probe-radius",
            "1.4",
            "--offset",
            "0.0",
            "--nbrcut",
            "2.5",
            "--solutemask",
            "protein",
            "--n-sphere-points",
            "32",
            "--radii",
            "1.0,1.5",
            "--radii-mode",
            "vdw",
            "--atom-area",
            "--volume",
            "--residue-area",
            "--frame-indices",
            "0,2",
        ]
    )
    spec = cli_specs._spec_surf(args, None)
    assert spec == {
        "mask": "protein",
        "algorithm": "sasa",
        "probe_radius": 1.4,
        "offset": 0.0,
        "nbrcut": 2.5,
        "solutemask": "protein",
        "n_sphere_points": 32,
        "radii": [1.0, 1.5],
        "radii_mode": "vdw",
        "atom_area": True,
        "volume": True,
        "residue_area": True,
        "frame_indices": [0, 2],
    }
    plan = cli_builders.PLAN_BUILDERS["surf"](None, spec)
    assert plan._kwargs == spec

    parser = argparse.ArgumentParser()
    cli_args.setup_molsurf_args(parser)
    args = parser.parse_args(
        [
            "--mask",
            "protein",
            "--algorithm",
            "sasa",
            "--probe",
            "1.2",
            "--radii",
            "parse",
            "--frame-indices",
            "1",
        ]
    )
    spec = cli_specs._spec_molsurf(args, None)
    assert spec == {
        "mask": "protein",
        "algorithm": "sasa",
        "probe": 1.2,
        "n_sphere_points": 64,
        "radii": "parse",
        "radii_mode": "gb",
        "atom_area": False,
        "volume": False,
        "residue_area": False,
        "frame_indices": [1],
    }
    plan = cli_builders.PLAN_BUILDERS["molsurf"](None, spec)
    assert plan._kwargs == spec

    parser = argparse.ArgumentParser()
    cli_args.setup_watershell_args(parser)
    args = parser.parse_args(
        [
            "--solute-mask",
            "protein",
            "--solvent-mask",
            "resname WAT",
            "--lower",
            "3.0",
            "--upper",
            "5.0",
            "--no-image",
            "--frame-indices",
            "0,2",
        ]
    )
    spec = cli_specs._spec_watershell(args, None)
    assert spec == {
        "solute_mask": "protein",
        "solvent_mask": "resname WAT",
        "lower": 3.0,
        "upper": 5.0,
        "image": False,
        "frame_indices": [0, 2],
    }
    plan = cli_builders.PLAN_BUILDERS["watershell"](None, spec)
    assert plan._kwargs == spec


def test_dssp_diffusion_rmsf_cli_specs_and_builders_use_callable_wrappers() -> None:
    parser = argparse.ArgumentParser()
    cli_args.setup_dssp_args(parser)
    args = parser.parse_args(
        [
            "--mask",
            "protein",
            "--no-simplified",
            "--dtype",
            "full",
            "--frame-indices",
            "0,2",
        ]
    )
    spec = cli_specs._spec_dssp(args, None)
    assert spec == {
        "mask": "protein",
        "simplified": False,
        "dtype": "full",
        "frame_indices": [0, 2],
    }
    plan = cli_builders.PLAN_BUILDERS["dssp"](None, spec)
    assert plan._kwargs == spec

    parser = argparse.ArgumentParser()
    cli_args.setup_diffusion_args(parser)
    args = parser.parse_args(
        [
            "--mask",
            "resname WAT",
            "--tstep",
            "2.0",
            "--individual",
            "--frame-indices",
            "1,3",
        ]
    )
    spec = cli_specs._spec_diffusion(args, None)
    assert spec == {
        "mask": "resname WAT",
        "tstep": 2.0,
        "individual": True,
        "frame_indices": [1, 3],
    }
    plan = cli_builders.PLAN_BUILDERS["diffusion"](None, spec)
    assert plan._kwargs == spec

    parser = argparse.ArgumentParser()
    cli_args.setup_rmsf_args(parser)
    args = parser.parse_args(
        [
            "--mask",
            "name CA",
            "--byres",
            "--calcadp",
            "--length-scale",
            "0.1",
            "--frame-indices",
            "0,2",
        ]
    )
    spec = cli_specs._spec_rmsf(args, None)
    assert spec == {
        "mask": "name CA",
        "byres": True,
        "bymask": False,
        "calcadp": True,
        "length_scale": 0.1,
        "frame_indices": [0, 2],
    }
    plan = cli_builders.PLAN_BUILDERS["rmsf"](None, spec)
    assert plan._kwargs == spec


def test_native_contacts_cli_spec_and_builder_use_callable_wrapper() -> None:
    parser = argparse.ArgumentParser()
    cli_args.setup_native_contacts_args(parser)
    args = parser.parse_args(
        [
            "--mask",
            "name CA",
            "--mask2",
            "name CB",
            "--ref",
            "topology",
            "--distance",
            "4.5",
            "--mindist",
            "1.0",
            "--maxdist",
            "6.0",
            "--image",
            "--frame-indices",
            "0,2",
        ]
    )
    spec = cli_specs._spec_native_contacts(args, None)
    assert spec == {
        "mask": "name CA",
        "mask2": "name CB",
        "ref": "topology",
        "distance": 4.5,
        "mindist": 1.0,
        "maxdist": 6.0,
        "image": True,
        "frame_indices": [0, 2],
    }
    plan = cli_builders.PLAN_BUILDERS["native_contacts"](None, spec)
    assert cli_builders.CLI_TO_PLAN["native-contacts"] == "native_contacts"
    assert plan._kwargs == spec


def test_docking_cli_spec_and_builder_use_callable_wrapper() -> None:
    parser = argparse.ArgumentParser()
    cli_args.setup_docking_args(parser)
    args = parser.parse_args(
        [
            "--receptor-mask",
            "protein",
            "--ligand-mask",
            "resname LIG",
            "--close-contact-cutoff",
            "5.0",
            "--hydrophobic-cutoff",
            "4.5",
            "--hydrogen-bond-cutoff",
            "3.2",
            "--clash-cutoff",
            "2.0",
            "--salt-bridge-cutoff",
            "5.0",
            "--halogen-bond-cutoff",
            "5.2",
            "--metal-coordination-cutoff",
            "3.0",
            "--cation-pi-cutoff",
            "6.5",
            "--pi-pi-cutoff",
            "7.0",
            "--hbond-min-angle-deg",
            "130.0",
            "--donor-hydrogen-cutoff",
            "1.1",
            "--allow-missing-hydrogen",
            "--length-scale",
            "0.1",
            "--max-events-per-frame",
            "64",
            "--frame-indices",
            "0,2",
        ]
    )
    spec = cli_specs._spec_docking(args, None)
    assert spec == {
        "receptor_mask": "protein",
        "ligand_mask": "resname LIG",
        "close_contact_cutoff": 5.0,
        "hydrophobic_cutoff": 4.5,
        "hydrogen_bond_cutoff": 3.2,
        "clash_cutoff": 2.0,
        "salt_bridge_cutoff": 5.0,
        "halogen_bond_cutoff": 5.2,
        "metal_coordination_cutoff": 3.0,
        "cation_pi_cutoff": 6.5,
        "pi_pi_cutoff": 7.0,
        "hbond_min_angle_deg": 130.0,
        "donor_hydrogen_cutoff": 1.1,
        "allow_missing_hydrogen": True,
        "length_scale": 0.1,
        "max_events_per_frame": 64,
        "frame_indices": [0, 2],
    }
    plan = cli_builders.PLAN_BUILDERS["docking"](None, spec)
    assert plan._kwargs == spec


def test_pca_projection_tordiff_cli_specs_and_builders_use_callable_wrappers() -> None:
    parser = argparse.ArgumentParser()
    cli_args.setup_pca_args(parser)
    args = parser.parse_args(
        [
            "--mask",
            "name CA",
            "--n-vecs",
            "3",
            "--no-fit",
            "--ref",
            "0",
            "--ref-mask",
            "protein",
            "--dtype",
            "tuple",
        ]
    )
    spec = cli_specs._spec_pca(args, None)
    assert spec == {
        "mask": "name CA",
        "n_vecs": 3,
        "fit": False,
        "ref": 0,
        "ref_mask": "protein",
        "dtype": "tuple",
    }
    plan = cli_builders.PLAN_BUILDERS["pca"](None, spec)
    assert plan._kwargs == spec

    parser = argparse.ArgumentParser()
    cli_args.setup_projection_args(parser)
    args = parser.parse_args(
        [
            "--mask",
            "name CA",
            "--eigenvectors",
            "[[1,0,0],[0,1,0]]",
            "--eigenvalues",
            "[2,1]",
            "--average-coords",
            "[[0,0,0]]",
            "--scalar-type",
            "mwcovar",
            "--frame-indices",
            "0,2",
        ]
    )
    spec = cli_specs._spec_projection(args, None)
    assert spec["mask"] == "name CA"
    np.testing.assert_array_equal(spec["eigenvectors"], np.array([[1, 0, 0], [0, 1, 0]]))
    np.testing.assert_array_equal(spec["eigenvalues"], np.array([2, 1]))
    np.testing.assert_array_equal(spec["average_coords"], np.array([[0, 0, 0]]))
    assert spec["scalar_type"] == "mwcovar"
    assert spec["frame_indices"] == [0, 2]
    plan = cli_builders.PLAN_BUILDERS["projection"](None, spec)
    assert plan._kwargs == spec

    parser = argparse.ArgumentParser()
    cli_args.setup_tordiff_args(parser)
    args = parser.parse_args(
        [
            "--mask",
            "protein",
            "--mass",
            "--time",
            "2.5",
            "--diffout",
            "torsion.diff",
            "--return-transitions",
            "--transition-lag",
            "3",
            "--frame-indices",
            "1,4",
        ]
    )
    spec = cli_specs._spec_tordiff(args, None)
    assert spec == {
        "mask": "protein",
        "mass": True,
        "time": 2.5,
        "return_transitions": True,
        "transition_lag": 3,
        "diffout": "torsion.diff",
        "frame_indices": [1, 4],
    }
    plan = cli_builders.PLAN_BUILDERS["tordiff"](None, spec)
    assert plan._kwargs == spec


def test_nmr_and_jcoupling_cli_specs_and_builders_use_callable_wrappers() -> None:
    parser = argparse.ArgumentParser()
    cli_args.setup_nmr_args(parser)
    args = parser.parse_args(
        [
            "--selection",
            "name N H",
            "--vector-pairs",
            "[[0,1],[2,3]]",
            "--method",
            "timecorr_fit",
            "--order",
            "2",
            "--tstep",
            "0.5",
            "--tcorr",
            "10.0",
            "--length-scale",
            "0.1",
            "--pbc",
            "orthorhombic",
            "--frame-indices",
            "0,2",
        ]
    )
    spec = cli_specs._spec_nmr(args, None)
    assert spec == {
        "selection": "name N H",
        "vector_pairs": [[0, 1], [2, 3]],
        "method": "timecorr_fit",
        "order": 2,
        "tstep": 0.5,
        "tcorr": 10.0,
        "length_scale": 0.1,
        "pbc": "orthorhombic",
        "frame_indices": [0, 2],
    }
    plan = cli_builders.PLAN_BUILDERS["nmr"](None, spec)
    assert plan._kwargs == spec

    parser = argparse.ArgumentParser()
    cli_args.setup_jcoupling_args(parser)
    args = parser.parse_args(
        [
            "--dihedrals",
            "[[0,1,2,3],[4,5,6,7]]",
            "--karplus",
            "7.0,-1.0,2.0",
            "--phase-deg",
            "30.0",
            "--length-scale",
            "0.2",
            "--pbc",
            "orthorhombic",
            "--return-dihedral",
            "--frame-indices",
            "1,3",
        ]
    )
    spec = cli_specs._spec_jcoupling(args, None)
    assert spec == {
        "dihedrals": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "karplus": (7.0, -1.0, 2.0),
        "kfile": None,
        "phase_deg": 30.0,
        "length_scale": 0.2,
        "pbc": "orthorhombic",
        "return_dihedral": True,
        "frame_indices": [1, 3],
    }
    plan = cli_builders.PLAN_BUILDERS["jcoupling"](None, spec)
    assert plan._kwargs == {
        "dihedral_indices": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "karplus": (7.0, -1.0, 2.0),
        "phase_deg": 30.0,
        "length_scale": 0.2,
        "pbc": "orthorhombic",
        "return_dihedral": True,
        "frame_indices": [1, 3],
    }


def test_gist_cli_spec_and_builder_use_native_grid_wrapper() -> None:
    parser = argparse.ArgumentParser()
    cli_args.setup_gist_args(parser)
    args = parser.parse_args(
        [
            "--energy-method",
            "none",
            "--grid-spacing",
            "0.2",
            "--padding",
            "0.4",
            "--temperature",
            "310.0",
            "--length-scale",
            "1.0",
            "--orientation-bins",
            "4",
            "--solute-selection",
            "protein",
            "--max-frames",
            "8",
            "--water-resnames",
            "HOH,WAT",
            "--bulk-density",
            "33.3",
            "--frame-indices",
            "0,2",
        ]
    )
    spec = cli_specs._spec_gist(args, None)
    assert spec == {
        "energy_method": "none",
        "grid_spacing": 0.2,
        "padding": 0.4,
        "temperature": 310.0,
        "length_scale": 1.0,
        "orientation_bins": 4,
        "solute_selection": "protein",
        "max_frames": 8,
        "water_resnames": ("HOH", "WAT"),
        "bulk_density": 33.3,
        "frame_indices": [0, 2],
    }
    plan = cli_builders.PLAN_BUILDERS["gist"](None, spec)
    assert plan._kwargs == {}


def test_gyrate_cli_spec_parses_native_options() -> None:
    parser = argparse.ArgumentParser()
    cli_args.setup_gyrate_args(parser)
    args = parser.parse_args(
        [
            "--selection",
            "all",
            "--no-mass",
            "--axes",
            "--max-radius",
            "--tensor",
            "--length-scale",
            "1.0",
            "--frame-indices",
            "0,2",
        ]
    )

    assert cli_specs._spec_gyrate(args, None) == {
        "selection": "all",
        "mass": False,
        "axes": True,
        "nomax": False,
        "tensor": True,
        "dtype": "dict",
        "length_scale": 1.0,
        "frame_indices": [0, 2],
    }


def test_gyrate_builder_uses_callable_analysis_wrapper() -> None:
    plan = cli_builders.PLAN_BUILDERS["gyrate"](
        None,
        {
            "selection": "all",
            "mass": True,
            "axes": True,
            "nomax": True,
            "tensor": False,
            "dtype": "dict",
            "frame_indices": [1],
            "chunk_frames": 8,
        },
    )

    assert plan._kwargs == {
        "mask": "all",
        "mass": True,
        "axes": True,
        "nomax": True,
        "tensor": False,
        "dtype": "dict",
        "frame_indices": [1],
        "chunk_frames": 8,
    }


def test_native_easy_medium_cli_specs_parse_compact_options() -> None:
    parser = argparse.ArgumentParser()
    cli_args.setup_dipole_moments_args(parser)
    args = parser.parse_args(
        [
            "--selection",
            "all",
            "--charges",
            "[1.0, -1.0]",
            "--group-by",
            "resid",
            "--length-scale",
            "2.0",
            "--frame-indices",
            "0,2",
        ]
    )
    assert cli_specs._spec_dipole_moments(args, None) == {
        "selection": "all",
        "group_by": "resid",
        "charges": [1.0, -1.0],
        "dtype": "dict",
        "length_scale": 2.0,
        "frame_indices": [0, 2],
    }

    parser = argparse.ArgumentParser()
    cli_args.setup_lineardensity_args(parser)
    args = parser.parse_args(
        [
            "--selection",
            "all",
            "--axis",
            "x",
            "--bin",
            "0.5",
            "--range",
            "0,4",
            "--weight",
            "mass",
            "--norm",
            "density",
            "--cross-section-area",
            "12.0",
            "--frame-indices",
            "1,3",
        ]
    )
    assert cli_specs._spec_lineardensity(args, None) == {
        "selection": "all",
        "axis": "x",
        "bin": 0.5,
        "weight": "mass",
        "norm": "density",
        "range": (0.0, 4.0),
        "cross_section_area": 12.0,
        "frame_indices": [1, 3],
    }

    parser = argparse.ArgumentParser()
    cli_args.setup_nematic_order_args(parser)
    args = parser.parse_args(
        [
            "--selection",
            "@1,@2",
            "--reference-axis",
            "0,0,1",
            "--pbc",
            "--length-scale",
            "10.0",
            "--frame-indices",
            "0,2",
        ]
    )
    assert cli_specs._spec_nematic_order(args, None) == {
        "selection": "@1,@2",
        "pbc": True,
        "reference_axis": (0.0, 0.0, 1.0),
        "length_scale": 10.0,
        "frame_indices": [0, 2],
    }

    parser = argparse.ArgumentParser()
    cli_args.setup_kabsch_sander_args(parser)
    args = parser.parse_args(
        [
            "--selection",
            "all",
            "--energy-cutoff",
            "-0.3",
            "--frame-indices",
            "1,3",
        ]
    )
    assert cli_specs._spec_kabsch_sander(args, None) == {
        "selection": "all",
        "energy_cutoff": -0.3,
        "dtype": "dict",
        "frame_indices": [1, 3],
    }


def test_native_easy_medium_builders_use_callable_wrappers() -> None:
    system = warp_md.System.from_arrays(
        {
            "name": ["A", "B"],
            "resname": ["MOL", "MOL"],
            "resid": [1, 1],
            "chain_id": [0, 0],
            "mass": [1.0, 1.0],
        },
        positions0=np.zeros((2, 3), dtype=np.float32),
    )
    plan = cli_builders.PLAN_BUILDERS["dipole_moments"](
        system,
        {
            "selection": "all",
            "charges": [1.0, -1.0],
            "group_by": "resid",
            "length_scale": 2.0,
            "frame_indices": [0],
            "dtype": "dict",
        },
    )
    assert plan._kwargs == {
        "selection": "all",
        "group_by": "resid",
        "length_scale": 2.0,
        "frame_indices": [0],
        "dtype": "dict",
        "charges": [1.0, -1.0],
    }

    plan = cli_builders.PLAN_BUILDERS["shape_descriptors"](
        None,
        {"selection": "all", "mass": True, "frame_indices": [0]},
    )
    assert plan._kwargs == {"mass": True, "frame_indices": [0], "mask": "all"}

    plan = cli_builders.PLAN_BUILDERS["runningavg"](
        None,
        {"selection": "all", "window": 3, "frame_indices": [1, 2]},
    )
    assert plan._kwargs == {
        "selection": "all",
        "window": 3,
        "frame_indices": [1, 2],
    }

    plan = cli_builders.PLAN_BUILDERS["kabsch_sander"](
        None,
        {
            "selection": "all",
            "energy_cutoff": -0.3,
            "frame_indices": [1],
            "dtype": "dict",
        },
    )
    assert plan._kwargs == {
        "selection": "all",
        "energy_cutoff": -0.3,
        "frame_indices": [1],
        "dtype": "dict",
    }


def test_lipid_builder_loads_leaflet_array(tmp_path) -> None:
    leaflets = np.array([[1, -1], [1, -1]], dtype=np.int8)
    leaflets_path = tmp_path / "leaflets.npz"
    np.savez(leaflets_path, values=leaflets)

    plan = cli_builders.PLAN_BUILDERS["lipid_area"](
        None,
        {"selection": "name PO4", "leaflets": str(leaflets_path)},
    )

    np.testing.assert_array_equal(plan._kwargs["leaflets"], leaflets)
