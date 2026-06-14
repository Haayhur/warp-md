from warp_md import cli_args, cli_builders, cli_specs
import numpy as np


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


def test_lipid_builder_loads_leaflet_array(tmp_path) -> None:
    leaflets = np.array([[1, -1], [1, -1]], dtype=np.int8)
    leaflets_path = tmp_path / "leaflets.npz"
    np.savez(leaflets_path, values=leaflets)

    plan = cli_builders.PLAN_BUILDERS["lipid_area"](
        None,
        {"selection": "name PO4", "leaflets": str(leaflets_path)},
    )

    np.testing.assert_array_equal(plan._kwargs["leaflets"], leaflets)
