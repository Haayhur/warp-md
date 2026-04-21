from warp_md import cli_args, cli_builders, cli_specs


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
