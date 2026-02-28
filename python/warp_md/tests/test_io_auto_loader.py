from warp_md.io import open_trajectory_auto


class _FakeTrajectory:
    @staticmethod
    def open_dcd(path, system, length_scale=None):
        return ("dcd", path, system, length_scale)

    @staticmethod
    def open_xtc(path, system):
        return ("xtc", path, system, None)

    @staticmethod
    def open_pdb(path, system):
        return ("pdb", path, system, None)


def test_open_trajectory_auto_infers_dcd():
    out = open_trajectory_auto(
        "traj.dcd",
        system="sys",
        trajectory_cls=_FakeTrajectory,
        length_scale=10.0,
    )
    assert out == ("dcd", "traj.dcd", "sys", 10.0)


def test_open_trajectory_auto_uses_explicit_format():
    out = open_trajectory_auto(
        "traj.unknown",
        system="sys",
        format="xtc",
        trajectory_cls=_FakeTrajectory,
    )
    assert out == ("xtc", "traj.unknown", "sys", None)


def test_open_trajectory_auto_rejects_unknown_format():
    try:
        open_trajectory_auto("traj.foo", system="sys", trajectory_cls=_FakeTrajectory)
    except ValueError as exc:
        assert "unsupported trajectory format" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")

