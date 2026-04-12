from warp_md.io import open_trajectory_auto


class _FakeTrajectory:
    @staticmethod
    def open_dcd(path, system, length_scale=None):
        return ("dcd", path, system, length_scale)

    @staticmethod
    def open_xtc(path, system):
        return ("xtc", path, system, None)

    @staticmethod
    def open_gro(path, system):
        return ("gro", path, system, None)

    @staticmethod
    def open_g96(path, system):
        return ("g96", path, system, None)

    @staticmethod
    def open_cpt(path, system):
        return ("cpt", path, system, None)

    @staticmethod
    def open_h5md(path, system):
        return ("h5md", path, system, None)

    @staticmethod
    def open_tng(path, system):
        return ("tng", path, system, None)

    @staticmethod
    def open_trr(path, system):
        return ("trr", path, system, None)

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


def test_open_trajectory_auto_infers_trr():
    out = open_trajectory_auto(
        "traj.trr",
        system="sys",
        trajectory_cls=_FakeTrajectory,
    )
    assert out == ("trr", "traj.trr", "sys", None)


def test_open_trajectory_auto_infers_gro():
    out = open_trajectory_auto(
        "traj.gro",
        system="sys",
        trajectory_cls=_FakeTrajectory,
    )
    assert out == ("gro", "traj.gro", "sys", None)


def test_open_trajectory_auto_infers_g96():
    out = open_trajectory_auto(
        "traj.g96",
        system="sys",
        trajectory_cls=_FakeTrajectory,
    )
    assert out == ("g96", "traj.g96", "sys", None)


def test_open_trajectory_auto_infers_tng():
    out = open_trajectory_auto(
        "traj.tng",
        system="sys",
        trajectory_cls=_FakeTrajectory,
    )
    assert out == ("tng", "traj.tng", "sys", None)


def test_open_trajectory_auto_infers_cpt():
    out = open_trajectory_auto(
        "state.cpt",
        system="sys",
        trajectory_cls=_FakeTrajectory,
    )
    assert out == ("cpt", "state.cpt", "sys", None)


def test_open_trajectory_auto_infers_h5md():
    out = open_trajectory_auto(
        "traj.h5md",
        system="sys",
        trajectory_cls=_FakeTrajectory,
    )
    assert out == ("h5md", "traj.h5md", "sys", None)


def test_open_trajectory_auto_rejects_unknown_format():
    try:
        open_trajectory_auto("traj.foo", system="sys", trajectory_cls=_FakeTrajectory)
    except ValueError as exc:
        assert "unsupported trajectory format" in str(exc)
        assert "h5md" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
