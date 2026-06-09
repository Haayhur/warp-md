import warp_md as wmd


def test_top_level_bindings_expose_streaming_formats_and_helix_orientation():
    assert getattr(wmd, "_IMPORT_ERROR", None) is None
    assert hasattr(wmd.System, "from_file")
    assert hasattr(wmd.Trajectory, "open_gro")
    assert hasattr(wmd.Trajectory, "open_g96")
    assert hasattr(wmd.Trajectory, "open_cpt")
    assert hasattr(wmd.Trajectory, "open_h5md")
    assert hasattr(wmd.Trajectory, "open_tng")
    assert hasattr(wmd.TrajectoryWriter, "open")
    assert hasattr(wmd, "HelixOrientationPlan")
    assert hasattr(wmd, "HelixPlan")
