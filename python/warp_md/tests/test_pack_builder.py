
import pytest
from warp_md.pack import PackConfigBuilder, ValidationError, water_pdb
from warp_md.pack.config import PackConfig

def test_pack_builder_cubic_box():
    cfg = PackConfigBuilder().cubic_box(50.0).build(validate=False)
    assert cfg.box.size == (50.0, 50.0, 50.0)
    assert cfg.box.shape == "cubic"

def test_pack_builder_orthorhombic_box():
    cfg = PackConfigBuilder().box(40.0, 50.0, 60.0).build(validate=False)
    assert cfg.box.size == (40.0, 50.0, 60.0)
    assert cfg.box.shape == "orthorhombic"

def test_pack_builder_box_invalid_dims():
    with pytest.raises(ValidationError):
        PackConfigBuilder().box(10.0, 20.0) # Missing Z

    with pytest.raises(ValidationError):
        PackConfigBuilder().box(10.0, 20.0, shape="invalid") # Invalid kwarg implication, but here we just check explicit call

def test_structure_builder_fluent_chaining():
    # Test the new symmetric methods and fluent chaining
    cfg = (PackConfigBuilder()
        .cubic_box(100)
        .add("protein.pdb", count=1)
            .center(True)
            .rotate(False)
            .connect(False)
            .min_distance(5.0)
            .done()
        .build(validate=False)
    )
    
    s = cfg.structures[0]
    assert s.path == "protein.pdb"
    assert s.count == 1
    assert s.center is True
    assert s.rotate is False
    assert s.connect is False
    assert s.min_distance == 5.0

def test_structure_builder_constraints():
    cfg = (PackConfigBuilder()
        .cubic_box(100)
        .add("water.pdb", count=10)
            .inside_sphere((0,0,0), 10.0)
            .outside_box((0,0,0), (5,5,5))
            .done()
        .build(validate=False)
    )
    
    s = cfg.structures[0]
    assert len(s.constraints) == 2
    assert s.constraints[0].mode == "inside"
    assert s.constraints[0].shape == "sphere"
    assert s.constraints[0].center == (0,0,0)
    assert s.constraints[0].radius == 10.0
    
    assert s.constraints[1].mode == "outside"
    assert s.constraints[1].shape == "box"
    assert s.constraints[1].min == (0,0,0)
    assert s.constraints[1].max == (5,5,5)

def test_build_validation_trigger():
    # Should raise because no structures added
    builder = PackConfigBuilder().cubic_box(10.0)
    with pytest.raises(ValidationError):
        builder.build(validate=True)

    # Should raise because no box defined
    builder = PackConfigBuilder().add("foo.pdb") 
    with pytest.raises(ValidationError):
        builder.build(validate=True)
