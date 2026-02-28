
import pytest
from warp_md.pack import PackConfig, Structure, Box, Constraint, ValidationError, AtomOverride

def test_pack_config_validation_basics():
    # Valid config
    cfg = PackConfig(
        structures=[Structure("water.pdb", count=1)],
        box=Box((10,10,10))
    )
    cfg.validate() # Should not raise

    # No structures
    cfg.structures = []
    with pytest.raises(ValidationError, match="requires at least one structure"):
        cfg.validate()

    # Invalid min_distance
    cfg.structures = [Structure("water.pdb")]
    cfg.min_distance = -1.0
    with pytest.raises(ValidationError, match="min_distance must be positive"):
        cfg.validate()

def test_box_validation():
    # Invalid size
    with pytest.raises(ValidationError, match="must have 3 dimensions"):
        Box((10,10)).validate()
    
    # Negative dimension
    with pytest.raises(ValidationError, match="must be positive"):
        Box((10, -5, 10)).validate()
        
    # Invalid shape
    with pytest.raises(ValidationError, match="Invalid box shape"):
        Box((10,10,10), shape="hypercube").validate()

def test_structure_validation():
    # Empty path
    with pytest.raises(ValidationError, match="path cannot be empty"):
        Structure(path="").validate()
        
    # Invalid count
    with pytest.raises(ValidationError, match="count must be >= 1"):
        Structure("test.pdb", count=0).validate()
        
    # Negative min_distance
    with pytest.raises(ValidationError, match="min_distance must be positive"):
        Structure("test.pdb", min_distance=-0.5).validate()

def test_constraint_validation():
    # Invalid mode
    with pytest.raises(ValidationError, match="Invalid mode"):
        Constraint(mode="teleport", shape="box").validate()

    # Invalid shape
    with pytest.raises(ValidationError, match="Invalid shape"):
        Constraint(mode="inside", shape="pyramid").validate()

    # Box missing min/max
    with pytest.raises(ValidationError, match="requires 'min' and 'max'"):
        Constraint(mode="inside", shape="box", min=(0,0,0)).validate()

    # Sphere missing center/radius
    with pytest.raises(ValidationError, match="requires 'center' and 'radius'"):
        Constraint(mode="inside", shape="sphere", radius=5.0).validate()

    # Sphere negative radius
    with pytest.raises(ValidationError, match="radius must be positive"):
        Constraint(mode="inside", shape="sphere", center=(0,0,0), radius=-1.0).validate()

def test_atom_override_validation():
    with pytest.raises(ValidationError, match="requires at least one index"):
        AtomOverride(indices=[]).validate()

    with pytest.raises(ValidationError, match="indices must be non-negative"):
        AtomOverride(indices=[1, -1]).validate()

    with pytest.raises(ValidationError, match="radius must be positive"):
        AtomOverride(indices=[1], radius=-0.5).validate()
