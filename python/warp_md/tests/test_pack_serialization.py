
import pytest
from warp_md.pack import PackConfig, Structure, Box, Constraint, AtomOverride, OutputSpec

def test_box_serialization():
    original = Box((10.0, 20.0, 30.0), shape="orthorhombic")
    dumped = original.to_dict()
    restored = Box.from_dict(dumped)
    
    assert dumped == {"size": [10.0, 20.0, 30.0], "shape": "orthorhombic"}
    assert restored == original

def test_structure_serialization_minimal():
    original = Structure("test.pdb", count=5)
    dumped = original.to_dict()
    restored = Structure.from_dict(dumped)
    
    # Defaults should be preserved/handled
    assert restored.path == "test.pdb"
    assert restored.count == 5
    assert restored.center is True # default
    assert restored == original

def test_structure_serialization_full():
    original = Structure(
        "complex.pdb",
        count=10,
        name="Complex",
        topology="top.psf",
        chain="A",
        segid="SEG",
        rotate=False,
        fixed=True,
        center=False,
        connect=False,
        min_distance=3.5,
        positions=[(1,1,1), (2,2,2)],
        translate=(5,5,5),
        constraints=[
            Constraint(mode="inside", shape="sphere", center=(0,0,0), radius=10.0)
        ],
        atom_overrides=[
            AtomOverride(indices=[1,2,3], radius=1.5)
        ]
    )
    dumped = original.to_dict()
    restored = Structure.from_dict(dumped)
    
    assert restored == original
    assert restored.constraints[0].mode == "inside"
    assert restored.atom_overrides[0].indices == [1,2,3]

def test_pack_config_serialization():
    original = PackConfig(
        structures=[Structure("water.pdb", count=100)],
        box=Box((50,50,50)),
        seed=42,
        output=OutputSpec("out.pdb", "pdb"),
        min_distance=2.2,
        pbc=True,
        max_attempts=500
    )
    dumped = original.to_dict()
    restored = PackConfig.from_dict(dumped)
    
    assert restored == original
    assert restored.seed == 42
    assert restored.output.path == "out.pdb"


def test_pack_config_defaults_use_packmol_gencan_controls():
    restored = PackConfig.from_dict(
        {
            "box": {"size": [10.0, 10.0, 10.0], "shape": "orthorhombic"},
            "structures": [{"path": "water.pdb", "count": 1}],
        }
    )
    assert restored.gencan_maxit is None


def test_pack_config_omits_removed_use_gencan_field():
    cfg = PackConfig(
        structures=[Structure("water.pdb", count=1)],
        box=Box((10.0, 10.0, 10.0)),
    )
    dumped = cfg.to_dict()
    assert "use_gencan" not in dumped
