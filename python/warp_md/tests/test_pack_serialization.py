
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


def test_pack_config_validate_allows_none_max_attempts():
    cfg = PackConfig(
        structures=[Structure("water.pdb", count=1)],
        box=Box((10.0, 10.0, 10.0)),
        max_attempts=None,
    )
    cfg.validate()


def test_pack_config_to_dict_omits_none_seed_and_max_attempts():
    cfg = PackConfig(
        structures=[Structure("water.pdb", count=1)],
        box=Box((10.0, 10.0, 10.0)),
        seed=None,
        max_attempts=None,
    )
    dumped = cfg.to_dict()
    assert "seed" not in dumped
    assert "max_attempts" not in dumped


def test_structure_from_dict_accepts_explicit_none_optionals():
    restored = Structure.from_dict(
        {
            "path": "ligand.pdb",
            "count": 1,
            "name": None,
            "topology": None,
            "restart_from": None,
            "restart_to": None,
            "fixed_eulers": None,
            "chain": None,
            "changechains": False,
            "segid": None,
            "connect": True,
            "format": None,
            "rotate": False,
            "fixed": True,
            "positions": None,
            "translate": None,
            "center": True,
            "min_distance": None,
            "resnumbers": None,
            "maxmove": None,
            "nloop": None,
            "nloop0": None,
            "constraints": [],
            "radius": None,
            "fscale": None,
            "short_radius": None,
            "short_radius_scale": None,
            "atom_overrides": [],
            "rot_bounds": None,
        }
    )

    assert restored.path == "ligand.pdb"
    assert restored.fixed is True
    assert restored.fixed_eulers is None
    assert restored.positions is None
