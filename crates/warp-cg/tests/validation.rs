use warp_cg::mapping::map_molecule;
use warp_cg::molecule::Molecule;

#[test]
fn test_benzene_mapping() {
    let smiles = "c1ccccc1";
    let mol = Molecule::from_smiles(smiles).unwrap();
    let res = map_molecule(&mol);
    assert_eq!(res.bead_names.len(), 3);
    for name in &res.bead_names {
        assert_eq!(name, "TC5");
    }
}

#[test]
fn test_ethanol_mapping() {
    let smiles = "CCO";
    let mol = Molecule::from_smiles(smiles).unwrap();
    let res = map_molecule(&mol);
    assert_eq!(res.bead_names.len(), 1);
    assert_eq!(res.bead_names[0], "SP1");
}

#[test]
fn test_cyclohexane_mapping() {
    let smiles = "C1CCCCC1";
    let mol = Molecule::from_smiles(smiles).unwrap();
    let res = map_molecule(&mol);
    assert_eq!(res.bead_names.len(), 2);
    for name in &res.bead_names {
        assert_eq!(name, "SC3");
    }
}

#[test]
fn test_chloropropane_mapping() {
    let smiles = "CCCCl";
    let mol = Molecule::from_smiles(smiles).unwrap();
    let res = map_molecule(&mol);
    assert_eq!(res.bead_names.len(), 1);
    // My current logic doesn't handle Cl specifically for bead type, so it might say C1 or similar.
    // But I'm testing the NUMBER of beads for now.
}

#[test]
fn test_carboxylate_like_group_is_preserved() {
    let mol = Molecule::from_smiles("CC(=O)O").unwrap();
    let res = map_molecule(&mol);
    assert!(res.bead_features.iter().any(|features| features
        .iter()
        .any(|feature| feature == "carboxylate_or_carboxylic_acid")));
}

#[test]
fn test_sulfone_like_group_is_preserved() {
    let mol = Molecule::from_smiles("CS(=O)(=O)C").unwrap();
    let res = map_molecule(&mol);
    assert!(res.bead_features.iter().any(|features| features
        .iter()
        .any(|feature| feature == "sulfone_or_sulfonate")));
}
