use std::path::Path;

use crate::error::StructureResult as PackResult;
use crate::io::MoleculeData;
use traj_core::pdb_gro::PdbParseOptions;

use super::pdb::read_pdb_with_options;

pub fn read_pdbqt(path: &Path) -> PackResult<MoleculeData> {
    let options = PdbParseOptions {
        include_conect: false,
        include_ter: false,
        non_standard_conect: false,
        strict: false,
        only_first_model: true,
    };
    read_pdb_with_options(path, &options)
}
