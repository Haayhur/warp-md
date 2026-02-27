//! PDB format writer.

use crate::residue::Structure;
use std::io::{self, Write};

/// Write structure as PDB ATOM records to the given writer.
pub fn write_pdb<W: Write>(struc: &Structure, mut out: W) -> io::Result<()> {
    let mut serial: u32 = 1;
    for chain in &struc.chains {
        for res in &chain.residues {
            for atom in &res.atoms {
                // PDB ATOM record format (columns 1-80)
                // ATOM  serial name altLoc resName chainID resSeq iCode  x y z occ bfac  element
                let atom_name_fmt = if atom.name.len() < 4 {
                    format!(" {:<3}", atom.name)
                } else {
                    format!("{:<4}", atom.name)
                };
                writeln!(
                    out,
                    "ATOM  {:>5} {} {:>3} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {:>2}",
                    serial,
                    atom_name_fmt,
                    res.name.as_str(),
                    chain.id,
                    res.seq_id,
                    atom.coord.x,
                    atom.coord.y,
                    atom.coord.z,
                    1.00,
                    0.00,
                    atom.element,
                )?;
                serial += 1;
            }
        }
        writeln!(
            out,
            "TER   {:>5}      {:>3} {}{:>4}",
            serial,
            chain.residues.last().map(|r| r.name.as_str()).unwrap_or("UNK"),
            chain.id,
            chain.residues.last().map(|r| r.seq_id).unwrap_or(0),
        )?;
        serial += 1;
    }
    writeln!(out, "END")?;
    Ok(())
}

/// Write structure to a PDB string.
pub fn to_pdb_string(struc: &Structure) -> String {
    let mut buf = Vec::new();
    write_pdb(struc, &mut buf).expect("write to vec failed");
    String::from_utf8(buf).expect("invalid utf8")
}
