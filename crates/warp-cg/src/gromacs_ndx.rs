use std::path::Path;

use anyhow::{anyhow, Context, Result};

use crate::trajectory::BeadMapping;

pub fn read_gromacs_ndx_mapping(path: impl AsRef<Path>) -> Result<BeadMapping> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read Gromacs NDX mapping '{}'", path.display()))?;
    parse_gromacs_ndx_mapping(&text)
        .with_context(|| format!("failed to parse Gromacs NDX mapping '{}'", path.display()))
}

pub fn parse_gromacs_ndx_mapping(text: &str) -> Result<BeadMapping> {
    let mut bead_names = Vec::new();
    let mut atom_indices = Vec::<Vec<usize>>::new();
    let mut current_section: Option<usize> = None;

    for (line_idx, raw_line) in text.lines().enumerate() {
        let line_number = line_idx + 1;
        let line = raw_line
            .split_once(';')
            .map(|(code, _)| code)
            .unwrap_or(raw_line)
            .trim();
        if line.is_empty() {
            continue;
        }
        if let Some(section_name) = parse_section_name(line) {
            if section_name.is_empty() {
                return Err(anyhow!("empty NDX section name at line {line_number}"));
            }
            bead_names.push(section_name.to_string());
            atom_indices.push(Vec::new());
            current_section = Some(atom_indices.len() - 1);
            continue;
        }

        let section_idx = current_section.ok_or_else(|| {
            anyhow!("NDX atom index line appears before first section at line {line_number}")
        })?;
        for token in line.split_whitespace() {
            let atom_id = token.parse::<usize>().map_err(|_| {
                anyhow!("invalid NDX atom id '{token}' at line {line_number}; expected integer")
            })?;
            if atom_id == 0 {
                return Err(anyhow!(
                    "invalid NDX atom id 0 at line {line_number}; Gromacs NDX ids are 1-based"
                ));
            }
            atom_indices[section_idx].push(atom_id - 1);
        }
    }

    if bead_names.is_empty() {
        return Err(anyhow!("Gromacs NDX mapping contains no bead sections"));
    }
    for (idx, atoms) in atom_indices.iter().enumerate() {
        if atoms.is_empty() {
            return Err(anyhow!(
                "Gromacs NDX mapping section '{}' contains no atoms",
                bead_names[idx]
            ));
        }
    }

    Ok(BeadMapping {
        bead_names,
        atom_indices,
    })
}

fn parse_section_name(line: &str) -> Option<&str> {
    let stripped = line.strip_prefix('[')?.strip_suffix(']')?.trim();
    Some(stripped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_gromacs_ndx_mapping_preserves_split_mapping() {
        let mapping = parse_gromacs_ndx_mapping(
            "\
[ B1 ]\n\
1 2 3\n\
\n\
[ B2 ] ; shared atom\n\
3 4\n",
        )
        .unwrap();

        assert_eq!(mapping.bead_names, vec!["B1", "B2"]);
        assert_eq!(mapping.atom_indices, vec![vec![0, 1, 2], vec![2, 3]]);
    }

    #[test]
    fn parse_gromacs_ndx_mapping_rejects_zero_atom_id() {
        let err = parse_gromacs_ndx_mapping("[ B1 ]\n0\n").unwrap_err();

        assert!(err.to_string().contains("1-based"));
    }
}
