use std::collections::HashMap;
use std::io::BufRead;

use crate::elements::{infer_element_from_atom_name, normalize_element};
use crate::error::{TrajError, TrajResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PdbRecordKind {
    Atom,
    HetAtom,
}

#[derive(Clone, Debug)]
pub struct PdbAtom {
    pub record_kind: PdbRecordKind,
    pub serial: i32,
    pub name: String,
    pub resname: String,
    pub chain: char,
    pub resid: i32,
    pub element: String,
    pub segid: String,
    pub position: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct PdbParseOptions {
    pub include_conect: bool,
    pub non_standard_conect: bool,
    pub include_ter: bool,
    pub strict: bool,
    pub only_first_model: bool,
}

impl Default for PdbParseOptions {
    fn default() -> Self {
        Self {
            include_conect: true,
            non_standard_conect: false,
            include_ter: true,
            strict: true,
            only_first_model: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PdbParseResult {
    pub atoms: Vec<PdbAtom>,
    pub bonds: Vec<(usize, usize)>,
    pub ter_after: Vec<usize>,
}

pub fn parse_pdb_reader<R: BufRead>(
    reader: R,
    options: &PdbParseOptions,
) -> TrajResult<PdbParseResult> {
    let mut atoms = Vec::new();
    let mut serial_map: HashMap<i32, usize> = HashMap::new();
    let mut conect_lines = Vec::new();
    let mut ter_after = Vec::new();
    let mut last_atom_idx: Option<usize> = None;
    let mut in_model = false;
    let mut saw_model = false;

    for line in reader.lines() {
        let line = line?;
        if line.starts_with("MODEL") {
            if options.only_first_model && saw_model {
                break;
            }
            saw_model = true;
            in_model = true;
            continue;
        }
        if line.starts_with("ENDMDL") {
            if options.only_first_model && saw_model {
                break;
            }
            if saw_model {
                in_model = false;
            }
            continue;
        }
        if saw_model && !in_model {
            continue;
        }
        if line.starts_with("TER") {
            if options.include_ter {
                if let Some(idx) = last_atom_idx {
                    ter_after.push(idx);
                }
            }
            continue;
        }
        if line.starts_with("CONECT") {
            if options.include_conect {
                conect_lines.push(line);
            }
            continue;
        }
        if !(line.starts_with("ATOM") || line.starts_with("HETATM")) {
            continue;
        }
        let alt_loc = line.chars().nth(16).unwrap_or(' ');
        if alt_loc != ' ' && alt_loc != 'A' {
            continue;
        }

        let record_kind = if line.starts_with("HETATM") {
            PdbRecordKind::HetAtom
        } else {
            PdbRecordKind::Atom
        };
        let serial = parse_int_opt(slice_trim_opt(&line, 6, 11), "serial", options.strict)?
            .unwrap_or((atoms.len() + 1) as i32);
        let name = slice_required(&line, 12, 16, "name", options.strict)?;
        let resname = slice_required(&line, 17, 20, "resname", options.strict)?;
        let chain = slice_char(&line, 21).unwrap_or(' ');
        let resid =
            parse_int_opt(slice_trim_opt(&line, 22, 26), "resid", options.strict)?.unwrap_or(1);
        let x = parse_float(slice_trim_opt(&line, 30, 38), "x")?;
        let y = parse_float(slice_trim_opt(&line, 38, 46), "y")?;
        let z = parse_float(slice_trim_opt(&line, 46, 54), "z")?;
        let element_str = slice_trim_opt(&line, 76, 78).unwrap_or("");
        let segid = slice_required(&line, 72, 76, "segid", false)?;
        let element = normalize_element(element_str)
            .or_else(|| infer_element_from_atom_name(name.trim()))
            .unwrap_or_else(|| "".to_string());

        serial_map.insert(serial, atoms.len());
        atoms.push(PdbAtom {
            record_kind,
            serial,
            name,
            resname,
            chain,
            resid,
            element,
            segid,
            position: [x, y, z],
        });
        last_atom_idx = Some(atoms.len() - 1);
    }

    if atoms.is_empty() {
        return Err(TrajError::Parse("no atoms found in PDB".into()));
    }
    let bonds = if options.include_conect {
        parse_conect(&conect_lines, &serial_map, options.non_standard_conect)
    } else {
        Vec::new()
    };
    ter_after.sort_unstable();
    ter_after.dedup();
    Ok(PdbParseResult {
        atoms,
        bonds,
        ter_after,
    })
}

#[derive(Clone, Debug)]
pub struct GroAtom {
    pub resid: i32,
    pub resname: String,
    pub name: String,
    pub element: String,
    pub position: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct GroParseResult {
    pub atoms: Vec<GroAtom>,
}

pub fn parse_gro_reader<R: BufRead>(mut reader: R, strict: bool) -> TrajResult<GroParseResult> {
    let mut line = String::new();
    line.clear();
    reader.read_line(&mut line)?;
    line.clear();
    reader.read_line(&mut line)?;
    let n_atoms: usize = line
        .trim()
        .parse()
        .map_err(|_| TrajError::Parse(format!("invalid atom count '{}'", line.trim())))?;

    let mut atoms = Vec::with_capacity(n_atoms);
    for _ in 0..n_atoms {
        line.clear();
        reader.read_line(&mut line)?;
        if line.len() < 20 {
            if strict {
                return Err(TrajError::Parse("GRO atom line too short".into()));
            }
            continue;
        }
        let (resid, resname, name, x, y, z) = if line.len() >= 44 {
            let resid = slice_trim_opt(&line, 0, 5)
                .and_then(|v| v.parse::<i32>().ok())
                .unwrap_or(1);
            let resname = slice_trim(&line, 5, 10)?;
            let name = slice_trim(&line, 10, 15)?;
            let x = parse_float(slice_trim_opt(&line, 20, 28), "x")?;
            let y = parse_float(slice_trim_opt(&line, 28, 36), "y")?;
            let z = parse_float(slice_trim_opt(&line, 36, 44), "z")?;
            (resid, resname, name, x, y, z)
        } else {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 6 {
                if strict {
                    return Err(TrajError::Parse("GRO atom line too short".into()));
                }
                continue;
            }
            let resid = parts[0].parse::<i32>().unwrap_or(1);
            let resname = parts[1].to_string();
            let name = parts[2].to_string();
            let x: f32 = parts[3]
                .parse()
                .map_err(|_| TrajError::Parse("bad gro x".into()))?;
            let y: f32 = parts[4]
                .parse()
                .map_err(|_| TrajError::Parse("bad gro y".into()))?;
            let z: f32 = parts[5]
                .parse()
                .map_err(|_| TrajError::Parse("bad gro z".into()))?;
            (resid, resname, name, x, y, z)
        };
        let element = infer_element_from_atom_name(&name).unwrap_or_else(|| "".to_string());
        atoms.push(GroAtom {
            resid,
            resname,
            name,
            element,
            position: [x, y, z],
        });
    }

    if atoms.is_empty() {
        return Err(TrajError::Parse("no atoms found in GRO".into()));
    }

    Ok(GroParseResult { atoms })
}

fn slice_trim_opt(line: &str, start: usize, end: usize) -> Option<&str> {
    if line.len() < start {
        return None;
    }
    let end = end.min(line.len());
    Some(line[start..end].trim())
}

fn slice_trim(line: &str, start: usize, end: usize) -> TrajResult<String> {
    slice_trim_opt(line, start, end)
        .map(|s| s.to_string())
        .ok_or_else(|| TrajError::Parse("line too short".into()))
}

fn slice_required(
    line: &str,
    start: usize,
    end: usize,
    label: &str,
    strict: bool,
) -> TrajResult<String> {
    match slice_trim_opt(line, start, end) {
        Some(value) => Ok(value.to_string()),
        None => {
            if strict {
                Err(TrajError::Parse(format!("missing {label} field")))
            } else {
                Ok(String::new())
            }
        }
    }
}

fn slice_char(line: &str, idx: usize) -> Option<char> {
    line.chars().nth(idx)
}

fn parse_int_opt(value: Option<&str>, label: &str, strict: bool) -> TrajResult<Option<i32>> {
    let Some(raw) = value else {
        if strict {
            return Err(TrajError::Parse(format!("missing {label} field")));
        }
        return Ok(None);
    };
    if raw.is_empty() {
        if strict {
            return Err(TrajError::Parse(format!("missing {label} field")));
        }
        return Ok(None);
    }
    raw.parse::<i32>()
        .map(Some)
        .map_err(|_| TrajError::Parse(format!("invalid {label} '{raw}'")))
        .or_else(|err| if strict { Err(err) } else { Ok(None) })
}

fn parse_float(value: Option<&str>, label: &str) -> TrajResult<f32> {
    let Some(raw) = value else {
        return Err(TrajError::Parse(format!("missing {label} field")));
    };
    raw.parse::<f32>()
        .map_err(|_| TrajError::Parse(format!("invalid {label} '{raw}'")))
}

fn parse_conect(
    lines: &[String],
    serial_map: &HashMap<i32, usize>,
    non_standard_conect: bool,
) -> Vec<(usize, usize)> {
    let mut bonds = Vec::new();
    for line in lines {
        let mut entries = Vec::new();
        if non_standard_conect {
            entries.push(slice_int(line, 6, 11));
            for start in [11, 16, 21, 26, 31] {
                entries.push(slice_int(line, start, start + 5));
            }
        } else {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                for part in &parts[1..] {
                    if let Ok(v) = part.parse::<i32>() {
                        entries.push(Some(v));
                    }
                }
            }
        }
        if entries.is_empty() {
            continue;
        }
        let from = entries[0].and_then(|id| serial_map.get(&id).copied());
        if let Some(a) = from {
            for entry in entries.iter().skip(1) {
                let Some(b_id) = entry.and_then(|id| serial_map.get(&id).copied()) else {
                    continue;
                };
                let (i, j) = if a <= b_id { (a, b_id) } else { (b_id, a) };
                if i != j {
                    bonds.push((i, j));
                }
            }
        }
    }
    bonds.sort_unstable();
    bonds.dedup();
    bonds
}

fn slice_int(line: &str, start: usize, end: usize) -> Option<i32> {
    slice_trim_opt(line, start, end).and_then(|s| s.parse::<i32>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn multi_model_fixture() -> &'static str {
        "MODEL        1\n\
ATOM      1  N   ALA A   1       1.000   0.000   0.000  1.00 20.00           N\n\
ENDMDL\n\
MODEL        2\n\
ATOM      1  N   ALA A   1       2.000   0.000   0.000  1.00 20.00           N\n\
ENDMDL\n"
    }

    #[test]
    fn parse_pdb_reader_defaults_to_first_model_only() {
        let parsed =
            parse_pdb_reader(Cursor::new(multi_model_fixture()), &PdbParseOptions::default())
                .expect("parse");
        assert_eq!(parsed.atoms.len(), 1);
        assert!((parsed.atoms[0].position[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn parse_pdb_reader_can_include_all_models() {
        let options = PdbParseOptions {
            include_conect: false,
            non_standard_conect: false,
            include_ter: false,
            strict: true,
            only_first_model: false,
        };
        let parsed = parse_pdb_reader(Cursor::new(multi_model_fixture()), &options).expect("parse");
        assert_eq!(parsed.atoms.len(), 2);
        assert!((parsed.atoms[0].position[0] - 1.0).abs() < 1e-6);
        assert!((parsed.atoms[1].position[0] - 2.0).abs() < 1e-6);
    }
}
