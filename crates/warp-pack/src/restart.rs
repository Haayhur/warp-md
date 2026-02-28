use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::error::{PackError, PackResult};
use crate::geom::Vec3;

#[derive(Clone, Copy, Debug)]
pub(crate) struct RestartEntry {
    pub(crate) center: Vec3,
    pub(crate) euler: [f32; 3],
}

pub(crate) fn read_restart(path: &Path) -> PackResult<Vec<RestartEntry>> {
    let file = File::open(path).map_err(|err| {
        PackError::Invalid(format!(
            "failed to open restart file {}: {err}",
            path.display()
        ))
    })?;
    let reader = BufReader::new(file);
    let mut entries = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.map_err(|err| {
            PackError::Invalid(format!(
                "failed to read restart file {}: {err}",
                path.display()
            ))
        })?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        let mut values = Vec::with_capacity(6);
        for token in line.split_whitespace() {
            let value: f32 = token.parse().map_err(|_| {
                PackError::Invalid(format!(
                    "invalid number in restart file {} on line {}",
                    path.display(),
                    line_no + 1
                ))
            })?;
            values.push(value);
        }
        if values.len() < 6 {
            return Err(PackError::Invalid(format!(
                "restart file {} line {} must have 6 values",
                path.display(),
                line_no + 1
            )));
        }
        entries.push(RestartEntry {
            center: Vec3::new(values[0], values[1], values[2]),
            euler: [values[3], values[4], values[5]],
        });
    }
    Ok(entries)
}

pub(crate) fn write_restart(path: &Path, entries: &[RestartEntry]) -> PackResult<()> {
    let mut file = File::create(path).map_err(|err| {
        PackError::Invalid(format!(
            "failed to write restart file {}: {err}",
            path.display()
        ))
    })?;
    for entry in entries {
        writeln!(
            file,
            "{:>23.16e} {:>23.16e} {:>23.16e} {:>23.16e} {:>23.16e} {:>23.16e}",
            entry.center.x,
            entry.center.y,
            entry.center.z,
            entry.euler[0],
            entry.euler[1],
            entry.euler[2]
        )
        .map_err(|err| {
            PackError::Invalid(format!(
                "failed to write restart file {}: {err}",
                path.display()
            ))
        })?;
    }
    Ok(())
}
