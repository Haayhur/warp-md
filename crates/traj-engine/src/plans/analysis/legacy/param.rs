use std::collections::HashMap;
use std::fs;
use std::path::Path;

use traj_core::error::{TrajError, TrajResult};

#[derive(Debug, Clone)]
pub struct LegacyParam {
    sections: HashMap<String, Vec<String>>,
}

impl LegacyParam {
    pub fn from_file(path: impl AsRef<Path>) -> TrajResult<Self> {
        let text = fs::read_to_string(&path).map_err(|err| {
            TrajError::Parse(format!(
                "failed to read legacy param file {}: {err}",
                path.as_ref().display()
            ))
        })?;
        let mut sections: HashMap<String, Vec<String>> = HashMap::new();
        let mut current: Option<String> = None;
        for raw in text.lines() {
            let mut line = raw.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(idx) = line.find('#') {
                line = line[..idx].trim();
            }
            if let Some(idx) = line.find('!') {
                line = line[..idx].trim();
            }
            if line.is_empty() {
                continue;
            }
            if line.starts_with('[') && line.contains(']') {
                let end = line.find(']').unwrap();
                let name = line[1..end].trim().to_ascii_lowercase();
                current = Some(name);
                continue;
            }
            let section = current
                .clone()
                .ok_or_else(|| TrajError::Parse("param entry before section header".into()))?;
            sections.entry(section).or_default().push(line.to_string());
        }
        Ok(Self { sections })
    }

    pub fn get(&self, key: &str) -> Option<&[String]> {
        self.sections
            .get(&key.to_ascii_lowercase())
            .map(|v| v.as_slice())
    }

    pub fn xtc_files(&self) -> Option<Vec<String>> {
        let items = self.get("xtcfiles")?;
        let mut out = Vec::new();
        if items.is_empty() {
            return Some(out);
        }
        let count: usize = items[0].split_whitespace().next()?.parse().ok()?;
        for entry in items.iter().skip(1).take(count) {
            out.push(entry.to_string());
        }
        Some(out)
    }
}
