use std::collections::HashMap;

#[derive(Debug, Default, Clone)]
pub struct StringInterner {
    map: HashMap<String, u32>,
    values: Vec<String>,
}

impl StringInterner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern(&mut self, value: &str) -> u32 {
        if let Some(id) = self.map.get(value) {
            return *id;
        }
        let id = self.values.len() as u32;
        self.values.push(value.to_string());
        self.map.insert(value.to_string(), id);
        id
    }

    pub fn intern_upper(&mut self, value: &str) -> u32 {
        let upper = value.to_ascii_uppercase();
        self.intern(&upper)
    }

    pub fn resolve(&self, id: u32) -> Option<&str> {
        self.values.get(id as usize).map(|s| s.as_str())
    }
}
