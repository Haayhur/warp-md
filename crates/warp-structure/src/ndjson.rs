use serde::Serialize;
use std::time::Duration;

#[derive(Debug, Clone, Copy)]
pub struct NdjsonEmitter {
    enabled: bool,
}

impl NdjsonEmitter {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn disabled() -> Self {
        Self { enabled: false }
    }

    pub fn enabled() -> Self {
        Self { enabled: true }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn emit<T: Serialize>(&self, value: &T) {
        if !self.enabled {
            return;
        }
        if let Ok(line) = serde_json::to_string(value) {
            eprintln!("{line}");
        }
    }
}

pub fn duration_ms(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}
