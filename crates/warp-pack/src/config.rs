use serde::{Deserialize, Serialize};

use crate::constraints::ConstraintSpec;
use crate::error::{PackError, PackResult};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackConfig {
    #[serde(rename = "box")]
    pub box_: BoxSpec,
    pub structures: Vec<StructureSpec>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub max_attempts: Option<usize>,
    #[serde(default)]
    pub min_distance: Option<f32>,
    #[serde(default)]
    pub filetype: Option<String>,
    #[serde(default)]
    pub add_box_sides: bool,
    #[serde(default)]
    pub add_box_sides_fix: Option<f32>,
    #[serde(default)]
    pub add_amber_ter: bool,
    #[serde(default)]
    pub amber_ter_preserve: bool,
    #[serde(default)]
    pub hexadecimal_indices: bool,
    #[serde(default)]
    pub ignore_conect: bool,
    #[serde(default)]
    pub non_standard_conect: bool,
    #[serde(default)]
    pub pbc: bool,
    #[serde(default)]
    pub pbc_min: Option<[f32; 3]>,
    #[serde(default)]
    pub pbc_max: Option<[f32; 3]>,
    #[serde(default)]
    pub maxit: Option<usize>,
    #[serde(default)]
    pub nloop: Option<usize>,
    #[serde(default)]
    pub nloop0: Option<usize>,
    #[serde(default = "default_true")]
    pub avoid_overlap: bool,
    #[serde(default)]
    pub packall: bool,
    #[serde(default)]
    pub check: bool,
    #[serde(default)]
    pub sidemax: Option<f32>,
    #[serde(default)]
    pub discale: Option<f32>,
    #[serde(default)]
    pub precision: Option<f32>,
    #[serde(default)]
    pub chkgrad: bool,
    #[serde(default)]
    pub iprint1: Option<i32>,
    #[serde(default)]
    pub iprint2: Option<i32>,
    #[serde(default)]
    pub gencan_maxit: Option<usize>,
    #[serde(default)]
    pub gencan_step: Option<f32>,
    #[serde(default)]
    pub use_short_tol: bool,
    #[serde(default)]
    pub short_tol_dist: Option<f32>,
    #[serde(default)]
    pub short_tol_scale: Option<f32>,
    #[serde(default)]
    pub movefrac: Option<f32>,
    #[serde(default)]
    pub movebadrandom: bool,
    #[serde(default)]
    pub disable_movebad: bool,
    #[serde(default)]
    pub maxmove: Option<usize>,
    #[serde(default)]
    pub randominitialpoint: bool,
    #[serde(default)]
    pub fbins: Option<f32>,
    #[serde(default)]
    pub writeout: Option<f32>,
    #[serde(default)]
    pub writebad: bool,
    #[serde(default)]
    pub restart_from: Option<String>,
    #[serde(default)]
    pub restart_to: Option<String>,
    #[serde(default)]
    pub relax_steps: Option<usize>,
    #[serde(default)]
    pub relax_step: Option<f32>,
    #[serde(default, alias = "writecrd")]
    pub write_crd: Option<String>,
    #[serde(default)]
    pub output: Option<OutputSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoxSpec {
    pub size: [f32; 3],
    #[serde(default = "default_shape")]
    pub shape: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StructureSpec {
    pub path: String,
    #[serde(default = "default_count")]
    pub count: usize,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub topology: Option<String>,
    #[serde(default)]
    pub restart_from: Option<String>,
    #[serde(default)]
    pub restart_to: Option<String>,
    #[serde(default)]
    pub fixed_eulers: Option<Vec<[f32; 3]>>,
    #[serde(default)]
    pub chain: Option<String>,
    #[serde(default)]
    pub changechains: bool,
    #[serde(default)]
    pub segid: Option<String>,
    #[serde(default = "default_true")]
    pub connect: bool,
    #[serde(default)]
    #[serde(alias = "filetype")]
    pub format: Option<String>,
    #[serde(default = "default_true")]
    pub rotate: bool,
    #[serde(default)]
    pub fixed: bool,
    #[serde(default)]
    pub positions: Option<Vec<[f32; 3]>>,
    #[serde(default)]
    pub translate: Option<[f32; 3]>,
    #[serde(default = "default_true")]
    pub center: bool,
    #[serde(default)]
    pub min_distance: Option<f32>,
    #[serde(default)]
    pub resnumbers: Option<i32>,
    #[serde(default)]
    pub maxmove: Option<usize>,
    #[serde(default)]
    pub nloop: Option<usize>,
    #[serde(default)]
    pub nloop0: Option<usize>,
    #[serde(default)]
    pub constraints: Vec<ConstraintSpec>,
    #[serde(default)]
    pub radius: Option<f32>,
    #[serde(default)]
    pub fscale: Option<f32>,
    #[serde(default)]
    pub short_radius: Option<f32>,
    #[serde(default)]
    pub short_radius_scale: Option<f32>,
    #[serde(default)]
    pub atom_overrides: Vec<AtomOverride>,
    #[serde(default)]
    pub atom_constraints: Vec<AtomConstraintSpec>,
    #[serde(default)]
    pub rot_bounds: Option<[[f32; 2]; 3]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputSpec {
    pub path: String,
    pub format: String,
    #[serde(default)]
    pub scale: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AtomOverride {
    pub indices: Vec<usize>,
    #[serde(default)]
    pub radius: Option<f32>,
    #[serde(default)]
    pub fscale: Option<f32>,
    #[serde(default)]
    pub short_radius: Option<f32>,
    #[serde(default)]
    pub short_radius_scale: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AtomConstraintSpec {
    pub indices: Vec<usize>,
    pub constraint: ConstraintSpec,
}

fn default_shape() -> String {
    "orthorhombic".to_string()
}

fn default_count() -> usize {
    1
}

fn default_true() -> bool {
    true
}

impl PackConfig {
    pub fn normalized(&self) -> PackResult<PackConfig> {
        let mut cfg = self.clone();
        cfg.validate()?;
        if let (Some(min), Some(max)) = (cfg.pbc_min, cfg.pbc_max) {
            cfg.pbc = true;
            for i in 0..3 {
                cfg.box_.size[i] = max[i] - min[i];
            }
        }
        if cfg.seed.is_none() {
            cfg.seed = Some(1_234_567);
        }
        if cfg.max_attempts.is_none() {
            cfg.max_attempts = Some(100000);
        }
        if cfg.min_distance.is_none() {
            cfg.min_distance = Some(2.0);
        }
        let ntypes = cfg.structures.len().max(1);
        if cfg.nloop0.is_none() {
            cfg.nloop0 = Some(20 * ntypes);
        }
        if cfg.nloop.is_none() {
            cfg.nloop = Some(200 * ntypes);
        }
        if cfg.precision.is_none() {
            cfg.precision = Some(1.0e-2);
        }
        if cfg.movefrac.is_none() {
            cfg.movefrac = Some(0.05);
        }
        if cfg.discale.is_none() {
            cfg.discale = Some(1.1);
        }
        if cfg.fbins.is_none() {
            cfg.fbins = Some(3.0f32.sqrt());
        }
        if cfg.relax_steps.is_none() {
            cfg.relax_steps = Some(0);
        }
        if cfg.relax_step.is_none() {
            cfg.relax_step = Some(0.5);
        }
        if cfg.gencan_maxit.is_none() {
            cfg.gencan_maxit = Some(cfg.maxit.unwrap_or(20));
        }
        Ok(cfg)
    }

    pub fn validate(&self) -> PackResult<()> {
        if self.structures.is_empty() {
            return Err(PackError::Invalid("structures list is empty".into()));
        }
        if self.pbc_min.is_some() ^ self.pbc_max.is_some() {
            return Err(PackError::Invalid(
                "pbc_min and pbc_max must be provided together".into(),
            ));
        }
        if let (Some(min), Some(max)) = (self.pbc_min, self.pbc_max) {
            for i in 0..3 {
                if max[i] <= min[i] {
                    return Err(PackError::Invalid(
                        "pbc_max must be greater than pbc_min".into(),
                    ));
                }
            }
        }
        if self.box_.size.iter().any(|&v| v <= 0.0) {
            return Err(PackError::Invalid("box size must be positive".into()));
        }
        if self.box_.shape.to_lowercase() != "orthorhombic" {
            return Err(PackError::Invalid(
                "only orthorhombic boxes are supported".into(),
            ));
        }
        if let Some(sidemax) = self.sidemax {
            if sidemax <= 0.0 {
                return Err(PackError::Invalid("sidemax must be > 0".into()));
            }
            if self.box_.size.iter().any(|&v| v > sidemax) {
                return Err(PackError::Invalid("box size exceeds sidemax".into()));
            }
        }
        if let Some(maxit) = self.maxit {
            if maxit == 0 {
                return Err(PackError::Invalid("maxit must be > 0".into()));
            }
        }
        if let Some(nloop) = self.nloop {
            if nloop == 0 {
                return Err(PackError::Invalid("nloop must be > 0".into()));
            }
        }
        if let Some(nloop0) = self.nloop0 {
            if nloop0 == 0 {
                return Err(PackError::Invalid("nloop0 must be > 0".into()));
            }
        }
        if let Some(discale) = self.discale {
            if discale <= 0.0 {
                return Err(PackError::Invalid(
                    "discale must be a positive scale".into(),
                ));
            }
        }
        if let Some(precision) = self.precision {
            if precision <= 0.0 {
                return Err(PackError::Invalid(
                    "precision must be a positive tolerance".into(),
                ));
            }
        }
        if let Some(gencan_step) = self.gencan_step {
            if gencan_step <= 0.0 {
                return Err(PackError::Invalid(
                    "gencan_step must be a positive value".into(),
                ));
            }
        }
        if let Some(gencan_maxit) = self.gencan_maxit {
            if gencan_maxit == 0 {
                return Err(PackError::Invalid("gencan_maxit must be > 0".into()));
            }
        }
        if let Some(short_tol_dist) = self.short_tol_dist {
            if short_tol_dist <= 0.0 {
                return Err(PackError::Invalid("short_tol_dist must be positive".into()));
            }
            if let Some(tol) = self.min_distance {
                if short_tol_dist > tol {
                    return Err(PackError::Invalid(
                        "short_tol_dist must be <= tolerance".into(),
                    ));
                }
            }
        }
        if let Some(short_tol_scale) = self.short_tol_scale {
            if short_tol_scale <= 0.0 {
                return Err(PackError::Invalid(
                    "short_tol_scale must be positive".into(),
                ));
            }
        }
        if let Some(movefrac) = self.movefrac {
            if movefrac <= 0.0 || movefrac > 1.0 {
                return Err(PackError::Invalid("movefrac must be in (0, 1]".into()));
            }
        }
        if let Some(maxmove) = self.maxmove {
            if maxmove == 0 {
                return Err(PackError::Invalid("maxmove must be > 0".into()));
            }
        }
        if let Some(fbins) = self.fbins {
            if fbins < 1.0 {
                return Err(PackError::Invalid("fbins must be >= 1.0".into()));
            }
        }
        if let Some(writeout) = self.writeout {
            if writeout <= 0.0 {
                return Err(PackError::Invalid(
                    "writeout must be a positive number of seconds".into(),
                ));
            }
        }
        if let Some(relax_step) = self.relax_step {
            if relax_step <= 0.0 {
                return Err(PackError::Invalid(
                    "relax_step must be a positive value".into(),
                ));
            }
        }
        for s in &self.structures {
            if s.count == 0 {
                return Err(PackError::Invalid("structure count must be > 0".into()));
            }
            if let (Some(pos), Some(eul)) = (&s.positions, &s.fixed_eulers) {
                if pos.len() != eul.len() {
                    return Err(PackError::Invalid(
                        "fixed_eulers length must match positions length".into(),
                    ));
                }
            }
            if let Some(bounds) = s.rot_bounds {
                for axis in 0..3 {
                    if bounds[axis][1] < bounds[axis][0] {
                        return Err(PackError::Invalid(
                            "rotation bounds must satisfy min <= max".into(),
                        ));
                    }
                }
            }
            if let Some(mode) = s.resnumbers {
                if !(0..=3).contains(&mode) {
                    return Err(PackError::Invalid(
                        "resnumbers must be 0, 1, 2, or 3".into(),
                    ));
                }
            }
            if let Some(maxmove) = s.maxmove {
                if maxmove == 0 {
                    return Err(PackError::Invalid("structure maxmove must be > 0".into()));
                }
            }
            if let Some(nloop) = s.nloop {
                if nloop == 0 {
                    return Err(PackError::Invalid("structure nloop must be > 0".into()));
                }
            }
            if let Some(nloop0) = s.nloop0 {
                if nloop0 == 0 {
                    return Err(PackError::Invalid("structure nloop0 must be > 0".into()));
                }
            }
            if let Some(chain) = &s.chain {
                if chain.is_empty() {
                    return Err(PackError::Invalid("chain cannot be empty".into()));
                }
                if s.changechains {
                    return Err(PackError::Invalid(
                        "changechains and chain are not compatible".into(),
                    ));
                }
            }
            if let Some(segid) = &s.segid {
                if segid.is_empty() {
                    return Err(PackError::Invalid("segid cannot be empty".into()));
                }
            }
            if s.fixed && s.count > 1 {
                if let Some(positions) = &s.positions {
                    if positions.len() != s.count {
                        return Err(PackError::Invalid(
                            "fixed positions length must match count".into(),
                        ));
                    }
                } else if s.restart_from.is_none() && self.restart_from.is_none() {
                    return Err(PackError::Invalid(
                        "fixed structures with count>1 require positions or restart_from".into(),
                    ));
                }
            }
            if s.positions.is_some() && s.restart_from.is_some() {
                return Err(PackError::Invalid(
                    "structure cannot use positions and restart_from together".into(),
                ));
            }
            for constraint in &s.constraints {
                constraint.validate()?;
            }
            for atom_constraint in &s.atom_constraints {
                if atom_constraint.indices.is_empty() {
                    return Err(PackError::Invalid(
                        "atom constraint indices must be non-empty".into(),
                    ));
                }
                if atom_constraint.indices.iter().any(|&idx| idx == 0) {
                    return Err(PackError::Invalid(
                        "atom constraint indices must be 1-based".into(),
                    ));
                }
                atom_constraint.constraint.validate()?;
            }
        }
        Ok(())
    }
}
