use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::{
    AtomConstraintSpec, AtomOverride, BoxSpec, OutputSpec, PackConfig, StructureSpec,
};
use crate::constraints::{ConstraintMode, ConstraintSpec, ShapeSpec};
use crate::error::{PackError, PackResult};

pub fn parse_packmol_inp(input: &str) -> PackResult<PackConfig> {
    let mut cfg = PackConfig {
        box_: BoxSpec {
            size: [1.0, 1.0, 1.0],
            shape: "orthorhombic".into(),
        },
        structures: Vec::new(),
        seed: None,
        max_attempts: None,
        min_distance: None,
        filetype: None,
        add_box_sides: false,
        add_box_sides_fix: None,
        add_amber_ter: false,
        amber_ter_preserve: false,
        hexadecimal_indices: false,
        ignore_conect: false,
        non_standard_conect: true,
        pbc: false,
        pbc_min: None,
        pbc_max: None,
        maxit: None,
        nloop: None,
        nloop0: None,
        avoid_overlap: true,
        packall: false,
        check: false,
        sidemax: None,
        discale: None,
        precision: None,
        chkgrad: false,
        iprint1: None,
        iprint2: None,
        gencan_maxit: None,
        gencan_step: None,
        use_short_tol: false,
        short_tol_dist: None,
        short_tol_scale: None,
        movefrac: None,
        movebadrandom: false,
        disable_movebad: false,
        maxmove: None,
        randominitialpoint: false,
        fbins: None,
        writeout: None,
        writebad: false,
        restart_from: None,
        restart_to: None,
        relax_steps: None,
        relax_step: None,
        write_crd: None,
        output: None,
    };

    let mut current: Option<StructureSpec> = None;
    let mut in_atoms = false;
    let mut atom_selection: Vec<usize> = Vec::new();
    let mut bounds_min: Option<[f32; 3]> = None;
    let mut bounds_max: Option<[f32; 3]> = None;

    for raw_line in input.lines() {
        let line = strip_comments(raw_line);
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }
        let kw = tokens[0].to_lowercase();

        if kw == "structure" {
            if tokens.len() < 2 {
                return Err(PackError::Parse("structure requires a path".into()));
            }
            if let Some(s) = current.take() {
                cfg.structures.push(s);
            }
            current = Some(default_structure(tokens[1].to_string()));
            in_atoms = false;
            atom_selection.clear();
            continue;
        }
        if kw == "end" && tokens.get(1).map(|v| v.to_lowercase()) == Some("structure".into()) {
            if let Some(s) = current.take() {
                cfg.structures.push(s);
            }
            in_atoms = false;
            atom_selection.clear();
            continue;
        }
        if kw == "atoms" {
            in_atoms = true;
            atom_selection = parse_atom_selection(&tokens[1..])?;
            continue;
        }
        if kw == "end" && tokens.get(1).map(|v| v.to_lowercase()) == Some("atoms".into()) {
            in_atoms = false;
            atom_selection.clear();
            continue;
        }

        if let Some(spec) = current.as_mut() {
            if in_atoms {
                handle_atoms_block_kw(spec, &kw, &tokens, &atom_selection)?;
                continue;
            }
            if handle_structure_kw(spec, &kw, &tokens)? {
                if kw == "inside" {
                    if let Some(c) = spec.constraints.last() {
                        update_bounds_from_constraint(c, &mut bounds_min, &mut bounds_max);
                    }
                }
                continue;
            }
        }

        handle_global_kw(&mut cfg, &kw, &tokens)?;
    }

    if let Some(s) = current.take() {
        cfg.structures.push(s);
    }

    if cfg.min_distance.is_none() {
        return Err(PackError::Parse("tolerance not set in input".into()));
    }
    if cfg.output.is_none() {
        return Err(PackError::Parse("output not set in input".into()));
    }
    if cfg.pbc_min.is_none() && bounds_min.is_some() && bounds_max.is_some() {
        cfg.pbc_min = bounds_min;
        cfg.pbc_max = bounds_max;
    }
    Ok(cfg)
}

fn strip_comments(line: &str) -> &str {
    let mut end = line.len();
    for (i, ch) in line.char_indices() {
        if ch == '#' || ch == '!' {
            end = i;
            break;
        }
    }
    &line[..end]
}

fn default_structure(path: String) -> StructureSpec {
    StructureSpec {
        path,
        count: 1,
        name: None,
        topology: None,
        restart_from: None,
        restart_to: None,
        fixed_eulers: None,
        chain: None,
        changechains: false,
        segid: None,
        connect: true,
        format: None,
        rotate: true,
        fixed: false,
        positions: None,
        translate: None,
        center: true,
        min_distance: None,
        resnumbers: None,
        maxmove: None,
        nloop: None,
        nloop0: None,
        constraints: Vec::new(),
        radius: None,
        fscale: None,
        short_radius: None,
        short_radius_scale: None,
        atom_overrides: Vec::new(),
        atom_constraints: Vec::new(),
        rot_bounds: None,
    }
}

fn handle_global_kw(cfg: &mut PackConfig, kw: &str, tokens: &[&str]) -> PackResult<()> {
    match kw {
        "seed" => {
            let seed = parse_i64(tokens.get(1))?;
            if seed == -1 {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                cfg.seed = Some(now);
            } else {
                cfg.seed = Some(seed.max(0) as u64);
            }
        }
        "randominitialpoint" => cfg.randominitialpoint = true,
        "check" => cfg.check = true,
        "writebad" => cfg.writebad = true,
        "ignore_conect" => cfg.ignore_conect = true,
        "non_standard_conect" => cfg.non_standard_conect = true,
        "precision" => cfg.precision = Some(parse_f32(tokens.get(1))?),
        "movefrac" => cfg.movefrac = Some(parse_f32(tokens.get(1))?),
        "movebadrandom" => cfg.movebadrandom = true,
        "disable_movebad" => cfg.disable_movebad = true,
        "chkgrad" => cfg.chkgrad = true,
        "writeout" => cfg.writeout = Some(parse_f32(tokens.get(1))?),
        "hexadecimal_indices" => cfg.hexadecimal_indices = true,
        "maxit" => {
            let val = parse_usize(tokens.get(1))?;
            cfg.maxit = Some(val);
            cfg.gencan_maxit = Some(val);
        }
        "nloop" => cfg.nloop = Some(parse_usize(tokens.get(1))?),
        "nloop0" => cfg.nloop0 = Some(parse_usize(tokens.get(1))?),
        "discale" => cfg.discale = Some(parse_f32(tokens.get(1))?),
        "sidemax" => cfg.sidemax = Some(parse_f32(tokens.get(1))?),
        "fbins" => cfg.fbins = Some(parse_f32(tokens.get(1))?),
        "add_amber_ter" => cfg.add_amber_ter = true,
        "amber_ter_preserve" => cfg.amber_ter_preserve = true,
        "avoid_overlap" => {
            cfg.avoid_overlap = tokens
                .get(1)
                .map(|v| v.eq_ignore_ascii_case("yes"))
                .unwrap_or(true);
        }
        "packall" => cfg.packall = true,
        "use_short_tol" => cfg.use_short_tol = true,
        "writecrd" => {
            cfg.write_crd = tokens.get(1).map(|v| v.to_string());
        }
        "add_box_sides" => {
            cfg.add_box_sides = true;
            if let Some(val) = tokens.get(1) {
                if let Ok(v) = val.parse::<f32>() {
                    cfg.add_box_sides_fix = Some(v);
                }
            }
        }
        "iprint1" => cfg.iprint1 = Some(parse_i32(tokens.get(1))?),
        "iprint2" => cfg.iprint2 = Some(parse_i32(tokens.get(1))?),
        "pbc" => {
            let vals: Vec<f32> = tokens[1..]
                .iter()
                .filter(|v| !v.eq_ignore_ascii_case(&"none"))
                .map(|v| parse_f32(Some(v)))
                .collect::<PackResult<Vec<_>>>()?;
            if vals.len() == 3 {
                cfg.pbc_min = Some([0.0, 0.0, 0.0]);
                cfg.pbc_max = Some([vals[0], vals[1], vals[2]]);
                cfg.pbc = true;
            } else if vals.len() >= 6 {
                cfg.pbc_min = Some([vals[0], vals[1], vals[2]]);
                cfg.pbc_max = Some([vals[3], vals[4], vals[5]]);
                cfg.pbc = true;
            } else {
                return Err(PackError::Parse("pbc requires 3 or 6 values".into()));
            }
        }
        "output" => {
            let path = tokens
                .get(1)
                .ok_or_else(|| PackError::Parse("output requires path".into()))?;
            let fmt = path.split('.').last().unwrap_or("pdb").to_lowercase();
            cfg.output = Some(OutputSpec {
                path: path.to_string(),
                format: fmt,
                scale: None,
            });
        }
        "filetype" => {
            cfg.filetype = tokens.get(1).map(|v| v.to_string());
        }
        "tolerance" => {
            cfg.min_distance = Some(parse_f32(tokens.get(1))?);
        }
        "short_tol_dist" => {
            cfg.short_tol_dist = Some(parse_f32(tokens.get(1))?);
        }
        "short_tol_scale" => {
            cfg.short_tol_scale = Some(parse_f32(tokens.get(1))?);
        }
        "restart_from" => cfg.restart_from = tokens.get(1).map(|v| v.to_string()),
        "restart_to" => cfg.restart_to = tokens.get(1).map(|v| v.to_string()),
        _ => {
            return Err(PackError::Parse(format!("unrecognized keyword: {kw}")));
        }
    }
    Ok(())
}

fn handle_structure_kw(spec: &mut StructureSpec, kw: &str, tokens: &[&str]) -> PackResult<bool> {
    match kw {
        "number" => {
            spec.count = parse_usize(tokens.get(1))?;
            Ok(true)
        }
        "fixed" => {
            if tokens.len() < 4 {
                return Err(PackError::Parse("fixed requires x y z".into()));
            }
            let x = parse_f32(tokens.get(1))?;
            let y = parse_f32(tokens.get(2))?;
            let z = parse_f32(tokens.get(3))?;
            let euler = if tokens.len() >= 7 {
                [
                    parse_f32(tokens.get(4))?,
                    parse_f32(tokens.get(5))?,
                    parse_f32(tokens.get(6))?,
                ]
            } else {
                [0.0, 0.0, 0.0]
            };
            spec.fixed = true;
            spec.positions.get_or_insert_with(Vec::new).push([x, y, z]);
            spec.fixed_eulers.get_or_insert_with(Vec::new).push(euler);
            Ok(true)
        }
        "inside" | "outside" => {
            let mode = if kw == "inside" {
                ConstraintMode::Inside
            } else {
                ConstraintMode::Outside
            };
            let constraint = parse_shape_constraint(mode, tokens)?;
            spec.constraints.push(constraint);
            Ok(true)
        }
        "over" | "above" | "below" => {
            let mode = if kw == "below" {
                ConstraintMode::Below
            } else {
                ConstraintMode::Above
            };
            let constraint = parse_shape_constraint(mode, tokens)?;
            spec.constraints.push(constraint);
            Ok(true)
        }
        "center" | "centerofmass" => {
            spec.center = true;
            Ok(true)
        }
        "translate" => {
            if tokens.len() < 4 {
                return Err(PackError::Parse("translate requires x y z".into()));
            }
            spec.translate = Some([
                parse_f32(tokens.get(1))?,
                parse_f32(tokens.get(2))?,
                parse_f32(tokens.get(3))?,
            ]);
            Ok(true)
        }
        "rotate" => {
            spec.rotate = true;
            Ok(true)
        }
        "norotate" => {
            spec.rotate = false;
            Ok(true)
        }
        "constrain_rotation" => {
            if tokens.len() < 4 {
                return Err(PackError::Parse(
                    "constrain_rotation requires axis center delta".into(),
                ));
            }
            let axis = tokens[1].to_lowercase();
            let center = parse_f32(tokens.get(2))?;
            let delta = parse_f32(tokens.get(3))?.abs();
            let deg_to_rad = |v: f32| v * std::f32::consts::PI / 180.0;
            let center = deg_to_rad(center);
            let delta = deg_to_rad(delta);
            let mut bounds = spec
                .rot_bounds
                .unwrap_or([[-std::f32::consts::PI, std::f32::consts::PI]; 3]);
            let idx = match axis.as_str() {
                "x" | "1" => 2,
                "y" | "2" => 0,
                "z" | "3" => 1,
                _ => return Err(PackError::Parse("invalid rotation axis".into())),
            };
            bounds[idx][0] = center - delta;
            bounds[idx][1] = center + delta;
            spec.rot_bounds = Some(bounds);
            Ok(true)
        }
        "resnumbers" => {
            spec.resnumbers = Some(parse_i32(tokens.get(1))?);
            Ok(true)
        }
        "maxmove" => {
            spec.maxmove = Some(parse_usize(tokens.get(1))?);
            Ok(true)
        }
        "nloop" => {
            spec.nloop = Some(parse_usize(tokens.get(1))?);
            Ok(true)
        }
        "nloop0" => {
            spec.nloop0 = Some(parse_usize(tokens.get(1))?);
            Ok(true)
        }
        "changechains" => {
            spec.changechains = true;
            Ok(true)
        }
        "chain" => {
            spec.chain = tokens.get(1).map(|v| v.to_string());
            Ok(true)
        }
        "segid" => {
            spec.segid = tokens.get(1).map(|v| v.to_string());
            Ok(true)
        }
        "connect" => {
            spec.connect = tokens
                .get(1)
                .map(|v| v.eq_ignore_ascii_case("yes"))
                .unwrap_or(true);
            Ok(true)
        }
        "restart_from" => {
            spec.restart_from = tokens.get(1).map(|v| v.to_string());
            Ok(true)
        }
        "restart_to" => {
            spec.restart_to = tokens.get(1).map(|v| v.to_string());
            Ok(true)
        }
        "filetype" => {
            spec.format = tokens.get(1).map(|v| v.to_string());
            Ok(true)
        }
        "radius" => {
            spec.radius = Some(parse_f32(tokens.get(1))?);
            Ok(true)
        }
        "fscale" => {
            spec.fscale = Some(parse_f32(tokens.get(1))?);
            Ok(true)
        }
        "short_radius" => {
            spec.short_radius = Some(parse_f32(tokens.get(1))?);
            Ok(true)
        }
        "short_radius_scale" => {
            spec.short_radius_scale = Some(parse_f32(tokens.get(1))?);
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn handle_atom_override(
    spec: &mut StructureSpec,
    kw: &str,
    tokens: &[&str],
    selection: &[usize],
) -> PackResult<()> {
    let value = parse_f32(tokens.get(1))?;
    let indices = if tokens.len() > 2 {
        tokens[2..]
            .iter()
            .map(|t| parse_usize(Some(t)))
            .collect::<PackResult<Vec<_>>>()?
    } else {
        selection.to_vec()
    };
    if indices.is_empty() {
        return Err(PackError::Parse("atoms block missing indices".into()));
    }
    let mut ov = AtomOverride {
        indices,
        radius: None,
        fscale: None,
        short_radius: None,
        short_radius_scale: None,
    };
    match kw {
        "radius" => ov.radius = Some(value),
        "fscale" => ov.fscale = Some(value),
        "short_radius" => ov.short_radius = Some(value),
        "short_radius_scale" => ov.short_radius_scale = Some(value),
        _ => return Err(PackError::Parse(format!("unknown atoms keyword {kw}"))),
    }
    spec.atom_overrides.push(ov);
    Ok(())
}

fn handle_atom_constraint(
    spec: &mut StructureSpec,
    mode: ConstraintMode,
    tokens: &[&str],
    selection: &[usize],
) -> PackResult<()> {
    if selection.is_empty() {
        return Err(PackError::Parse("atoms block missing indices".into()));
    }
    let constraint = parse_shape_constraint(mode, tokens)?;
    spec.atom_constraints.push(AtomConstraintSpec {
        indices: selection.to_vec(),
        constraint,
    });
    Ok(())
}

fn handle_atoms_block_kw(
    spec: &mut StructureSpec,
    kw: &str,
    tokens: &[&str],
    selection: &[usize],
) -> PackResult<()> {
    match kw {
        "radius" | "fscale" | "short_radius" | "short_radius_scale" => {
            handle_atom_override(spec, kw, tokens, selection)
        }
        "inside" | "outside" => {
            let mode = if kw == "inside" {
                ConstraintMode::Inside
            } else {
                ConstraintMode::Outside
            };
            handle_atom_constraint(spec, mode, tokens, selection)
        }
        "over" | "above" | "below" => {
            let mode = if kw == "below" {
                ConstraintMode::Below
            } else {
                ConstraintMode::Above
            };
            handle_atom_constraint(spec, mode, tokens, selection)
        }
        _ => Err(PackError::Parse(format!("unknown atoms keyword {kw}"))),
    }
}

fn parse_shape_constraint(mode: ConstraintMode, tokens: &[&str]) -> PackResult<ConstraintSpec> {
    let shape = tokens
        .get(1)
        .ok_or_else(|| PackError::Parse("constraint shape missing".into()))?;
    let shape = shape.to_lowercase();
    let shape = match shape.as_str() {
        "cube" => {
            if tokens.len() < 6 {
                return Err(PackError::Parse("cube requires center x y z side".into()));
            }
            let center = [
                parse_f32(tokens.get(2))?,
                parse_f32(tokens.get(3))?,
                parse_f32(tokens.get(4))?,
            ];
            let side = parse_f32(tokens.get(5))?;
            ShapeSpec::Cube { center, side }
        }
        "box" => {
            if tokens.len() < 8 {
                return Err(PackError::Parse("box requires min and max".into()));
            }
            let min = [
                parse_f32(tokens.get(2))?,
                parse_f32(tokens.get(3))?,
                parse_f32(tokens.get(4))?,
            ];
            let max = [
                parse_f32(tokens.get(5))?,
                parse_f32(tokens.get(6))?,
                parse_f32(tokens.get(7))?,
            ];
            ShapeSpec::Box { min, max }
        }
        "sphere" => {
            if tokens.len() < 6 {
                return Err(PackError::Parse("sphere requires center and radius".into()));
            }
            let center = [
                parse_f32(tokens.get(2))?,
                parse_f32(tokens.get(3))?,
                parse_f32(tokens.get(4))?,
            ];
            let radius = parse_f32(tokens.get(5))?;
            ShapeSpec::Sphere { center, radius }
        }
        "ellipsoid" => {
            if tokens.len() < 9 {
                return Err(PackError::Parse(
                    "ellipsoid requires center and radii".into(),
                ));
            }
            let center = [
                parse_f32(tokens.get(2))?,
                parse_f32(tokens.get(3))?,
                parse_f32(tokens.get(4))?,
            ];
            let radii = [
                parse_f32(tokens.get(5))?,
                parse_f32(tokens.get(6))?,
                parse_f32(tokens.get(7))?,
            ];
            ShapeSpec::Ellipsoid { center, radii }
        }
        "cylinder" => {
            if tokens.len() < 10 {
                return Err(PackError::Parse(
                    "cylinder requires base, axis, radius, height".into(),
                ));
            }
            let base = [
                parse_f32(tokens.get(2))?,
                parse_f32(tokens.get(3))?,
                parse_f32(tokens.get(4))?,
            ];
            let axis = [
                parse_f32(tokens.get(5))?,
                parse_f32(tokens.get(6))?,
                parse_f32(tokens.get(7))?,
            ];
            let radius = parse_f32(tokens.get(8))?;
            let height = parse_f32(tokens.get(9))?;
            ShapeSpec::Cylinder {
                base,
                axis,
                radius,
                height,
            }
        }
        "plane" => {
            if tokens.len() < 6 {
                return Err(PackError::Parse("plane requires nx ny nz d".into()));
            }
            let normal = [
                parse_f32(tokens.get(2))?,
                parse_f32(tokens.get(3))?,
                parse_f32(tokens.get(4))?,
            ];
            let d = parse_f32(tokens.get(5))?;
            let norm2 = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
            let point = if norm2 > 1.0e-8 {
                [
                    normal[0] * d / norm2,
                    normal[1] * d / norm2,
                    normal[2] * d / norm2,
                ]
            } else {
                [0.0, 0.0, d]
            };
            ShapeSpec::Plane { point, normal }
        }
        "xygauss" => {
            if tokens.len() < 8 {
                return Err(PackError::Parse("xygauss requires x y sx sy z0 amp".into()));
            }
            let center = [parse_f32(tokens.get(2))?, parse_f32(tokens.get(3))?];
            let sigma = [parse_f32(tokens.get(4))?, parse_f32(tokens.get(5))?];
            let z0 = parse_f32(tokens.get(6))?;
            let amplitude = parse_f32(tokens.get(7))?;
            ShapeSpec::XyGauss {
                center,
                sigma,
                z0,
                amplitude,
            }
        }
        _ => {
            return Err(PackError::Parse(format!(
                "unsupported constraint shape: {}",
                shape
            )));
        }
    };
    Ok(ConstraintSpec { mode, shape })
}

fn parse_atom_selection(tokens: &[&str]) -> PackResult<Vec<usize>> {
    if tokens.is_empty() {
        return Err(PackError::Parse("atoms block missing indices".into()));
    }
    if tokens.len() == 2 {
        let start = parse_usize(tokens.first())?;
        let end = parse_usize(tokens.get(1))?;
        let (lo, hi) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        return Ok((lo..=hi).collect());
    }
    tokens
        .iter()
        .map(|t| parse_usize(Some(t)))
        .collect::<PackResult<Vec<_>>>()
}

fn update_bounds_from_constraint(
    constraint: &ConstraintSpec,
    min: &mut Option<[f32; 3]>,
    max: &mut Option<[f32; 3]>,
) {
    let (cmin, cmax) = match &constraint.shape {
        ShapeSpec::Box { min, max } => (*min, *max),
        ShapeSpec::Cube { center, side } => {
            let half = *side * 0.5;
            (
                [center[0] - half, center[1] - half, center[2] - half],
                [center[0] + half, center[1] + half, center[2] + half],
            )
        }
        ShapeSpec::Sphere { center, radius } => (
            [
                center[0] - *radius,
                center[1] - *radius,
                center[2] - *radius,
            ],
            [
                center[0] + *radius,
                center[1] + *radius,
                center[2] + *radius,
            ],
        ),
        ShapeSpec::Ellipsoid { center, radii } => (
            [
                center[0] - radii[0],
                center[1] - radii[1],
                center[2] - radii[2],
            ],
            [
                center[0] + radii[0],
                center[1] + radii[1],
                center[2] + radii[2],
            ],
        ),
        _ => return,
    };
    *min = Some(match min {
        Some(existing) => [
            existing[0].min(cmin[0]),
            existing[1].min(cmin[1]),
            existing[2].min(cmin[2]),
        ],
        None => cmin,
    });
    *max = Some(match max {
        Some(existing) => [
            existing[0].max(cmax[0]),
            existing[1].max(cmax[1]),
            existing[2].max(cmax[2]),
        ],
        None => cmax,
    });
}

fn parse_usize(value: Option<&&str>) -> PackResult<usize> {
    parse_i64(value).and_then(|v| {
        if v <= 0 {
            Err(PackError::Parse("value must be positive".into()))
        } else {
            Ok(v as usize)
        }
    })
}

fn parse_i32(value: Option<&&str>) -> PackResult<i32> {
    parse_i64(value).map(|v| v as i32)
}

fn parse_i64(value: Option<&&str>) -> PackResult<i64> {
    let s = value.ok_or_else(|| PackError::Parse("missing value".into()))?;
    s.parse::<i64>()
        .map_err(|_| PackError::Parse(format!("invalid integer: {s}")))
}

fn parse_f32(value: Option<&&str>) -> PackResult<f32> {
    let s = value.ok_or_else(|| PackError::Parse("missing value".into()))?;
    s.parse::<f32>()
        .map_err(|_| PackError::Parse(format!("invalid float: {s}")))
}
