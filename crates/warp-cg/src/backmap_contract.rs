use crate::backmap::{
    BackmapArtifact, BackmapAtomMetadata, BackmapDiagnostics, BackmapPlan,
    BACKMAP_PLAN_SCHEMA_VERSION,
};
use crate::backmap_contract_io::{box3, box_vectors_from_box3, to_f32, write_gro, write_pdb};
use anyhow::{anyhow, Context, Result};
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use traj_core::frame::Box3;
use traj_core::frame::FrameChunkBuilder;
use traj_core::minimum_image_vector;
use traj_io::dcd::DcdWriter;
use traj_io::xtc::XtcWriter;

pub const BACKMAP_REQUEST_SCHEMA_VERSION: &str = "warp-cg.backmap.v1";
pub const BACKMAP_RESULT_SCHEMA_VERSION: &str = "warp-cg.backmap-result.v1";

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BackmapRequest {
    pub schema_version: String,
    #[serde(default)]
    pub plan_path: Option<String>,
    #[serde(default)]
    pub plan: Option<BackmapArtifact>,
    #[serde(default)]
    pub frames: Vec<Vec<[f64; 3]>>,
    #[serde(default)]
    pub trajectory_path: Option<String>,
    #[serde(default)]
    pub trajectory_format: Option<String>,
    #[serde(default)]
    pub chunk_frames: Option<usize>,
    #[serde(default)]
    pub include_coordinates: bool,
    #[serde(default)]
    pub box_vectors: Option<[[f64; 3]; 3]>,
    #[serde(default)]
    pub make_whole: bool,
    #[serde(default)]
    pub quality: BackmapQuality,
    #[serde(default)]
    pub output: BackmapOutput,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BackmapQuality {
    #[serde(default = "default_bead_tolerance")]
    pub max_bead_error: f64,
    #[serde(default = "default_bond_tolerance")]
    pub max_link_bond_error: f64,
    #[serde(default = "default_internal_bond_tolerance")]
    pub max_internal_bond_error: f64,
    #[serde(default)]
    pub max_chirality_inversions: usize,
    #[serde(default)]
    pub max_steric_clashes: usize,
    #[serde(default = "default_violation_policy")]
    pub on_violation: String,
}

impl Default for BackmapQuality {
    fn default() -> Self {
        Self {
            max_bead_error: default_bead_tolerance(),
            max_link_bond_error: default_bond_tolerance(),
            max_internal_bond_error: default_internal_bond_tolerance(),
            max_chirality_inversions: 0,
            max_steric_clashes: 0,
            on_violation: default_violation_policy(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BackmapOutput {
    #[serde(default)]
    pub out_dir: Option<String>,
    #[serde(default)]
    pub prefix: Option<String>,
    #[serde(default)]
    pub formats: Vec<String>,
    #[serde(default)]
    pub minimization_handoff: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BackmapArtifactRecord {
    pub kind: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct BackmapResult {
    pub schema_version: String,
    pub status: String,
    pub frame_count: usize,
    pub atom_count: usize,
    pub atom_metadata: Vec<BackmapAtomMetadata>,
    pub diagnostics: Vec<BackmapDiagnostics>,
    pub coordinates: Vec<Vec<[f64; 3]>>,
    pub artifacts: Vec<BackmapArtifactRecord>,
    pub warnings: Vec<String>,
}

pub fn schema_json(kind: &str) -> Result<String> {
    let value = match kind {
        "request" => serde_json::to_value(schema_for!(BackmapRequest))?,
        "result" => serde_json::to_value(schema_for!(BackmapResult))?,
        other => return Err(anyhow!("unknown backmap schema kind: {other}")),
    };
    Ok(serde_json::to_string_pretty(&value)?)
}

pub fn capabilities() -> Value {
    json!({
        "tool": "warp-cg backmap",
        "schema_version": BACKMAP_REQUEST_SCHEMA_VERSION,
        "plan_schema_version": BACKMAP_PLAN_SCHEMA_VERSION,
        "input": {
            "plan": ["plan_path", "inline plan"],
            "frames": "one or more CG coordinate frames",
            "pbc": ["orthorhombic", "triclinic metadata preserved"]
        },
        "outputs": ["json", "pdb", "gro", "xtc", "dcd", "minimization_handoff_json"],
        "quality_gates": [
            "mapped_bead_max_error",
            "link_bond_max_error",
            "internal_bond_max_error",
            "chirality_inversion_count",
            "steric_clash_count"
        ],
        "deterministic": true
    })
}

pub fn validate_request_json(text: &str) -> (i32, Value) {
    match parse_request(text).and_then(|request| validate_request(&request)) {
        Ok(()) => (
            0,
            json!({"valid": true, "schema_version": BACKMAP_REQUEST_SCHEMA_VERSION}),
        ),
        Err(err) => (
            2,
            json!({
                "valid": false,
                "schema_version": BACKMAP_REQUEST_SCHEMA_VERSION,
                "error": {"code": "warp_cg.backmap_invalid_request", "message": err.to_string()}
            }),
        ),
    }
}

pub fn run_request_json(text: &str) -> (i32, Value) {
    let result = parse_request(text)
        .and_then(|request| validate_request(&request).map(|_| request))
        .and_then(run_request);
    match result {
        Ok(result) => (
            0,
            serde_json::to_value(result).expect("serialize backmap result"),
        ),
        Err(err) => (
            1,
            json!({
                "schema_version": BACKMAP_RESULT_SCHEMA_VERSION,
                "status": "failed",
                "error": {"code": "warp_cg.backmap_failed", "message": err.to_string()}
            }),
        ),
    }
}

fn parse_request(text: &str) -> Result<BackmapRequest> {
    serde_json::from_str(text).context("parse backmap request")
}

fn validate_request(request: &BackmapRequest) -> Result<()> {
    if request.schema_version != BACKMAP_REQUEST_SCHEMA_VERSION {
        return Err(anyhow!(
            "schema_version must be {BACKMAP_REQUEST_SCHEMA_VERSION}"
        ));
    }
    if request.plan.is_some() == request.plan_path.is_some() {
        return Err(anyhow!("provide exactly one of plan or plan_path"));
    }
    if request.frames.is_empty() == request.trajectory_path.is_none() {
        return Err(anyhow!(
            "provide exactly one of non-empty frames or trajectory_path"
        ));
    }
    if !request.quality.max_bead_error.is_finite() || request.quality.max_bead_error < 0.0 {
        return Err(anyhow!(
            "quality.max_bead_error must be finite and non-negative"
        ));
    }
    if !request.quality.max_link_bond_error.is_finite() || request.quality.max_link_bond_error < 0.0
    {
        return Err(anyhow!(
            "quality.max_link_bond_error must be finite and non-negative"
        ));
    }
    if !request.quality.max_internal_bond_error.is_finite()
        || request.quality.max_internal_bond_error < 0.0
    {
        return Err(anyhow!(
            "quality.max_internal_bond_error must be finite and non-negative"
        ));
    }
    if !matches!(request.quality.on_violation.as_str(), "error" | "warn") {
        return Err(anyhow!("quality.on_violation must be error or warn"));
    }
    for format in &request.output.formats {
        if !matches!(format.as_str(), "json" | "pdb" | "gro" | "xtc" | "dcd") {
            return Err(anyhow!("unsupported output format {format}"));
        }
    }
    Ok(())
}

fn run_request(request: BackmapRequest) -> Result<BackmapResult> {
    let artifact = resolve_plan(&request)?;
    let plan = artifact.plan;
    if request.trajectory_path.is_some() {
        return run_trajectory_request(request, plan);
    }
    let frames = if request.make_whole {
        request
            .frames
            .iter()
            .map(|frame| make_whole_cg(frame, &plan, request.box_vectors))
            .collect::<Result<Vec<_>>>()?
    } else {
        request.frames.clone()
    };
    let mut coordinates = Vec::with_capacity(frames.len());
    let mut diagnostics = Vec::with_capacity(frames.len());
    let mut warnings = Vec::new();
    for (frame_idx, frame) in frames.iter().enumerate() {
        let rebuilt = plan
            .execute_frame(frame)
            .map_err(|err| anyhow!("frame {frame_idx}: {err}"))?;
        enforce_quality(&request, frame_idx, &rebuilt.diagnostics, &mut warnings)?;
        coordinates.push(rebuilt.coordinates);
        diagnostics.push(rebuilt.diagnostics);
    }
    let atom_metadata = plan.atom_metadata_in_output_order();
    let artifacts = write_outputs(&request, &plan, &coordinates, &diagnostics, &atom_metadata)?;
    let include_coordinates = request.include_coordinates;
    Ok(BackmapResult {
        schema_version: BACKMAP_RESULT_SCHEMA_VERSION.to_string(),
        status: if warnings.is_empty() {
            "completed".to_string()
        } else {
            "completed_with_warnings".to_string()
        },
        frame_count: coordinates.len(),
        atom_count: atom_metadata.len(),
        atom_metadata,
        diagnostics,
        coordinates: if include_coordinates {
            coordinates
        } else {
            Vec::new()
        },
        artifacts,
        warnings,
    })
}

fn run_trajectory_request(request: BackmapRequest, plan: BackmapPlan) -> Result<BackmapResult> {
    let path = Path::new(
        request
            .trajectory_path
            .as_deref()
            .expect("validated trajectory"),
    );
    let mut reader =
        crate::trajectory::open_reader(path, request.trajectory_format.as_deref(), None)?;
    let expected_beads = plan
        .templates
        .iter()
        .flat_map(|template| template.bead_sites.iter())
        .map(|site| site.target_bead_index)
        .max()
        .map_or(0, |idx| idx + 1);
    if reader.n_atoms() != expected_beads {
        return Err(anyhow!(
            "trajectory has {} beads but backmap plan requires {}",
            reader.n_atoms(),
            expected_beads
        ));
    }
    if request.output.formats.iter().any(|format| format == "dcd")
        && reader.n_frames_hint().is_none()
    {
        return Err(anyhow!(
            "streaming DCD output requires an input frame-count hint; use XTC output"
        ));
    }
    let out_dir = PathBuf::from(request.output.out_dir.as_deref().unwrap_or("."));
    fs::create_dir_all(&out_dir)?;
    let prefix = request.output.prefix.as_deref().unwrap_or("backmapped");
    let metadata = plan.atom_metadata_in_output_order();
    let box_override = box3(request.box_vectors);
    let mut xtc = request
        .output
        .formats
        .iter()
        .any(|format| format == "xtc")
        .then(|| XtcWriter::create(out_dir.join(format!("{prefix}.xtc")), metadata.len()))
        .transpose()?;
    let mut dcd = request
        .output
        .formats
        .iter()
        .any(|format| format == "dcd")
        .then(|| {
            DcdWriter::create(
                out_dir.join(format!("{prefix}.dcd")),
                metadata.len(),
                reader.n_frames_hint().unwrap_or(0),
            )
        })
        .transpose()?;
    let chunk_frames = request.chunk_frames.unwrap_or(64).max(1);
    let mut builder = FrameChunkBuilder::new(expected_beads, chunk_frames);
    builder.set_requirements(true, true);
    let mut diagnostics = Vec::new();
    let mut returned_coordinates = Vec::new();
    let mut warnings = Vec::new();
    let mut first_frame = None;
    let mut frame_count = 0usize;
    loop {
        let read = reader.read_chunk(chunk_frames, &mut builder)?;
        if read == 0 {
            break;
        }
        let chunk = builder.finish_take()?;
        for local_frame in 0..chunk.n_frames {
            let source =
                &chunk.coords[local_frame * chunk.n_atoms..(local_frame + 1) * chunk.n_atoms];
            let mut cg = source
                .iter()
                .map(|coord| {
                    [
                        f64::from(coord[0]),
                        f64::from(coord[1]),
                        f64::from(coord[2]),
                    ]
                })
                .collect::<Vec<_>>();
            let frame_box = chunk.box_.get(local_frame).copied().unwrap_or(Box3::None);
            if request.make_whole {
                let vectors = request
                    .box_vectors
                    .or_else(|| box_vectors_from_box3(frame_box))
                    .ok_or_else(|| anyhow!("make_whole requires frame or request box metadata"))?;
                cg = make_whole_cg(&cg, &plan, Some(vectors))?;
            }
            let rebuilt = plan
                .execute_frame(&cg)
                .map_err(|err| anyhow!("frame {frame_count}: {err}"))?;
            enforce_quality(&request, frame_count, &rebuilt.diagnostics, &mut warnings)?;
            let coords_f32 = to_f32(&rebuilt.coordinates);
            let output_box = if matches!(box_override, Box3::None) {
                frame_box
            } else {
                box_override
            };
            if let Some(writer) = xtc.as_mut() {
                writer.write_frame(&coords_f32, output_box, frame_count, None)?;
            }
            if let Some(writer) = dcd.as_mut() {
                writer.write_frame(&coords_f32, output_box)?;
            }
            if first_frame.is_none() {
                first_frame = Some(rebuilt.coordinates.clone());
            }
            if request.include_coordinates {
                returned_coordinates.push(rebuilt.coordinates);
            }
            diagnostics.push(rebuilt.diagnostics);
            frame_count += 1;
        }
        builder.reset(expected_beads, chunk_frames);
    }
    if let Some(writer) = xtc.as_mut() {
        writer.flush()?;
    }
    if let Some(writer) = dcd.as_mut() {
        writer.flush()?;
    }
    let first_frame = first_frame.ok_or_else(|| anyhow!("trajectory contains no frames"))?;
    let mut artifacts = Vec::new();
    for format in &request.output.formats {
        let path = out_dir.join(format!("{prefix}.{format}"));
        match format.as_str() {
            "pdb" => write_pdb(&path, &first_frame, &metadata, request.box_vectors)?,
            "gro" => write_gro(&path, &first_frame, &metadata, request.box_vectors)?,
            "json" => fs::write(
                &path,
                serde_json::to_vec_pretty(&json!({
                    "schema_version": BACKMAP_RESULT_SCHEMA_VERSION,
                    "frame_count": frame_count,
                    "atom_count": metadata.len(),
                    "diagnostics": diagnostics
                }))?,
            )?,
            "xtc" | "dcd" => {}
            _ => unreachable!("validated format"),
        }
        artifacts.push(BackmapArtifactRecord {
            kind: format!("backmapped_{format}"),
            path: path.to_string_lossy().to_string(),
        });
    }
    append_minimization_handoff(&request, &out_dir, prefix, &diagnostics, &mut artifacts)?;
    Ok(BackmapResult {
        schema_version: BACKMAP_RESULT_SCHEMA_VERSION.to_string(),
        status: if warnings.is_empty() {
            "completed".to_string()
        } else {
            "completed_with_warnings".to_string()
        },
        frame_count,
        atom_count: metadata.len(),
        atom_metadata: metadata,
        diagnostics,
        coordinates: returned_coordinates,
        artifacts,
        warnings,
    })
}

fn resolve_plan(request: &BackmapRequest) -> Result<BackmapArtifact> {
    if let Some(artifact) = &request.plan {
        if artifact.schema_version != BACKMAP_PLAN_SCHEMA_VERSION {
            return Err(anyhow!("unsupported inline backmap plan schema"));
        }
        return Ok(artifact.clone());
    }
    let path = request.plan_path.as_deref().expect("validated plan path");
    BackmapArtifact::from_json(&fs::read_to_string(path)?).map_err(anyhow::Error::msg)
}

fn write_outputs(
    request: &BackmapRequest,
    plan: &BackmapPlan,
    frames: &[Vec<[f64; 3]>],
    diagnostics: &[BackmapDiagnostics],
    metadata: &[BackmapAtomMetadata],
) -> Result<Vec<BackmapArtifactRecord>> {
    if request.output.formats.is_empty() && !request.output.minimization_handoff {
        return Ok(Vec::new());
    }
    let out_dir = PathBuf::from(request.output.out_dir.as_deref().unwrap_or("."));
    fs::create_dir_all(&out_dir)?;
    let prefix = request.output.prefix.as_deref().unwrap_or("backmapped");
    let mut artifacts = Vec::new();
    let box_ = box3(request.box_vectors);
    for format in &request.output.formats {
        let path = out_dir.join(format!("{prefix}.{format}"));
        match format.as_str() {
            "json" => fs::write(
                &path,
                serde_json::to_vec_pretty(&json!({
                    "schema_version": BACKMAP_RESULT_SCHEMA_VERSION,
                    "coordinates": frames,
                    "diagnostics": diagnostics,
                    "atom_metadata": metadata
                }))?,
            )?,
            "pdb" => write_pdb(&path, &frames[0], metadata, request.box_vectors)?,
            "gro" => write_gro(&path, &frames[0], metadata, request.box_vectors)?,
            "xtc" => {
                let mut writer = XtcWriter::create(&path, metadata.len())?;
                for (frame_idx, frame) in frames.iter().enumerate() {
                    writer.write_frame(&to_f32(frame), box_, frame_idx, None)?;
                }
                writer.flush()?;
            }
            "dcd" => {
                let mut writer = DcdWriter::create(&path, metadata.len(), frames.len())?;
                for frame in frames {
                    writer.write_frame(&to_f32(frame), box_)?;
                }
                writer.flush()?;
            }
            _ => unreachable!("validated format"),
        }
        artifacts.push(BackmapArtifactRecord {
            kind: format!("backmapped_{format}"),
            path: path.to_string_lossy().to_string(),
        });
    }
    append_minimization_handoff(request, &out_dir, prefix, diagnostics, &mut artifacts)?;
    let _ = plan;
    Ok(artifacts)
}

fn enforce_quality(
    request: &BackmapRequest,
    frame_idx: usize,
    diagnostics: &BackmapDiagnostics,
    warnings: &mut Vec<String>,
) -> Result<()> {
    let violation = diagnostics.mapped_bead_max_error > request.quality.max_bead_error
        || diagnostics.link_bond_max_error > request.quality.max_link_bond_error
        || diagnostics.internal_bond_max_error > request.quality.max_internal_bond_error
        || diagnostics.chirality_inversion_count > request.quality.max_chirality_inversions
        || diagnostics.steric_clash_count > request.quality.max_steric_clashes;
    if !violation {
        return Ok(());
    }
    let message = format!(
        "frame {frame_idx} quality gate failed: bead_max={} link_bond_max={} internal_bond_max={} chirality_inversions={} steric_clashes={}",
        diagnostics.mapped_bead_max_error,
        diagnostics.link_bond_max_error,
        diagnostics.internal_bond_max_error,
        diagnostics.chirality_inversion_count,
        diagnostics.steric_clash_count
    );
    if request.quality.on_violation == "error" {
        Err(anyhow!(message))
    } else {
        warnings.push(message);
        Ok(())
    }
}

fn append_minimization_handoff(
    request: &BackmapRequest,
    out_dir: &Path,
    prefix: &str,
    diagnostics: &[BackmapDiagnostics],
    artifacts: &mut Vec<BackmapArtifactRecord>,
) -> Result<()> {
    if !request.output.minimization_handoff {
        return Ok(());
    }
    let path = out_dir.join(format!("{prefix}_minimization_handoff.json"));
    fs::write(
        &path,
        serde_json::to_vec_pretty(&json!({
            "schema_version": "warp-cg.backmap-minimization-handoff.v1",
            "status": "required",
            "reason": "backmapping produces initialization coordinates; force-field minimization is required before production dynamics",
            "recommended_inputs": {
                "coordinates": artifacts.iter().find(|artifact| matches!(artifact.kind.as_str(), "backmapped_pdb" | "backmapped_gro")).map(|artifact| &artifact.path),
                "topology_from_original_aa_model": true,
                "hydrogen_policy": "existing source hydrogens are reconstructed when present in the plan; missing hydrogens must be added by the selected AA force-field preparation tool before minimization"
            },
            "quality": diagnostics
        }))?,
    )?;
    artifacts.push(BackmapArtifactRecord {
        kind: "minimization_handoff_json".to_string(),
        path: path.to_string_lossy().to_string(),
    });
    Ok(())
}

fn make_whole_cg(
    frame: &[[f64; 3]],
    plan: &BackmapPlan,
    vectors: Option<[[f64; 3]; 3]>,
) -> Result<Vec<[f64; 3]>> {
    let Some(vectors) = vectors else {
        return Err(anyhow!("make_whole requires box_vectors"));
    };
    let periodic_box = box3(Some(vectors));
    let mut output = frame.to_vec();
    let mut adjacency = vec![Vec::new(); frame.len()];
    for link in &plan.links {
        let left = plan.templates[link.from_template].bead_sites[0].target_bead_index;
        let right = plan.templates[link.to_template].bead_sites[0].target_bead_index;
        if left != right {
            adjacency[left].push(right);
            adjacency[right].push(left);
        }
    }
    let mut visited = vec![false; frame.len()];
    for root in 0..frame.len() {
        if visited[root] {
            continue;
        }
        visited[root] = true;
        let mut queue = std::collections::VecDeque::from([root]);
        while let Some(node) = queue.pop_front() {
            for &neighbor in &adjacency[node] {
                if visited[neighbor] {
                    continue;
                }
                let delta = minimum_image_vector(frame[node], frame[neighbor], periodic_box, 1.0)?;
                for axis in 0..3 {
                    output[neighbor][axis] = output[node][axis] + delta[axis];
                }
                visited[neighbor] = true;
                queue.push_back(neighbor);
            }
        }
    }
    Ok(output)
}

fn default_bead_tolerance() -> f64 {
    1.0e-5
}

fn default_bond_tolerance() -> f64 {
    0.25
}

fn default_internal_bond_tolerance() -> f64 {
    1.0e-6
}

fn default_violation_policy() -> String {
    "error".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backmap::{BeadSite, ResidueTemplate};

    fn artifact() -> BackmapArtifact {
        BackmapArtifact::new(BackmapPlan {
            templates: vec![ResidueTemplate {
                name: "MOL".to_string(),
                atom_names: vec!["A".to_string(), "B".to_string()],
                elements: vec!["C".to_string(), "C".to_string()],
                residue_names: vec!["MOL".to_string(), "MOL".to_string()],
                residue_ids: vec![1, 1],
                chains: vec!["A".to_string(), "A".to_string()],
                source_atom_indices: vec![0, 1],
                reference_coords: vec![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                bead_sites: vec![BeadSite {
                    target_bead_index: 0,
                    atom_indices: vec![0, 1],
                    weights: None,
                }],
                bonds: Vec::new(),
                chirality: Vec::new(),
            }],
            links: vec![],
            fudge_factor: 1.0,
        })
    }

    #[test]
    fn inline_request_runs_multiple_frames_and_writes_structures() {
        let tmp = tempfile::tempdir().unwrap();
        let request = BackmapRequest {
            schema_version: BACKMAP_REQUEST_SCHEMA_VERSION.to_string(),
            plan_path: None,
            plan: Some(artifact()),
            frames: vec![vec![[0.0, 0.0, 0.0]], vec![[2.0, 0.0, 0.0]]],
            trajectory_path: None,
            trajectory_format: None,
            chunk_frames: None,
            include_coordinates: true,
            box_vectors: Some([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]),
            make_whole: false,
            quality: BackmapQuality::default(),
            output: BackmapOutput {
                out_dir: Some(tmp.path().to_string_lossy().to_string()),
                prefix: Some("test".to_string()),
                formats: vec!["json".to_string(), "pdb".to_string(), "gro".to_string()],
                minimization_handoff: true,
            },
        };
        let result = run_request(request).unwrap();
        assert_eq!(result.frame_count, 2);
        assert!(tmp.path().join("test.pdb").is_file());
        assert!(tmp.path().join("test.gro").is_file());
        assert!(tmp.path().join("test_minimization_handoff.json").is_file());
    }

    #[test]
    fn trajectory_request_streams_xtc_without_returning_coordinates() {
        let tmp = tempfile::tempdir().unwrap();
        let input = tmp.path().join("cg.xtc");
        let mut writer = XtcWriter::create(&input, 1).unwrap();
        writer
            .write_frame(
                &[[0.0, 0.0, 0.0]],
                Box3::Orthorhombic {
                    lx: 10.0,
                    ly: 10.0,
                    lz: 10.0,
                },
                0,
                None,
            )
            .unwrap();
        writer
            .write_frame(
                &[[2.0, 0.0, 0.0]],
                Box3::Orthorhombic {
                    lx: 10.0,
                    ly: 10.0,
                    lz: 10.0,
                },
                1,
                None,
            )
            .unwrap();
        writer.flush().unwrap();
        let request = BackmapRequest {
            schema_version: BACKMAP_REQUEST_SCHEMA_VERSION.to_string(),
            plan_path: None,
            plan: Some(artifact()),
            frames: Vec::new(),
            trajectory_path: Some(input.to_string_lossy().to_string()),
            trajectory_format: Some("xtc".to_string()),
            chunk_frames: Some(1),
            include_coordinates: false,
            box_vectors: None,
            make_whole: false,
            quality: BackmapQuality::default(),
            output: BackmapOutput {
                out_dir: Some(tmp.path().to_string_lossy().to_string()),
                prefix: Some("aa".to_string()),
                formats: vec!["xtc".to_string(), "json".to_string()],
                minimization_handoff: false,
            },
        };
        let result = run_request(request).unwrap();
        assert_eq!(result.frame_count, 2);
        assert!(result.coordinates.is_empty());
        assert!(tmp.path().join("aa.xtc").is_file());
        assert!(tmp.path().join("aa.json").is_file());
    }
}
