use crate::reference::ReferenceData;

use super::{CgArtifact, CgReferenceMetadata, CgReferenceResult};

pub(super) fn reference_result_from_data(reference: &ReferenceData) -> CgReferenceResult {
    CgReferenceResult {
        source_kind: reference.source_kind.clone(),
        target_set_available: reference.target_set.is_some(),
        mapped_trajectory: reference
            .mapped_trajectory
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
        metrics: reference.metrics.clone(),
        metadata: CgReferenceMetadata {
            frames_read: reference.metadata.frames_read,
            frames_written: reference.metadata.frames_written,
            source_path: reference
                .metadata
                .source_path
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
            mapped_by: reference.metadata.mapped_by.clone(),
        },
        artifacts: reference
            .artifacts
            .iter()
            .map(|artifact| CgArtifact {
                path: artifact.path.to_string_lossy().to_string(),
                kind: artifact.kind.clone(),
            })
            .collect(),
    }
}
