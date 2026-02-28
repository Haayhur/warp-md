use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use traj_core::error::{TrajError, TrajResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSchema {
    pub name: String,
    pub dtype: String,
    pub dim: usize,
    pub n_streams: usize,
    pub chunk_frames: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkIndex {
    pub offset_bytes: u64,
    pub frames: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureIndex {
    pub schema: FeatureSchema,
    pub chunks: Vec<ChunkIndex>,
    pub total_frames: usize,
}

pub struct FeatureStoreWriter {
    data: File,
    index: FeatureIndex,
    data_pos: u64,
    data_path: PathBuf,
    index_path: PathBuf,
}

impl FeatureStoreWriter {
    pub fn new(path: impl AsRef<Path>, schema: FeatureSchema) -> TrajResult<Self> {
        let base = path.as_ref();
        let data_path = base.with_extension("bin");
        let index_path = base.with_extension("json");
        let data = File::create(&data_path).map_err(TrajError::Io)?;
        Ok(Self {
            data,
            index: FeatureIndex {
                schema,
                chunks: Vec::new(),
                total_frames: 0,
            },
            data_pos: 0,
            data_path,
            index_path,
        })
    }

    pub fn write_chunk(&mut self, frames: usize, data: &[f32]) -> TrajResult<()> {
        let schema = &self.index.schema;
        let expected = frames
            .saturating_mul(schema.n_streams)
            .saturating_mul(schema.dim);
        if data.len() != expected {
            return Err(TrajError::Mismatch(
                "feature store chunk size does not match schema".into(),
            ));
        }
        let bytes = bytemuck::cast_slice(data);
        self.data.write_all(bytes).map_err(TrajError::Io)?;
        self.index.chunks.push(ChunkIndex {
            offset_bytes: self.data_pos,
            frames,
        });
        self.data_pos += bytes.len() as u64;
        self.index.total_frames += frames;
        Ok(())
    }

    pub fn finish(mut self) -> TrajResult<()> {
        self.data.flush().map_err(TrajError::Io)?;
        let json = serde_json::to_vec_pretty(&self.index)
            .map_err(|e| TrajError::Parse(format!("feature store index encode failed: {e}")))?;
        let mut index = File::create(&self.index_path).map_err(TrajError::Io)?;
        index.write_all(&json).map_err(TrajError::Io)?;
        Ok(())
    }

    pub fn data_path(&self) -> &Path {
        &self.data_path
    }

    pub fn index_path(&self) -> &Path {
        &self.index_path
    }
}

pub struct FeatureStoreReader {
    data: File,
    index: FeatureIndex,
}

impl FeatureStoreReader {
    pub fn open(path: impl AsRef<Path>) -> TrajResult<Self> {
        let base = path.as_ref();
        let data_path = base.with_extension("bin");
        let index_path = base.with_extension("json");
        let mut index_file = File::open(&index_path).map_err(TrajError::Io)?;
        let mut buf = Vec::new();
        index_file.read_to_end(&mut buf).map_err(TrajError::Io)?;
        let index: FeatureIndex = serde_json::from_slice(&buf)
            .map_err(|e| TrajError::Parse(format!("feature store index decode failed: {e}")))?;
        let data = File::open(&data_path).map_err(TrajError::Io)?;
        Ok(Self { data, index })
    }

    pub fn index(&self) -> &FeatureIndex {
        &self.index
    }

    pub fn read_chunk(&mut self, chunk_idx: usize) -> TrajResult<Vec<f32>> {
        let chunk =
            self.index.chunks.get(chunk_idx).ok_or_else(|| {
                TrajError::Mismatch("feature store chunk index out of range".into())
            })?;
        let schema = &self.index.schema;
        let count = chunk
            .frames
            .saturating_mul(schema.n_streams)
            .saturating_mul(schema.dim);
        let bytes = count.saturating_mul(std::mem::size_of::<f32>());
        self.data
            .seek(SeekFrom::Start(chunk.offset_bytes))
            .map_err(TrajError::Io)?;
        let mut buf = vec![0u8; bytes];
        self.data.read_exact(&mut buf).map_err(TrajError::Io)?;
        let floats = bytemuck::cast_slice::<u8, f32>(&buf).to_vec();
        Ok(floats)
    }
}
