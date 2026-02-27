#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(not(feature = "cuda"))]
use traj_core::error::{TrajError, TrajResult};

#[cfg(not(feature = "cuda"))]
pub struct GpuContext;

#[cfg(not(feature = "cuda"))]
impl GpuContext {
    pub fn new(_device: usize) -> TrajResult<Self> {
        Err(TrajError::Unsupported(
            "cuda feature disabled; rebuild with --features cuda".into(),
        ))
    }
}
