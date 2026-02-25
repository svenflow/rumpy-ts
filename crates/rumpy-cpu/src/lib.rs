//! CPU Backend for RumPy
//!
//! Uses ndarray for array operations and faer for linear algebra.

mod array;
mod compare;
mod creation;
mod linalg;
mod manipulation;
mod math;
mod random;
mod sort;
mod stats;

pub use array::CpuArray;

use rumpy_core::{ops::*, Backend};

/// CPU backend using ndarray + faer
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn name() -> &'static str {
        "cpu"
    }

    fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    #[cfg(target_feature = "avx2")]
    fn has_simd() -> bool {
        true
    }

    #[cfg(not(target_feature = "avx2"))]
    fn has_simd() -> bool {
        false
    }
}

// Re-export the array type
pub type Array = CpuArray;
