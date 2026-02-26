//! WASM bindings for RumPy
//!
//! This crate provides JavaScript/TypeScript bindings for RumPy using wasm-bindgen.
//! It wraps the CPU backend for use in web browsers and Node.js.
//!
//! ## Zero-Copy Memory Access
//!
//! When SharedArrayBuffer is available (requires COOP/COEP headers), you can use
//! `asTypedArrayView()` for zero-copy access to array data. Otherwise, use
//! `toTypedArray()` which creates a copy.

use js_sys::{Float32Array, Float64Array};
use rumpy_core::{ops::*, Array};
use rumpy_cpu::{simd_gemm, CpuArray, CpuBackend};
use wasm_bindgen::prelude::*;

// Re-export wasm-bindgen-rayon's init function for Web Worker setup
pub use wasm_bindgen_rayon::init_thread_pool;

/// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// N-dimensional array type exposed to JavaScript
#[wasm_bindgen]
pub struct NDArray {
    inner: CpuArray,
}

impl NDArray {
    fn new(inner: CpuArray) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen]
impl NDArray {
    /// Get array shape
    #[wasm_bindgen(getter)]
    pub fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    /// Number of dimensions
    #[wasm_bindgen(getter)]
    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Total number of elements
    #[wasm_bindgen(getter)]
    pub fn size(&self) -> usize {
        self.inner.size()
    }

    /// Data type
    #[wasm_bindgen(getter)]
    pub fn dtype(&self) -> String {
        self.inner.dtype().to_string()
    }

    /// Convert to Float64Array (creates a copy)
    ///
    /// This method always works but involves copying data from WASM memory to JS.
    /// For zero-copy access when SharedArrayBuffer is available, use `asTypedArrayView()`.
    #[wasm_bindgen(js_name = toTypedArray)]
    pub fn to_typed_array(&self) -> Float64Array {
        Float64Array::from(self.inner.as_f64_slice().as_slice())
    }

    /// Get pointer to the underlying data buffer
    ///
    /// Returns the byte offset into WASM linear memory where this array's data begins.
    /// Use with `memory()` to create a zero-copy TypedArray view.
    ///
    /// WARNING: The pointer is only valid while this NDArray exists and WASM memory
    /// hasn't been resized. Cache invalidation is the caller's responsibility.
    #[wasm_bindgen(js_name = dataPtr)]
    pub fn data_ptr(&self) -> usize {
        self.inner.as_ndarray().as_ptr() as usize
    }

    /// Get the number of elements in the array
    #[wasm_bindgen(js_name = len)]
    pub fn len(&self) -> usize {
        self.inner.size()
    }

    /// Check if array is empty
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.size() == 0
    }

    /// Get total size in bytes
    #[wasm_bindgen(js_name = nbytes)]
    pub fn nbytes(&self) -> usize {
        self.inner.size() * std::mem::size_of::<f64>()
    }

    /// Get element at flat index
    #[wasm_bindgen(js_name = getFlat)]
    pub fn get_flat(&self, index: usize) -> f64 {
        self.inner.get_flat(index)
    }

    /// Clone the array
    #[wasm_bindgen(js_name = clone)]
    pub fn clone_array(&self) -> NDArray {
        NDArray::new(self.inner.clone())
    }

    /// Explicitly free the array memory
    ///
    /// After calling this, the NDArray is consumed and cannot be used.
    /// This is useful for deterministic memory cleanup without waiting for GC.
    #[wasm_bindgen]
    pub fn free(self) {
        // self is dropped here, freeing the underlying memory
    }

    // Scalar operations
    #[wasm_bindgen(js_name = addScalar)]
    pub fn add_scalar(&self, scalar: f64) -> NDArray {
        NDArray::new(CpuBackend::add_scalar(&self.inner, scalar))
    }

    #[wasm_bindgen(js_name = subScalar)]
    pub fn sub_scalar(&self, scalar: f64) -> NDArray {
        NDArray::new(CpuBackend::sub_scalar(&self.inner, scalar))
    }

    #[wasm_bindgen(js_name = mulScalar)]
    pub fn mul_scalar(&self, scalar: f64) -> NDArray {
        NDArray::new(CpuBackend::mul_scalar(&self.inner, scalar))
    }

    #[wasm_bindgen(js_name = divScalar)]
    pub fn div_scalar(&self, scalar: f64) -> NDArray {
        NDArray::new(CpuBackend::div_scalar(&self.inner, scalar))
    }

    #[wasm_bindgen(js_name = powScalar)]
    pub fn pow_scalar(&self, scalar: f64) -> NDArray {
        NDArray::new(CpuBackend::pow_scalar(&self.inner, scalar))
    }

    // Element-wise operations
    pub fn add(&self, other: &NDArray) -> Result<NDArray, JsValue> {
        CpuBackend::add(&self.inner, &other.inner)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn sub(&self, other: &NDArray) -> Result<NDArray, JsValue> {
        CpuBackend::sub(&self.inner, &other.inner)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn mul(&self, other: &NDArray) -> Result<NDArray, JsValue> {
        CpuBackend::mul(&self.inner, &other.inner)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn div(&self, other: &NDArray) -> Result<NDArray, JsValue> {
        CpuBackend::div(&self.inner, &other.inner)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // Reductions
    pub fn sum(&self) -> f64 {
        CpuBackend::sum(&self.inner)
    }

    pub fn mean(&self) -> f64 {
        CpuBackend::mean(&self.inner)
    }

    pub fn min(&self) -> f64 {
        CpuBackend::min(&self.inner)
    }

    pub fn max(&self) -> f64 {
        CpuBackend::max(&self.inner)
    }

    #[wasm_bindgen(js_name = std)]
    pub fn std_dev(&self) -> f64 {
        CpuBackend::std(&self.inner)
    }

    pub fn var(&self) -> f64 {
        CpuBackend::var(&self.inner)
    }

    // Reshape
    pub fn reshape(&self, shape: Vec<usize>) -> Result<NDArray, JsValue> {
        CpuBackend::reshape(&self.inner, shape)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn flatten(&self) -> NDArray {
        NDArray::new(CpuBackend::flatten(&self.inner))
    }

    pub fn transpose(&self) -> NDArray {
        NDArray::new(CpuBackend::transpose(&self.inner))
    }
}

// ============ Creation functions ============

#[wasm_bindgen(js_name = arrayFromTyped)]
pub fn array_from_typed(data: &Float64Array, shape: Vec<usize>) -> Result<NDArray, JsValue> {
    let vec: Vec<f64> = data.to_vec();
    CpuArray::from_f64_vec(vec, shape)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn zeros(shape: Vec<usize>) -> NDArray {
    NDArray::new(CpuBackend::zeros(shape))
}

#[wasm_bindgen]
pub fn ones(shape: Vec<usize>) -> NDArray {
    NDArray::new(CpuBackend::ones(shape))
}

#[wasm_bindgen]
pub fn full(shape: Vec<usize>, value: f64) -> NDArray {
    NDArray::new(CpuBackend::full(shape, value))
}

#[wasm_bindgen]
pub fn arange(start: f64, stop: f64, step: f64) -> Result<NDArray, JsValue> {
    CpuBackend::arange(start, stop, step)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn linspace(start: f64, stop: f64, num: usize) -> NDArray {
    NDArray::new(CpuBackend::linspace(start, stop, num))
}

#[wasm_bindgen]
pub fn eye(n: usize) -> NDArray {
    NDArray::new(CpuBackend::eye(n))
}

// ============ Math functions ============
// Note: Function names use "Arr" suffix to avoid collision with libm's math symbols
// (sin/exp/cos/etc) which causes linker conflicts when atomics is enabled.

#[wasm_bindgen(js_name = sinArr)]
pub fn sin_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::sin(&arr.inner))
}

#[wasm_bindgen(js_name = cosArr)]
pub fn cos_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::cos(&arr.inner))
}

#[wasm_bindgen(js_name = tanArr)]
pub fn tan_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::tan(&arr.inner))
}

#[wasm_bindgen(js_name = expArr)]
pub fn exp_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::exp(&arr.inner))
}

#[wasm_bindgen(js_name = logArr)]
pub fn log_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::log(&arr.inner))
}

#[wasm_bindgen(js_name = sqrtArr)]
pub fn sqrt_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::sqrt(&arr.inner))
}

#[wasm_bindgen(js_name = absArr)]
pub fn abs_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::abs(&arr.inner))
}

#[wasm_bindgen(js_name = floorArr)]
pub fn floor_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::floor(&arr.inner))
}

#[wasm_bindgen(js_name = ceilArr)]
pub fn ceil_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::ceil(&arr.inner))
}

#[wasm_bindgen(js_name = roundArr)]
pub fn round_arr(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::round(&arr.inner))
}

// ============ Linear algebra ============

#[wasm_bindgen]
pub fn matmul(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    CpuBackend::matmul(&a.inner, &b.inner)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn dot(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    CpuBackend::dot(&a.inner, &b.inner)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn inv(arr: &NDArray) -> Result<NDArray, JsValue> {
    CpuBackend::inv(&arr.inner)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn det(arr: &NDArray) -> Result<f64, JsValue> {
    CpuBackend::det(&arr.inner).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
pub fn solve(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    CpuBackend::solve(&a.inner, &b.inner)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

// ============ Random ============

#[wasm_bindgen(js_name = randomSeed)]
pub fn random_seed(seed: u64) {
    CpuBackend::seed(seed);
}

#[wasm_bindgen(js_name = randomRand)]
pub fn random_rand(shape: Vec<usize>) -> NDArray {
    NDArray::new(CpuBackend::rand(shape))
}

#[wasm_bindgen(js_name = randomRandn)]
pub fn random_randn(shape: Vec<usize>) -> NDArray {
    NDArray::new(CpuBackend::randn(shape))
}

#[wasm_bindgen(js_name = randomUniform)]
pub fn random_uniform(low: f64, high: f64, shape: Vec<usize>) -> NDArray {
    NDArray::new(CpuBackend::uniform(low, high, shape))
}

#[wasm_bindgen(js_name = randomNormal)]
pub fn random_normal(loc: f64, scale: f64, shape: Vec<usize>) -> NDArray {
    NDArray::new(CpuBackend::normal(loc, scale, shape))
}

// ============ Memory access for zero-copy ============

/// Get WASM linear memory for zero-copy access
///
/// Returns the WebAssembly.Memory object that backs all arrays.
/// Use with `dataPtr()` and `len()` to create zero-copy TypedArray views:
///
/// ```javascript
/// const wasmMemory = rumpy.wasmMemory();
/// const ptr = array.dataPtr();
/// const len = array.len();
/// const view = new Float64Array(wasmMemory.buffer, ptr, len);
/// // view is now a zero-copy view into the array's data
/// ```
///
/// Note: Views are invalidated if WASM memory grows. Monitor memory size
/// or recreate views after operations that might allocate.
#[wasm_bindgen(js_name = wasmMemory)]
pub fn wasm_memory() -> JsValue {
    wasm_bindgen::memory()
}

/// Check if SharedArrayBuffer is available
///
/// Returns true if the environment supports SharedArrayBuffer (COOP/COEP headers set).
/// When false, `asTypedArrayView()` will not work and you should use `toTypedArray()`.
#[wasm_bindgen(js_name = hasSharedArrayBuffer)]
pub fn has_shared_array_buffer() -> bool {
    // In WASM context, check if memory is shared
    // This is a runtime check - the actual capability depends on browser headers
    js_sys::Reflect::has(&wasm_bindgen::memory(), &JsValue::from_str("buffer")).unwrap_or(false)
}

// ============ High-performance f32 SIMD matmul ============

/// Fast f32 matrix multiplication using WASM SIMD
///
/// This is a direct binding to the SIMD-optimized GEMM kernel, matching XNNPACK's approach.
/// Uses f32 (4 elements per v128) instead of f64 (2 elements per v128) for 2x throughput.
///
/// Parameters:
/// - a: Float32Array, row-major, shape [m, k]
/// - b: Float32Array, row-major, shape [k, n]
/// - m, n, k: matrix dimensions
///
/// Returns: Float32Array of shape [m, n]
#[wasm_bindgen(js_name = matmulF32)]
pub fn matmul_f32(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let c_vec = simd_gemm::matmul_dispatch_f32(&a_vec, &b_vec, m, n, k);
    Float32Array::from(c_vec.as_slice())
}

/// Fast f64 matrix multiplication using WASM SIMD
///
/// Direct binding to the SIMD-optimized GEMM kernel for f64.
/// Uses f64x2 (2 elements per v128).
///
/// Parameters:
/// - a: Float64Array, row-major, shape [m, k]
/// - b: Float64Array, row-major, shape [k, n]
/// - m, n, k: matrix dimensions
///
/// Returns: Float64Array of shape [m, n]
#[wasm_bindgen(js_name = matmulF64)]
pub fn matmul_f64(a: &Float64Array, b: &Float64Array, m: usize, n: usize, k: usize) -> Float64Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let c_vec = simd_gemm::matmul_dispatch_f64(&a_vec, &b_vec, m, n, k);
    Float64Array::from(c_vec.as_slice())
}

/// Fast f32 matrix multiplication with explicit matrix packing
///
/// This version always uses matrix packing regardless of size, for benchmarking.
/// Packing reorders B matrix into cache-friendly column panels.
///
/// Parameters:
/// - a: Float32Array, row-major, shape [m, k]
/// - b: Float32Array, row-major, shape [k, n]
/// - m, n, k: matrix dimensions
///
/// Returns: Float32Array of shape [m, n]
#[wasm_bindgen(js_name = matmulF32Packed)]
pub fn matmul_f32_packed(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let mut c_vec = vec![0.0f32; m * n];

    #[cfg(target_arch = "wasm32")]
    {
        if m >= 4 && n >= 8 {
            unsafe {
                simd_gemm::matmul_simd_f32_packed(&a_vec, &b_vec, &mut c_vec, m, n, k);
            }
        } else {
            simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
    }

    Float32Array::from(c_vec.as_slice())
}

/// Fast f32 matrix multiplication with FMA (fused multiply-add)
///
/// Uses relaxed-simd f32x4_relaxed_madd for better throughput.
/// FMA computes a*b+c in one instruction instead of two (mul + add).
///
/// Parameters:
/// - a: Float32Array, row-major, shape [m, k]
/// - b: Float32Array, row-major, shape [k, n]
/// - m, n, k: matrix dimensions
///
/// Returns: Float32Array of shape [m, n]
#[wasm_bindgen(js_name = matmulF32FMA)]
pub fn matmul_f32_fma(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let mut c_vec = vec![0.0f32; m * n];

    #[cfg(target_arch = "wasm32")]
    {
        if m >= 4 && n >= 8 {
            unsafe {
                simd_gemm::matmul_simd_f32_fma(&a_vec, &b_vec, &mut c_vec, m, n, k);
            }
        } else {
            simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
    }

    Float32Array::from(c_vec.as_slice())
}

/// Fast f32 matrix multiplication with FMA + packed B
///
/// Combines both optimizations: FMA instructions and B matrix packing.
/// This is the fastest kernel for large matrices.
///
/// Parameters:
/// - a: Float32Array, row-major, shape [m, k]
/// - b: Float32Array, row-major, shape [k, n]
/// - m, n, k: matrix dimensions
///
/// Returns: Float32Array of shape [m, n]
#[wasm_bindgen(js_name = matmulF32FMAPacked)]
pub fn matmul_f32_fma_packed(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let mut c_vec = vec![0.0f32; m * n];

    #[cfg(target_arch = "wasm32")]
    {
        if m >= 4 && n >= 8 {
            unsafe {
                simd_gemm::matmul_simd_f32_fma_packed(&a_vec, &b_vec, &mut c_vec, m, n, k);
            }
        } else {
            simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
    }

    Float32Array::from(c_vec.as_slice())
}

/// Auto-tuned f32 matrix multiplication
///
/// Automatically selects the best kernel based on matrix dimensions:
/// - 5x8 kernel for matrices where M % 5 == 0 (like 100x100)
/// - FMA for medium matrices (packing overhead not amortized)
/// - FMA + packed for large matrices (packing overhead amortized)
///
/// Parameters:
/// - a: Float32Array, row-major, shape [m, k]
/// - b: Float32Array, row-major, shape [k, n]
/// - m, n, k: matrix dimensions
///
/// Returns: Float32Array of shape [m, n]
#[wasm_bindgen(js_name = matmulF32Auto)]
pub fn matmul_f32_auto(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let mut c_vec = vec![0.0f32; m * n];

    #[cfg(target_arch = "wasm32")]
    {
        unsafe {
            simd_gemm::matmul_simd_f32_auto(&a_vec, &b_vec, &mut c_vec, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
    }

    Float32Array::from(c_vec.as_slice())
}

/// 5x8 kernel specifically for matrices where M is divisible by 5
///
/// Optimized for 100x100 case (and similar).
#[wasm_bindgen(js_name = matmulF325x8)]
pub fn matmul_f32_5x8(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let mut c_vec = vec![0.0f32; m * n];

    #[cfg(target_arch = "wasm32")]
    {
        if m >= 5 && n >= 8 {
            unsafe {
                simd_gemm::matmul_simd_f32_5x8(&a_vec, &b_vec, &mut c_vec, m, n, k);
            }
        } else {
            simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
    }

    Float32Array::from(c_vec.as_slice())
}

/// Verify correctness: compute max absolute difference between two f32 arrays
///
/// Returns the maximum |a[i] - b[i]| across all elements.
/// Use this to verify that different kernels produce the same results.
#[wasm_bindgen(js_name = maxAbsDiff)]
pub fn max_abs_diff(a: &Float32Array, b: &Float32Array) -> f32 {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();

    a_vec.iter()
        .zip(b_vec.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Compute checksum (sum of all elements) for verification
#[wasm_bindgen(js_name = checksum)]
pub fn checksum(a: &Float32Array) -> f32 {
    a.to_vec().iter().sum()
}

/// Use the gemm crate for highly optimized GEMM
///
/// The gemm crate uses BLIS-style optimizations:
/// - Cache-blocking at L1/L2/L3 levels
/// - Optimized micro-kernels
/// - Smart packing strategies
///
/// This should be as fast as or faster than our hand-written SIMD kernels.
#[wasm_bindgen(js_name = matmulGemm)]
pub fn matmul_gemm(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let c_vec = simd_gemm::matmul_gemm_f32(&a_vec, &b_vec, m, n, k);
    Float32Array::from(c_vec.as_slice())
}

/// Parallel f32 matrix multiplication using rayon + Web Workers
///
/// Uses rayon to parallelize across the M dimension with native WASM threads.
/// MUST call `initThreadPool(num_threads)` from JS before using this function!
///
/// For large matrices (256+), this scales with available cores.
/// Falls back to single-threaded for small matrices.
///
/// Parameters:
/// - a: Float32Array, row-major, shape [m, k]
/// - b: Float32Array, row-major, shape [k, n]
/// - m, n, k: matrix dimensions
///
/// Returns: Float32Array of shape [m, n]
#[wasm_bindgen(js_name = matmulF32Parallel)]
pub fn matmul_f32_parallel(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let c_vec = simd_gemm::matmul_parallel_f32(&a_vec, &b_vec, m, n, k);
    Float32Array::from(c_vec.as_slice())
}

/// Parallel f32 matrix multiplication V2 using rayon + Web Workers (zero-allocation)
///
/// This is an improved version that writes directly to pre-allocated memory,
/// avoiding per-thread allocations. This is significantly faster than V1
/// for large matrices.
///
/// MUST call `initThreadPool(num_threads)` from JS before using this function!
///
/// Parameters:
/// - a: Float32Array, row-major, shape [m, k]
/// - b: Float32Array, row-major, shape [k, n]
/// - m, n, k: matrix dimensions
///
/// Returns: Float32Array of shape [m, n]
#[wasm_bindgen(js_name = matmulF32ParallelV2)]
pub fn matmul_f32_parallel_v2(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let c_vec = simd_gemm::matmul_parallel_f32_v2(&a_vec, &b_vec, m, n, k);
    Float32Array::from(c_vec.as_slice())
}

/// Get the current number of rayon threads
#[wasm_bindgen(js_name = getNumThreads)]
pub fn get_num_threads() -> usize {
    rayon::current_num_threads()
}

/// Parallel f32 matrix multiplication using pthreadpool-rs
///
/// Uses pthreadpool-rs instead of rayon for parallelization.
/// On WASM, pthreadpool-rs uses wasm-bindgen-rayon under the hood.
///
/// MUST call `initThreadPool(num_threads)` from JS before using this function!
///
/// Parameters:
/// - a: Float32Array, row-major, shape [m, k]
/// - b: Float32Array, row-major, shape [k, n]
/// - m, n, k: matrix dimensions
///
/// Returns: Float32Array of shape [m, n]
#[wasm_bindgen(js_name = matmulF32Pthreadpool)]
pub fn matmul_f32_pthreadpool(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let c_vec = simd_gemm::matmul_pthreadpool_f32(&a_vec, &b_vec, m, n, k);
    Float32Array::from(c_vec.as_slice())
}

/// XNNPACK-style f32 GEMM with pre-packed B matrix
///
/// This is a two-phase API:
/// 1. Call `packB` once to convert B into XNNPACK format
/// 2. Call `matmulXnnpack` multiple times with different A matrices
///
/// This amortizes the packing cost over many matmuls, which is how XNNPACK works.
#[wasm_bindgen(js_name = packB)]
pub fn pack_b(b: &Float32Array, k: usize, n: usize) -> Float32Array {
    let b_vec = b.to_vec();
    let n_panels = n / 8;
    let mut packed = vec![0.0f32; n_panels * k * 8];
    simd_gemm::pack_b_xnnpack(&b_vec, &mut packed, k, n);
    Float32Array::from(packed.as_slice())
}

/// XNNPACK-style matmul with pre-packed B
///
/// Requires both the original B (for remaining columns) and packed_b (for SIMD panels).
/// This handles arbitrary N, not just multiples of 8.
#[wasm_bindgen(js_name = matmulXnnpack)]
pub fn matmul_xnnpack(a: &Float32Array, b: &Float32Array, packed_b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let pb_vec = packed_b.to_vec();
    let mut c_vec = vec![0.0f32; m * n];

    #[cfg(target_arch = "wasm32")]
    {
        if m >= 6 && n >= 8 {
            unsafe {
                simd_gemm::matmul_simd_f32_xnnpack_style_full(&a_vec, &b_vec, &pb_vec, &mut c_vec, m, n, k);
            }
        } else {
            simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
    }

    Float32Array::from(c_vec.as_slice())
}

/// Cache-blocked 6x8 GEMM for large matrices
///
/// Uses GOTO-style cache blocking to tile the computation:
/// - Outer loop tiles by N dimension (NC=256)
/// - Middle loop tiles by K dimension (KC=256)
/// - Inner loop tiles by M dimension (MC=128)
///
/// This ensures working set fits in L1/L2 cache for better performance
/// on large matrices (256x256 and above).
#[wasm_bindgen(js_name = matmulF32Blocked)]
pub fn matmul_f32_blocked(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let mut c_vec = vec![0.0f32; m * n];

    #[cfg(target_arch = "wasm32")]
    {
        if m >= 6 && n >= 8 {
            unsafe {
                simd_gemm::matmul_simd_f32_6x8_blocked(&a_vec, &b_vec, &mut c_vec, m, n, k);
            }
        } else {
            simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
    }

    Float32Array::from(c_vec.as_slice())
}

/// Highly optimized 6x8 GEMM with FMA, loadsplat, and cache blocking
///
/// This is the most optimized implementation, matching XNNPACK patterns:
/// - 6x8 micro-kernel (12 accumulators fit in 16 XMM registers)
/// - f32x4_relaxed_madd for FMA
/// - v128_load32_splat for A broadcast
/// - L1/L2 cache blocking (KC=256, MC=72, NC=128)
/// - B matrix packing for contiguous access
#[wasm_bindgen(js_name = matmulF32Optimized)]
pub fn matmul_f32_optimized(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();

    #[cfg(target_arch = "wasm32")]
    {
        let c_vec = simd_gemm::matmul_optimized_f32(&a_vec, &b_vec, m, n, k);
        Float32Array::from(c_vec.as_slice())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let c_vec = simd_gemm::matmul_dispatch_f32(&a_vec, &b_vec, m, n, k);
        Float32Array::from(c_vec.as_slice())
    }
}

/// Parallel version of optimized 6x8 GEMM using rayon
#[wasm_bindgen(js_name = matmulF32OptimizedParallel)]
pub fn matmul_f32_optimized_parallel(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();

    #[cfg(target_arch = "wasm32")]
    {
        let c_vec = simd_gemm::matmul_optimized_f32_parallel(&a_vec, &b_vec, m, n, k);
        Float32Array::from(c_vec.as_slice())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let c_vec = simd_gemm::matmul_parallel_f32(&a_vec, &b_vec, m, n, k);
        Float32Array::from(c_vec.as_slice())
    }
}

/// Cache-blocked XNNPACK-style matmul with pre-packed B
///
/// Combines cache blocking with B-matrix packing for optimal performance.
/// Best for large matrices where both cache blocking and packing help.
#[wasm_bindgen(js_name = matmulXnnpackBlocked)]
pub fn matmul_xnnpack_blocked(a: &Float32Array, b: &Float32Array, packed_b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let b_vec = b.to_vec();
    let pb_vec = packed_b.to_vec();
    let mut c_vec = vec![0.0f32; m * n];

    #[cfg(target_arch = "wasm32")]
    {
        if m >= 6 && n >= 8 {
            unsafe {
                simd_gemm::matmul_simd_f32_xnnpack_blocked(&a_vec, &b_vec, &pb_vec, &mut c_vec, m, n, k);
            }
        } else {
            simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        simd_gemm::matmul_scalar_f32(&a_vec, &b_vec, &mut c_vec, m, n, k);
    }

    Float32Array::from(c_vec.as_slice())
}
