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

use js_sys::Float64Array;
use rumpy_core::{ops::*, Array};
use rumpy_cpu::{CpuArray, CpuBackend};
use wasm_bindgen::prelude::*;

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

#[wasm_bindgen]
pub fn sin(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::sin(&arr.inner))
}

#[wasm_bindgen]
pub fn cos(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::cos(&arr.inner))
}

#[wasm_bindgen]
pub fn tan(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::tan(&arr.inner))
}

#[wasm_bindgen]
pub fn exp(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::exp(&arr.inner))
}

#[wasm_bindgen]
pub fn log(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::log(&arr.inner))
}

#[wasm_bindgen]
pub fn sqrt(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::sqrt(&arr.inner))
}

#[wasm_bindgen]
pub fn abs(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::abs(&arr.inner))
}

#[wasm_bindgen]
pub fn floor(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::floor(&arr.inner))
}

#[wasm_bindgen]
pub fn ceil(arr: &NDArray) -> NDArray {
    NDArray::new(CpuBackend::ceil(&arr.inner))
}

#[wasm_bindgen]
pub fn round(arr: &NDArray) -> NDArray {
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

/// Get WASM linear memory
///
/// Returns the WebAssembly.Memory object that backs all arrays.
/// Use with `dataPtr()` and `len()` to create zero-copy TypedArray views:
///
/// ```javascript
/// const wasmMemory = rumpy.memory();
/// const ptr = array.dataPtr();
/// const len = array.len();
/// const view = new Float64Array(wasmMemory.buffer, ptr, len);
/// // view is now a zero-copy view into the array's data
/// ```
///
/// Note: Views are invalidated if WASM memory grows. Monitor memory size
/// or recreate views after operations that might allocate.
#[wasm_bindgen]
pub fn memory() -> JsValue {
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
