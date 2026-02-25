//! WASM bindings for RumPy
//!
//! This crate provides JavaScript/TypeScript bindings for RumPy using wasm-bindgen.
//! It wraps the CPU backend for use in web browsers and Node.js.

use rumpy_cpu::{CpuArray, CpuBackend};
use rumpy_core::{ops::*, Array};
use wasm_bindgen::prelude::*;
use js_sys::Float64Array;

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

    fn inner(&self) -> &CpuArray {
        &self.inner
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

    /// Convert to Float64Array
    #[wasm_bindgen(js_name = toTypedArray)]
    pub fn to_typed_array(&self) -> Float64Array {
        Float64Array::from(self.inner.as_f64_slice().as_slice())
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
    CpuBackend::det(&arr.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))
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
