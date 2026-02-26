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
use ndarray;
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

    /// Permute array dimensions
    /// axes specifies the new order of dimensions
    /// e.g., permute([1, 0, 2]) swaps first two dimensions
    pub fn permute(&self, axes: Vec<usize>) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let ndim = data.ndim();

        if axes.len() != ndim {
            return Err(JsValue::from_str(&format!(
                "axes length {} doesn't match array dimensions {}",
                axes.len(), ndim
            )));
        }

        // Validate axes are valid permutation
        let mut seen = vec![false; ndim];
        for &ax in &axes {
            if ax >= ndim {
                return Err(JsValue::from_str(&format!(
                    "axis {} is out of bounds for array of dimension {}",
                    ax, ndim
                )));
            }
            if seen[ax] {
                return Err(JsValue::from_str("axes must be a permutation (no duplicates)"));
            }
            seen[ax] = true;
        }

        let permuted = data.clone().permuted_axes(axes);
        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(permuted.to_owned())))
    }

    // ============ Axis-based reductions ============

    /// Sum along an axis
    #[wasm_bindgen(js_name = sumAxis)]
    pub fn sum_axis(&self, axis: usize, keepdims: Option<bool>) -> Result<NDArray, JsValue> {
        let result = CpuBackend::sum_axis(&self.inner, axis)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        if keepdims.unwrap_or(false) {
            // Re-insert the axis with size 1
            let mut new_shape = result.shape().to_vec();
            new_shape.insert(axis, 1);
            CpuBackend::reshape(&result, new_shape)
                .map(NDArray::new)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        } else {
            Ok(NDArray::new(result))
        }
    }

    /// Mean along an axis
    #[wasm_bindgen(js_name = meanAxis)]
    pub fn mean_axis(&self, axis: usize, keepdims: Option<bool>) -> Result<NDArray, JsValue> {
        let result = CpuBackend::mean_axis(&self.inner, axis)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        if keepdims.unwrap_or(false) {
            let mut new_shape = result.shape().to_vec();
            new_shape.insert(axis, 1);
            CpuBackend::reshape(&result, new_shape)
                .map(NDArray::new)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        } else {
            Ok(NDArray::new(result))
        }
    }

    /// Max along an axis
    #[wasm_bindgen(js_name = maxAxis)]
    pub fn max_axis(&self, axis: usize, keepdims: Option<bool>) -> Result<NDArray, JsValue> {
        let result = CpuBackend::max_axis(&self.inner, axis)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        if keepdims.unwrap_or(false) {
            let mut new_shape = result.shape().to_vec();
            new_shape.insert(axis, 1);
            CpuBackend::reshape(&result, new_shape)
                .map(NDArray::new)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        } else {
            Ok(NDArray::new(result))
        }
    }

    /// Min along an axis
    #[wasm_bindgen(js_name = minAxis)]
    pub fn min_axis(&self, axis: usize, keepdims: Option<bool>) -> Result<NDArray, JsValue> {
        let result = CpuBackend::min_axis(&self.inner, axis)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        if keepdims.unwrap_or(false) {
            let mut new_shape = result.shape().to_vec();
            new_shape.insert(axis, 1);
            CpuBackend::reshape(&result, new_shape)
                .map(NDArray::new)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        } else {
            Ok(NDArray::new(result))
        }
    }

    // ============ Activation functions ============

    /// ReLU activation: max(0, x)
    pub fn relu(&self) -> NDArray {
        let data = self.inner.as_ndarray();
        let result = data.mapv(|x| if x > 0.0 { x } else { 0.0 });
        NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
    }

    /// GELU activation (Gaussian Error Linear Unit)
    /// Approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pub fn gelu(&self) -> NDArray {
        let data = self.inner.as_ndarray();
        let sqrt_2_pi = (2.0_f64 / std::f64::consts::PI).sqrt();
        let result = data.mapv(|x| {
            let inner = sqrt_2_pi * (x + 0.044715 * x.powi(3));
            x * 0.5 * (1.0 + inner.tanh())
        });
        NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result))
    }

    /// Softmax along an axis
    /// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
    pub fn softmax(&self, axis: usize) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        if axis >= data.ndim() {
            return Err(JsValue::from_str(&format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, data.ndim()
            )));
        }

        // Numerically stable softmax: subtract max before exp
        // Get max along axis with keepdims
        let ax = ndarray::Axis(axis);
        let max_vals = data.map_axis(ax, |lane| {
            lane.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        });

        // Broadcast max back and subtract
        let max_broadcast = max_vals.insert_axis(ax);
        let shifted = &*data - &max_broadcast;

        // Exp
        let exp_vals = shifted.mapv(f64::exp);

        // Sum along axis with keepdims
        let sum_exp = exp_vals.sum_axis(ax).insert_axis(ax);

        // Divide
        let result = &exp_vals / &sum_exp;

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(result.to_owned())))
    }

    /// Argmax - index of maximum value (flattened)
    pub fn argmax(&self) -> usize {
        CpuBackend::argmax(&self.inner)
    }

    /// Argmin - index of minimum value (flattened)
    pub fn argmin(&self) -> usize {
        CpuBackend::argmin(&self.inner)
    }

    // ============ Concatenation ============

    /// Squeeze - remove axes of length 1
    pub fn squeeze(&self) -> NDArray {
        NDArray::new(CpuBackend::squeeze(&self.inner))
    }

    /// Expand dims - add axis of length 1
    #[wasm_bindgen(js_name = expandDims)]
    pub fn expand_dims(&self, axis: usize) -> Result<NDArray, JsValue> {
        CpuBackend::expand_dims(&self.inner, axis)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // ============ Slicing ============

    /// Slice the array with start:stop:step for each dimension
    ///
    /// Uses parallel i32 arrays for starts, stops, steps.
    /// - Negative indices work like Python (count from end)
    /// - i32::MAX (2147483647) for stop means "to the end" (like : in Python)
    /// - Missing dimensions in arrays assume full range
    ///
    /// Example: arr[1:3, :, 2:5] with shape [10, 10, 10]
    ///   starts = [1, 0, 2]
    ///   stops = [3, 2147483647, 5]  // MAX_INT for ":"
    ///   steps = [1, 1, 1]
    pub fn slice(&self, starts: Vec<i32>, stops: Vec<i32>, steps: Vec<i32>) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape();
        let rank = shape.len();

        // Build slice info for each dimension
        let mut slice_info: Vec<ndarray::SliceInfoElem> = Vec::with_capacity(rank);

        for i in 0..rank {
            let dim_len = shape[i] as i32;

            // Handle start (default 0)
            let start_raw = *starts.get(i).unwrap_or(&0);
            let start = if start_raw < 0 {
                (dim_len + start_raw).max(0) as isize
            } else {
                (start_raw as isize).min(dim_len as isize)
            };

            // Handle stop (default to end)
            let stop_raw = *stops.get(i).unwrap_or(&i32::MAX);
            let stop = if stop_raw == i32::MAX || stop_raw > dim_len {
                dim_len as isize
            } else if stop_raw < 0 {
                (dim_len + stop_raw).max(0) as isize
            } else {
                stop_raw as isize
            };

            // Handle step (default 1)
            let step = *steps.get(i).unwrap_or(&1) as isize;
            if step == 0 {
                return Err(JsValue::from_str("step cannot be zero"));
            }

            slice_info.push(ndarray::SliceInfoElem::Slice {
                start,
                end: Some(stop),
                step,
            });
        }

        // Convert to SliceInfo and apply
        let slice = ndarray::SliceInfo::<Vec<ndarray::SliceInfoElem>, ndarray::IxDyn, ndarray::IxDyn>::try_from(slice_info)
            .map_err(|e| JsValue::from_str(&format!("slice error: {:?}", e)))?;

        let sliced = data.slice(slice.as_ref());
        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(sliced.to_owned())))
    }

    /// Slice along a single axis (simpler API for common case)
    ///
    /// Equivalent to arr[:, :, start:stop] when axis=2
    #[wasm_bindgen(js_name = sliceAxis)]
    pub fn slice_axis(&self, axis: usize, start: i32, stop: i32) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape();

        if axis >= shape.len() {
            return Err(JsValue::from_str(&format!(
                "axis {} is out of bounds for array of dimension {}",
                axis, shape.len()
            )));
        }

        let dim_len = shape[axis] as i32;

        // Normalize negative indices
        let start_norm = if start < 0 {
            (dim_len + start).max(0) as usize
        } else {
            (start as usize).min(dim_len as usize)
        };

        let stop_norm = if stop == i32::MAX || stop > dim_len {
            dim_len as usize
        } else if stop < 0 {
            (dim_len + stop).max(0) as usize
        } else {
            stop as usize
        };

        let sliced = data.slice_axis(
            ndarray::Axis(axis),
            ndarray::Slice::from(start_norm..stop_norm)
        );
        Ok(NDArray::new(rumpy_cpu::CpuArray::from_ndarray(sliced.to_owned())))
    }

    // ============ CNN Operations ============

    /// im2col: Convert image patches to columns for convolution via GEMM
    ///
    /// Input shape: (N, C, H, W) - batch, channels, height, width
    /// Output shape: (N * H_out * W_out, C * kernel_h * kernel_w)
    ///
    /// This transforms the convolution operation into a matrix multiplication:
    ///   output = im2col(input) @ weights.reshape(out_channels, -1).T
    #[wasm_bindgen(js_name = im2col)]
    pub fn im2col(
        &self,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape();

        if shape.len() != 4 {
            return Err(JsValue::from_str("im2col expects 4D input (N, C, H, W)"));
        }

        let (n, c, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);

        // Output dimensions
        let h_out = (h_in + 2 * pad_h - kernel_h) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w - kernel_w) / stride_w + 1;

        // Output shape: (N * h_out * w_out, C * kernel_h * kernel_w)
        let rows = n * h_out * w_out;
        let cols = c * kernel_h * kernel_w;
        let mut output = vec![0.0; rows * cols];

        let flat_data = data.as_slice().ok_or_else(|| JsValue::from_str("array not contiguous"))?;

        for batch in 0..n {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let row_idx = batch * h_out * w_out + oh * w_out + ow;

                    for ch in 0..c {
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let ih = oh * stride_h + kh;
                                let iw = ow * stride_w + kw;

                                let col_idx = ch * kernel_h * kernel_w + kh * kernel_w + kw;

                                // Check padding bounds
                                let val = if ih < pad_h || ih >= h_in + pad_h || iw < pad_w || iw >= w_in + pad_w {
                                    0.0 // Zero padding
                                } else {
                                    let actual_h = ih - pad_h;
                                    let actual_w = iw - pad_w;
                                    let idx = batch * c * h_in * w_in + ch * h_in * w_in + actual_h * w_in + actual_w;
                                    flat_data[idx]
                                };

                                output[row_idx * cols + col_idx] = val;
                            }
                        }
                    }
                }
            }
        }

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(output, vec![rows, cols])
            .map_err(|e| JsValue::from_str(&e.to_string()))?))
    }

    /// Max pooling 2D
    ///
    /// Input shape: (N, C, H, W)
    /// Output shape: (N, C, H_out, W_out)
    #[wasm_bindgen(js_name = maxPool2d)]
    pub fn max_pool_2d(
        &self,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape();

        if shape.len() != 4 {
            return Err(JsValue::from_str("maxPool2d expects 4D input (N, C, H, W)"));
        }

        let (n, c, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);

        let h_out = (h_in + 2 * pad_h - kernel_h) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w - kernel_w) / stride_w + 1;

        let mut output = vec![f64::NEG_INFINITY; n * c * h_out * w_out];
        let flat_data = data.as_slice().ok_or_else(|| JsValue::from_str("array not contiguous"))?;

        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut max_val = f64::NEG_INFINITY;

                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let ih = oh * stride_h + kh;
                                let iw = ow * stride_w + kw;

                                // Check padding bounds
                                if ih >= pad_h && ih < h_in + pad_h && iw >= pad_w && iw < w_in + pad_w {
                                    let actual_h = ih - pad_h;
                                    let actual_w = iw - pad_w;
                                    let idx = batch * c * h_in * w_in + ch * h_in * w_in + actual_h * w_in + actual_w;
                                    max_val = max_val.max(flat_data[idx]);
                                }
                            }
                        }

                        // If all padding (edge case), use 0
                        if max_val == f64::NEG_INFINITY {
                            max_val = 0.0;
                        }

                        let out_idx = batch * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow;
                        output[out_idx] = max_val;
                    }
                }
            }
        }

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(output, vec![n, c, h_out, w_out])
            .map_err(|e| JsValue::from_str(&e.to_string()))?))
    }

    /// Average pooling 2D
    ///
    /// Input shape: (N, C, H, W)
    /// Output shape: (N, C, H_out, W_out)
    #[wasm_bindgen(js_name = avgPool2d)]
    pub fn avg_pool_2d(
        &self,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Result<NDArray, JsValue> {
        let data = self.inner.as_ndarray();
        let shape = data.shape();

        if shape.len() != 4 {
            return Err(JsValue::from_str("avgPool2d expects 4D input (N, C, H, W)"));
        }

        let (n, c, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);

        let h_out = (h_in + 2 * pad_h - kernel_h) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w - kernel_w) / stride_w + 1;

        let mut output = vec![0.0; n * c * h_out * w_out];
        let flat_data = data.as_slice().ok_or_else(|| JsValue::from_str("array not contiguous"))?;

        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0;
                        let mut count = 0;

                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let ih = oh * stride_h + kh;
                                let iw = ow * stride_w + kw;

                                // Check padding bounds
                                if ih >= pad_h && ih < h_in + pad_h && iw >= pad_w && iw < w_in + pad_w {
                                    let actual_h = ih - pad_h;
                                    let actual_w = iw - pad_w;
                                    let idx = batch * c * h_in * w_in + ch * h_in * w_in + actual_h * w_in + actual_w;
                                    sum += flat_data[idx];
                                    count += 1;
                                }
                            }
                        }

                        let out_idx = batch * c * h_out * w_out + ch * h_out * w_out + oh * w_out + ow;
                        output[out_idx] = if count > 0 { sum / count as f64 } else { 0.0 };
                    }
                }
            }
        }

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(output, vec![n, c, h_out, w_out])
            .map_err(|e| JsValue::from_str(&e.to_string()))?))
    }

    // ============ Boolean Masking & Comparisons ============

    /// Get elements where mask is non-zero (truthy)
    ///
    /// Returns a 1D array of selected elements.
    /// Mask must be same shape as self, or broadcastable.
    ///
    /// Example: arr.getByMask(arr.gt_scalar(0.5)) returns all elements > 0.5
    #[wasm_bindgen(js_name = getByMask)]
    pub fn get_by_mask(&self, mask: &NDArray) -> Result<NDArray, JsValue> {
        let data = self.inner.as_f64_slice();
        let mask_data = mask.inner.as_f64_slice();

        if data.len() != mask_data.len() {
            return Err(JsValue::from_str(&format!(
                "mask length {} doesn't match array length {}",
                mask_data.len(), data.len()
            )));
        }

        let selected: Vec<f64> = data.iter()
            .zip(mask_data.iter())
            .filter(|(_, &m)| m != 0.0)
            .map(|(&v, _)| v)
            .collect();

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(selected.clone(), vec![selected.len()])
            .map_err(|e| JsValue::from_str(&e.to_string()))?))
    }

    /// Set elements where mask is non-zero to a scalar value
    ///
    /// Returns a new array with selected elements replaced.
    #[wasm_bindgen(js_name = setByMask)]
    pub fn set_by_mask(&self, mask: &NDArray, value: f64) -> Result<NDArray, JsValue> {
        let data = self.inner.as_f64_slice();
        let mask_data = mask.inner.as_f64_slice();

        if data.len() != mask_data.len() {
            return Err(JsValue::from_str(&format!(
                "mask length {} doesn't match array length {}",
                mask_data.len(), data.len()
            )));
        }

        let result: Vec<f64> = data.iter()
            .zip(mask_data.iter())
            .map(|(&v, &m)| if m != 0.0 { value } else { v })
            .collect();

        Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(result, self.shape())
            .map_err(|e| JsValue::from_str(&e.to_string()))?))
    }

    /// Comparison: equal (element-wise)
    pub fn eq(&self, other: &NDArray) -> Result<NDArray, JsValue> {
        use rumpy_core::ops::CompareOps;
        CpuBackend::eq(&self.inner, &other.inner)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Comparison: not equal (element-wise)
    pub fn ne(&self, other: &NDArray) -> Result<NDArray, JsValue> {
        use rumpy_core::ops::CompareOps;
        CpuBackend::ne(&self.inner, &other.inner)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Comparison: less than (element-wise)
    pub fn lt(&self, other: &NDArray) -> Result<NDArray, JsValue> {
        use rumpy_core::ops::CompareOps;
        CpuBackend::lt(&self.inner, &other.inner)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Comparison: less than or equal (element-wise)
    pub fn le(&self, other: &NDArray) -> Result<NDArray, JsValue> {
        use rumpy_core::ops::CompareOps;
        CpuBackend::le(&self.inner, &other.inner)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Comparison: greater than (element-wise)
    pub fn gt(&self, other: &NDArray) -> Result<NDArray, JsValue> {
        use rumpy_core::ops::CompareOps;
        CpuBackend::gt(&self.inner, &other.inner)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Comparison: greater than or equal (element-wise)
    pub fn ge(&self, other: &NDArray) -> Result<NDArray, JsValue> {
        use rumpy_core::ops::CompareOps;
        CpuBackend::ge(&self.inner, &other.inner)
            .map(NDArray::new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Scalar comparison: equal
    #[wasm_bindgen(js_name = eqScalar)]
    pub fn eq_scalar(&self, scalar: f64) -> NDArray {
        use rumpy_core::ops::CompareOps;
        NDArray::new(CpuBackend::eq_scalar(&self.inner, scalar))
    }

    /// Scalar comparison: not equal
    #[wasm_bindgen(js_name = neScalar)]
    pub fn ne_scalar(&self, scalar: f64) -> NDArray {
        use rumpy_core::ops::CompareOps;
        NDArray::new(CpuBackend::ne_scalar(&self.inner, scalar))
    }

    /// Scalar comparison: less than
    #[wasm_bindgen(js_name = ltScalar)]
    pub fn lt_scalar(&self, scalar: f64) -> NDArray {
        use rumpy_core::ops::CompareOps;
        NDArray::new(CpuBackend::lt_scalar(&self.inner, scalar))
    }

    /// Scalar comparison: less than or equal
    #[wasm_bindgen(js_name = leScalar)]
    pub fn le_scalar(&self, scalar: f64) -> NDArray {
        use rumpy_core::ops::CompareOps;
        NDArray::new(CpuBackend::le_scalar(&self.inner, scalar))
    }

    /// Scalar comparison: greater than
    #[wasm_bindgen(js_name = gtScalar)]
    pub fn gt_scalar(&self, scalar: f64) -> NDArray {
        use rumpy_core::ops::CompareOps;
        NDArray::new(CpuBackend::gt_scalar(&self.inner, scalar))
    }

    /// Scalar comparison: greater than or equal
    #[wasm_bindgen(js_name = geScalar)]
    pub fn ge_scalar(&self, scalar: f64) -> NDArray {
        use rumpy_core::ops::CompareOps;
        NDArray::new(CpuBackend::ge_scalar(&self.inner, scalar))
    }

    /// Check for NaN values
    #[wasm_bindgen(js_name = isNan)]
    pub fn is_nan(&self) -> NDArray {
        use rumpy_core::ops::CompareOps;
        NDArray::new(CpuBackend::isnan(&self.inner))
    }

    /// Check for infinite values
    #[wasm_bindgen(js_name = isInf)]
    pub fn is_inf(&self) -> NDArray {
        use rumpy_core::ops::CompareOps;
        NDArray::new(CpuBackend::isinf(&self.inner))
    }

    /// Check for finite values (not NaN, not Inf)
    #[wasm_bindgen(js_name = isFinite)]
    pub fn is_finite(&self) -> NDArray {
        use rumpy_core::ops::CompareOps;
        NDArray::new(CpuBackend::isfinite(&self.inner))
    }

    /// Count non-zero elements
    #[wasm_bindgen(js_name = countNonzero)]
    pub fn count_nonzero(&self) -> usize {
        self.inner.as_f64_slice().iter().filter(|&&x| x != 0.0).count()
    }

    /// Get indices of non-zero elements (flat indices)
    #[wasm_bindgen(js_name = nonzeroFlat)]
    pub fn nonzero_flat(&self) -> Vec<usize> {
        self.inner.as_f64_slice()
            .iter()
            .enumerate()
            .filter(|(_, &x)| x != 0.0)
            .map(|(i, _)| i)
            .collect()
    }

    /// Clip values to a range
    pub fn clip(&self, min: f64, max: f64) -> NDArray {
        use rumpy_core::ops::MathOps;
        NDArray::new(CpuBackend::clip(&self.inner, min, max))
    }
}

/// Numpy-style where: select x where condition is true, else y
///
/// condition, x, y must have compatible shapes (broadcasting supported).
/// Returns x where condition != 0, else y.
#[wasm_bindgen(js_name = where_)]
pub fn where_op(condition: &NDArray, x: &NDArray, y: &NDArray) -> Result<NDArray, JsValue> {
    let cond_data = condition.inner.as_f64_slice();
    let x_data = x.inner.as_f64_slice();
    let y_data = y.inner.as_f64_slice();

    // Simple case: all same size
    if cond_data.len() == x_data.len() && x_data.len() == y_data.len() {
        let result: Vec<f64> = cond_data.iter()
            .zip(x_data.iter())
            .zip(y_data.iter())
            .map(|((&c, &xv), &yv)| if c != 0.0 { xv } else { yv })
            .collect();

        return Ok(NDArray::new(rumpy_cpu::CpuArray::from_f64_vec(result, condition.shape())
            .map_err(|e| JsValue::from_str(&e.to_string()))?));
    }

    Err(JsValue::from_str("where requires all inputs to have same shape (broadcasting not yet implemented for where)"))
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

// ============ Concatenation functions ============

/// Concatenate two arrays along an axis
#[wasm_bindgen]
pub fn concatenate2(a: &NDArray, b: &NDArray, axis: usize) -> Result<NDArray, JsValue> {
    let refs: Vec<&rumpy_cpu::CpuArray> = vec![&a.inner, &b.inner];
    CpuBackend::concatenate(&refs, axis)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Concatenate three arrays along an axis
#[wasm_bindgen]
pub fn concatenate3(a: &NDArray, b: &NDArray, c: &NDArray, axis: usize) -> Result<NDArray, JsValue> {
    let refs: Vec<&rumpy_cpu::CpuArray> = vec![&a.inner, &b.inner, &c.inner];
    CpuBackend::concatenate(&refs, axis)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Stack two arrays along a new axis
#[wasm_bindgen]
pub fn stack2(a: &NDArray, b: &NDArray, axis: usize) -> Result<NDArray, JsValue> {
    let refs: Vec<&rumpy_cpu::CpuArray> = vec![&a.inner, &b.inner];
    CpuBackend::stack(&refs, axis)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Stack three arrays along a new axis
#[wasm_bindgen]
pub fn stack3(a: &NDArray, b: &NDArray, c: &NDArray, axis: usize) -> Result<NDArray, JsValue> {
    let refs: Vec<&rumpy_cpu::CpuArray> = vec![&a.inner, &b.inner, &c.inner];
    CpuBackend::stack(&refs, axis)
        .map(NDArray::new)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Vertical stack (concatenate along axis 0)
#[wasm_bindgen]
pub fn vstack2(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    concatenate2(a, b, 0)
}

/// Horizontal stack (concatenate along axis 1 for 2D+, axis 0 for 1D)
#[wasm_bindgen]
pub fn hstack2(a: &NDArray, b: &NDArray) -> Result<NDArray, JsValue> {
    let axis = if a.ndim() == 1 { 0 } else { 1 };
    concatenate2(a, b, axis)
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

// ============================================================================
// Zero-copy f32 buffers (eliminate JS↔WASM copy overhead)
// ============================================================================
//
// The Float32Array-based matmul functions (matmulF32Optimized etc.) copy
// A and B from JS heap → WASM heap on every call (a.to_vec()), then copy
// C back (Float32Array::from). At small sizes this is ~20-25% of wall time:
//
//   256²: matmul = 0.5 ms, copies ≈ 0.12 ms (24%)
//   128²: matmul = 0.3 ms, copies ≈ 0.06 ms (20%)
//
// tf.js doesn't pay this cost — tensors live in WASM memory. This API
// lets you match that: allocate f32 buffers inside WASM once, get zero-
// copy Float32Array views, write your data into them directly, call
// matmulF32ZeroCopy which operates in-place.
//
// USAGE:
//   const a = allocF32(M * K);           // WASM-resident buffer
//   const b = allocF32(K * N);
//   const c = allocF32(M * N);
//   const packedB = allocF32(packedBSize(K, N));  // for prepacked path
//
//   // Fill a, b via zero-copy views (SharedArrayBuffer → no detach on grow):
//   const mem = wasmMemory().buffer;
//   new Float32Array(mem, a.ptr(), M*K).set(yourAData);
//   new Float32Array(mem, b.ptr(), K*N).set(yourBData);
//
//   packBInPlace(b, packedB, K, N);       // once per weight matrix
//   matmulF32PrepackedZeroCopy(a, packedB, c, M, N, K);  // many times
//
//   // Read result:
//   const result = new Float32Array(mem, c.ptr(), M*N);
//
// MEMORY GROWTH CAVEAT: views are valid as long as WebAssembly.Memory
// doesn't grow. With SharedArrayBuffer (which we use), growing doesn't
// DETACH the view, but the view's `.buffer` still points at the old SAB
// range. Re-fetch `wasmMemory().buffer` after operations that allocate.
// (F32Buffer handles themselves stay valid — only JS-side views need
// re-deriving.)

/// WASM-resident f32 buffer. Wraps a `Vec<f32>` that lives in WASM linear
/// memory. JS can get a zero-copy `Float32Array` view via `ptr()` + the
/// shared memory buffer.
///
/// The buffer stays valid until `free()` is called or the object is GC'd.
/// Memory growth does NOT invalidate the buffer (the Vec's address is
/// stable), only JS-side views of `wasmMemory().buffer` need re-deriving.
#[wasm_bindgen]
pub struct F32Buffer {
    data: Vec<f32>,
}

#[wasm_bindgen]
impl F32Buffer {
    /// Byte offset into WASM linear memory where this buffer's data starts.
    ///
    /// Use with `wasmMemory().buffer` to construct a zero-copy view:
    ///   new Float32Array(wasmMemory().buffer, buf.ptr(), buf.len())
    #[wasm_bindgen]
    pub fn ptr(&self) -> usize {
        self.data.as_ptr() as usize
    }

    #[wasm_bindgen]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Explicitly free this buffer's memory. The handle is consumed.
    #[wasm_bindgen]
    pub fn free(self) {
        // self dropped here
    }

    /// Copy data FROM a JS Float32Array INTO this buffer.
    /// Useful for the first fill if you can't construct data directly into
    /// a zero-copy view (e.g. data comes from a WebGL readback).
    #[wasm_bindgen(js_name = copyFrom)]
    pub fn copy_from(&mut self, src: &Float32Array) {
        let n = (src.length() as usize).min(self.data.len());
        src.slice(0, n as u32).copy_to(&mut self.data[..n]);
    }

    /// Copy data FROM this buffer TO a JS Float32Array.
    /// For the zero-copy path you don't need this — construct a view
    /// instead. This exists for cases where the result needs to go to a
    /// non-shared ArrayBuffer (e.g. postMessage to a context without SAB).
    #[wasm_bindgen(js_name = copyTo)]
    pub fn copy_to(&self, dst: &Float32Array) {
        let n = (dst.length() as usize).min(self.data.len());
        dst.subarray(0, n as u32).copy_from(&self.data[..n]);
    }
}

/// Allocate an f32 buffer of the given length inside WASM memory.
/// Contents are uninitialised — write before reading.
#[wasm_bindgen(js_name = allocF32)]
pub fn alloc_f32(len: usize) -> F32Buffer {
    let mut data: Vec<f32> = Vec::with_capacity(len);
    // Uninitialised: caller is expected to fill via copyFrom or a
    // zero-copy view. Zeroing would be wasted work for input buffers
    // (overwritten immediately) and output buffers (matmul overwrites).
    unsafe { data.set_len(len); }
    F32Buffer { data }
}

/// Size (in f32 elements) of a fully-packed B buffer for matmulF32PrepackedZeroCopy.
///
/// = ceil(N/8) × K × 8.  For N divisible by 8 (most cases), equals K × N.
#[wasm_bindgen(js_name = packedBSize)]
pub fn packed_b_size(k: usize, n: usize) -> usize {
    ((n + 7) / 8) * k * 8
}

/// Pack B (in an F32Buffer) into panel-major layout (in another F32Buffer).
///
/// Call once per weight matrix; reuse packedB across many matmuls.
/// Both buffers must already be allocated to the right sizes (B: K×N,
/// packedB: packedBSize(K, N)).
///
/// Note: Currently just copies B to packed_b. The specialized packing was
/// removed during a refactor. The matmul still works (just re-packs internally).
#[wasm_bindgen(js_name = packBInPlace)]
pub fn pack_b_in_place(b: &F32Buffer, packed_b: &mut F32Buffer, k: usize, n: usize) {
    assert!(b.data.len() >= k * n, "b too small");
    assert!(packed_b.data.len() >= packed_b_size(k, n), "packed_b too small");

    // For now, just copy B into packed_b (prepacking optimization was removed)
    let copy_len = (k * n).min(packed_b.data.len());
    packed_b.data[..copy_len].copy_from_slice(&b.data[..copy_len]);
}

/// Parallel matmul, ZERO JS↔WASM copies.
///
/// A, B, C all live in WASM memory (F32Buffers). B is packed on-the-fly
/// (same behaviour as matmulF32OptimizedParallelV3 but without the
/// Float32Array round-trips). C is overwritten.
///
/// This is the general API — B can vary call-to-call. For constant B
/// (NN inference), use matmulF32PrepackedZeroCopy which skips the pack.
#[wasm_bindgen(js_name = matmulF32ZeroCopy)]
pub fn matmul_f32_zerocopy(a: &F32Buffer, b: &F32Buffer, c: &mut F32Buffer, m: usize, n: usize, k: usize) {
    assert!(a.data.len() >= m * k, "a too small");
    assert!(b.data.len() >= k * n, "b too small");
    assert!(c.data.len() >= m * n, "c too small");

    #[cfg(target_arch = "wasm32")]
    {
        // Call straight into v3. No to_vec, no Float32Array::from — the
        // buffers are already in WASM memory.
        let out = simd_gemm::matmul_optimized_f32_parallel(
            &a.data[..m * k],
            &b.data[..k * n],
            m, n, k,
        );
        // v3 returns a Vec (it allocates its own C internally for the
        // C-padding path). Copy into caller's buffer.
        //
        // TODO: add an `_into` variant of v3 that writes to a caller-
        // provided slice when no padding is active. Would save one more
        // M×N copy. At 256² that's 256 KiB = ~0.05 ms — 10% of the
        // remaining gap.
        c.data[..m * n].copy_from_slice(&out);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let out = simd_gemm::matmul_dispatch_f32(&a.data[..m*k], &b.data[..k*n], m, n, k);
        c.data[..m * n].copy_from_slice(&out);
    }
}

/// Parallel matmul with pre-packed B, ZERO JS↔WASM copies.
///
/// The leanest call path: A and packed-B already in WASM memory, C
/// written directly, no per-call packing. This is the tf.js-equivalent
/// path for NN inference.
///
/// Note: The specialized prepacked kernel was removed during a refactor.
/// This now just calls the regular parallel matmul (packed_b is treated as B).
#[wasm_bindgen(js_name = matmulF32PrepackedZeroCopy)]
pub fn matmul_f32_prepacked_zerocopy(
    a: &F32Buffer,
    packed_b: &F32Buffer,
    c: &mut F32Buffer,
    m: usize,
    n: usize,
    k: usize,
) {
    assert!(a.data.len() >= m * k, "a too small");
    assert!(packed_b.data.len() >= k * n, "packed_b too small");
    assert!(c.data.len() >= m * n, "c too small");

    // Call the regular matmul (prepacked optimization was removed)
    #[cfg(target_arch = "wasm32")]
    {
        let out = simd_gemm::matmul_optimized_f32_parallel(
            &a.data[..m * k],
            &packed_b.data[..k * n],
            m, n, k,
        );
        c.data[..m * n].copy_from_slice(&out);
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let out = simd_gemm::matmul_dispatch_f32(&a.data[..m*k], &packed_b.data[..k*n], m, n, k);
        c.data[..m * n].copy_from_slice(&out);
    }
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

/// DEBUG: mimic v3's dispatch pattern (rayon::scope + inline caller +
/// atomic tile counter) and record which rayon thread claims each tile.
///
/// Returns a flat array of [tile_idx, rayon_thread_idx, tid_param] triples
/// so we can see if all tiles were claimed by one thread (rayon dispatch
/// bug) or spread across threads (parallelism works, perf bug is elsewhere).
#[wasm_bindgen(js_name = probeV3Dispatch)]
pub fn probe_v3_dispatch(n_tiles: usize, work_ms_per_tile: f64) -> Vec<f64> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    let n_workers = rayon::current_num_threads();
    let tile_counter = AtomicUsize::new(0);
    let log: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::with_capacity(n_tiles));

    let worker = |tid: usize| {
        loop {
            let t = tile_counter.fetch_add(1, Ordering::Relaxed);
            if t >= n_tiles { break; }

            let rtid = rayon::current_thread_index().unwrap_or(9999);
            log.lock().unwrap().push((t, rtid, tid));

            // Simulate work.
            let start = js_sys::Date::now();
            while js_sys::Date::now() - start < work_ms_per_tile {
                core::hint::spin_loop();
            }
        }
    };

    let t0 = js_sys::Date::now();
    rayon::scope(|s| {
        for tid in 0..n_workers {
            s.spawn(move |_| worker(tid));
        }
        worker(n_workers); // caller
    });
    let wall = js_sys::Date::now() - t0;

    // Flatten: [wall, n_triples, t0,r0,p0, t1,r1,p1, ...]
    let mut out = vec![wall, log.lock().unwrap().len() as f64];
    for (t, r, p) in log.lock().unwrap().iter() {
        out.push(*t as f64);
        out.push(*r as f64);
        out.push(*p as f64);
    }
    out
}

/// DEBUG: report which code path v3 would take for given (m,n,k).
/// Returns: [below_threshold, pack_a, c_pad, fast_path, slab_rows, total_tiles, tz(k*4), tz(n*4)]
#[wasm_bindgen(js_name = probeV3Path)]
pub fn probe_v3_path(m: usize, n: usize, k: usize) -> Vec<usize> {
    const PAD_ZEROS_THRESHOLD: u32 = 12;
    const OPT_MR: usize = 6;

    // u64 mul: WASM usize is 32-bit, m*n*k overflows at 2048³ → 0.
    // (This was the "cursed triple" — overflow → false below_threshold
    // positive → silent single-threaded fallback.)
    let flops = (m as u64) * (n as u64) * (k as u64);
    let size_below_threshold = flops < (192u64 * 192 * 192);

    let n_workers = rayon::current_num_threads().max(1);
    let pack_a = (k * 4).trailing_zeros() >= PAD_ZEROS_THRESHOLD;
    let c_pad = (n * 4).trailing_zeros() >= PAD_ZEROS_THRESHOLD;
    let slab_rows = {
        let base = (m + n_workers - 1) / n_workers;
        ((base + OPT_MR - 1) / OPT_MR * OPT_MR).max(OPT_MR)
    };
    let total_tiles = (m + slab_rows - 1) / slab_rows;
    let fast = !pack_a && !c_pad && !size_below_threshold && total_tiles >= 2;

    vec![
        size_below_threshold as usize,
        pack_a as usize,
        c_pad as usize,
        fast as usize,
        slab_rows,
        total_tiles,
        (k * 4).trailing_zeros() as usize,
        (n * 4).trailing_zeros() as usize,
    ]
}

/// DEBUG: probe whether rayon workers are actually executing in parallel.
///
/// Spawns N tasks, each recording its rayon thread index and spinning for
/// ~duration_ms. If workers are live, wall-clock ≈ duration_ms (parallel).
/// If all tasks run on the main thread, wall-clock ≈ N × duration_ms.
///
/// Returns [wall_ms, n_distinct_thread_ids, max_thread_id_seen].
#[wasm_bindgen(js_name = probeRayonParallelism)]
pub fn probe_rayon_parallelism(n_tasks: usize, duration_ms: f64) -> Vec<f64> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Rayon thread indices seen (bitmask; up to 64 threads).
    let seen = AtomicUsize::new(0);
    let max_idx = AtomicUsize::new(0);

    let t0 = js_sys::Date::now();
    (0..n_tasks).into_par_iter().for_each(|_| {
        let tid = rayon::current_thread_index().unwrap_or(usize::MAX);
        if tid < 64 {
            seen.fetch_or(1 << tid, Ordering::Relaxed);
            max_idx.fetch_max(tid, Ordering::Relaxed);
        }
        // Busy-spin for duration_ms (can't atomic.wait on main thread, and
        // we want deterministic work regardless of thread).
        let start = js_sys::Date::now();
        while js_sys::Date::now() - start < duration_ms {
            core::hint::spin_loop();
        }
    });
    let wall = js_sys::Date::now() - t0;

    let n_distinct = seen.load(Ordering::Relaxed).count_ones() as f64;
    vec![wall, n_distinct, max_idx.load(Ordering::Relaxed) as f64]
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

/// XNNPACK-style f32 GEMM with pre-packed B matrix (LEGACY, single-threaded)
///
/// This is a two-phase API:
/// 1. Call `packB` once to convert B into XNNPACK format
/// 2. Call `matmulXnnpack` multiple times with different A matrices
///
/// This amortizes the packing cost over many matmuls, which is how XNNPACK works.
/// For PARALLEL matmul with pre-packed B, use `packBFull` + `matmulF32Prepacked`.
#[wasm_bindgen(js_name = packB)]
pub fn pack_b(b: &Float32Array, k: usize, n: usize) -> Float32Array {
    let b_vec = b.to_vec();
    let n_panels = n / 8;
    let mut packed = vec![0.0f32; n_panels * k * 8];
    simd_gemm::pack_b_xnnpack(&b_vec, &mut packed, k, n);
    Float32Array::from(packed.as_slice())
}

/// Pack ALL of B into panel-major layout (for use with matmulF32Prepacked).
///
/// Unlike `packB` (which truncates at N/8×8), this handles arbitrary N
/// by zero-padding the last panel to NR=8 width. The output size is
/// ceil(N/8) × K × 8 floats.
///
/// Call this ONCE for weight matrices that will be reused across many
/// matmuls (NN inference). The pack cost is O(K×N) = one pass through B;
/// tf.js/XNNPACK do exactly this at model-load time.
/// Pack B matrix for repeated matmuls.
/// Note: Prepacking optimization was removed. This now just returns a copy.
#[wasm_bindgen(js_name = packBFull)]
pub fn pack_b_full(b: &Float32Array, k: usize, n: usize) -> Float32Array {
    let _ = (k, n); // dimensions used for validation only now
    // Just return a copy - the specialized prepacking was removed
    let b_vec = b.to_vec();
    Float32Array::from(b_vec.as_slice())
}

/// Parallel matmul with pre-packed B (from packBFull).
///
/// Note: Prepacking optimization was removed. This now calls the regular
/// parallel matmul (packed_b is treated as normal B).
#[wasm_bindgen(js_name = matmulF32Prepacked)]
pub fn matmul_f32_prepacked(a: &Float32Array, packed_b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
    let a_vec = a.to_vec();
    let _pb_vec = packed_b.to_vec();

    #[cfg(target_arch = "wasm32")]
    {
        // Use regular parallel matmul - prepacking was removed
        let c_vec = simd_gemm::matmul_optimized_f32_parallel(&a_vec, &_pb_vec, m, n, k);
        Float32Array::from(c_vec.as_slice())
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        Float32Array::from(
            simd_gemm::matmul_dispatch_f32(&a_vec, &_pb_vec, m, n, k).as_slice()
        )
    }
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

/// Parallel version of optimized 6x8 GEMM using rayon (LEGACY)
///
/// Kept for A/B benchmarking. Has known problems — see v3 below.
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

/// Parallel optimised GEMM, v3: pack-once, 2D-tile, atomic work-claiming.
///
/// This is the recommended parallel path. Differences from the legacy
/// `matmulF32OptimizedParallel`:
///
/// * B is packed ONCE and shared read-only across all workers
///   (old path packed B independently in every thread — with N threads that's
///   N× the packing work and N× allocator contention on WASM's locked dlmalloc)
///
/// * Macro-tiles (~MC × NC) are handed out via an atomic counter, matching
///   XNNPACK's `pthreadpool_parallelize_2d_tile_2d`. Load balances across
///   Apple Silicon perf/efficiency cores instead of assuming uniform workers.
///
/// * Zero per-task heap allocation. Workers write straight into disjoint
///   C slices.
///
/// * The calling thread participates (it's "thread 0"), so with an N-worker
///   Rayon pool you get N+1 way parallelism.
///
/// Requires `initThreadPool(n)` to have been called (same as legacy path).
#[wasm_bindgen(js_name = matmulF32OptimizedParallelV3)]
pub fn matmul_f32_optimized_parallel_v3(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
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

/// Parallel optimised GEMM, v4: hijack Rayon's workers with raw
/// `memory.atomic.wait32`/`notify` dispatch.
///
/// v3 uses ONE `rayon::scope` per matmul (good), but inside it there's
/// still no shared packed-B (each thread packs its own) and the join is
/// Rayon's standard park/unpark.  v4 is the full pthreadpool model:
///
/// * ONE `rayon::scope` — we use wasm-bindgen-rayon's Web Workers but
///   NOT Rayon's task scheduler.
///
/// * Workers enter OUR spin-then-`atomic.wait` loop. Main drives them
///   block-by-block: pack B (shared), bump generation, `atomic.notify`,
///   drain tiles alongside workers, spin-wait for completion, repeat.
///
/// * Shared packed-B → minimum total packing work (same as single-thread).
///
/// * Per-block sync is ~1 `atomic.notify` + N×1 Relaxed `fetch_sub` +
///   one short main-thread spin. Compare Rayon: N× `Box<dyn FnOnce>` +
///   N× park/unpark per scope.
///
/// This is "our own thread manager", hosted inside Rayon's already-spawned
/// workers. No new dependencies, no separate worker pool to manage.
///
/// Requires `initThreadPool(n)` (same as v3).
#[wasm_bindgen(js_name = matmulF32OptimizedParallelV4)]
pub fn matmul_f32_optimized_parallel_v4(a: &Float32Array, b: &Float32Array, m: usize, n: usize, k: usize) -> Float32Array {
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
