use wasm_bindgen::prelude::*;

// ============ Unary Operations ============

#[wasm_bindgen]
pub fn unary_sin(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.sin()).collect()
}

#[wasm_bindgen]
pub fn unary_cos(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.cos()).collect()
}

#[wasm_bindgen]
pub fn unary_tan(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.tan()).collect()
}

#[wasm_bindgen]
pub fn unary_asin(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.asin()).collect()
}

#[wasm_bindgen]
pub fn unary_acos(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.acos()).collect()
}

#[wasm_bindgen]
pub fn unary_atan(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.atan()).collect()
}

#[wasm_bindgen]
pub fn unary_sinh(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.sinh()).collect()
}

#[wasm_bindgen]
pub fn unary_cosh(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.cosh()).collect()
}

#[wasm_bindgen]
pub fn unary_tanh(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.tanh()).collect()
}

#[wasm_bindgen]
pub fn unary_exp(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.exp()).collect()
}

#[wasm_bindgen]
pub fn unary_log(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.ln()).collect()
}

#[wasm_bindgen]
pub fn unary_log2(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.log2()).collect()
}

#[wasm_bindgen]
pub fn unary_log10(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.log10()).collect()
}

#[wasm_bindgen]
pub fn unary_sqrt(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.sqrt()).collect()
}

#[wasm_bindgen]
pub fn unary_cbrt(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.cbrt()).collect()
}

#[wasm_bindgen]
pub fn unary_abs(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.abs()).collect()
}

#[wasm_bindgen]
pub fn unary_ceil(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.ceil()).collect()
}

#[wasm_bindgen]
pub fn unary_floor(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.floor()).collect()
}

/// Banker's rounding (round half to even), matching NumPy's behavior.
fn bankers_round(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let floor = x.floor();
    let frac = x - floor;
    // Check if exactly halfway
    if (frac - 0.5).abs() < f64::EPSILON * x.abs().max(1.0) {
        // Round to even
        let floor_i = floor as i64;
        if floor_i % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        x.round()
    }
}

#[wasm_bindgen]
pub fn unary_round(data: &[f64]) -> Vec<f64> {
    data.iter().map(|&x| bankers_round(x)).collect()
}

#[wasm_bindgen]
pub fn unary_sign(data: &[f64]) -> Vec<f64> {
    data.iter()
        .map(|&x| {
            if x.is_nan() {
                f64::NAN
            } else if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
        .collect()
}

#[wasm_bindgen]
pub fn unary_negative(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| -x).collect()
}

#[wasm_bindgen]
pub fn unary_reciprocal(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| 1.0 / x).collect()
}

#[wasm_bindgen]
pub fn unary_square(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x * x).collect()
}

#[wasm_bindgen]
pub fn unary_expm1(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.exp_m1()).collect()
}

#[wasm_bindgen]
pub fn unary_log1p(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.ln_1p()).collect()
}

#[wasm_bindgen]
pub fn unary_trunc(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.trunc()).collect()
}

#[wasm_bindgen]
pub fn unary_asinh(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.asinh()).collect()
}

#[wasm_bindgen]
pub fn unary_acosh(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.acosh()).collect()
}

#[wasm_bindgen]
pub fn unary_atanh(data: &[f64]) -> Vec<f64> {
    data.iter().map(|x| x.atanh()).collect()
}

// ============ Binary Operations with Broadcasting ============

/// Core broadcasting logic for binary operations.
/// Takes two data slices with their shapes and applies the given operation.
fn binary_op(
    a_data: &[f64],
    a_shape: &[u32],
    b_data: &[f64],
    b_shape: &[u32],
    op: impl Fn(f64, f64) -> f64,
) -> Vec<f64> {
    let a_len = a_shape.len();
    let b_len = b_shape.len();

    // Fast path: same shape
    if a_len == b_len && a_shape == b_shape {
        let mut result = Vec::with_capacity(a_data.len());
        for i in 0..a_data.len() {
            result.push(op(a_data[i], b_data[i]));
        }
        return result;
    }

    // Fast path: scalar broadcast (one side has 1 element)
    if a_data.len() == 1 {
        let av = a_data[0];
        return b_data.iter().map(|&bv| op(av, bv)).collect();
    }
    if b_data.len() == 1 {
        let bv = b_data[0];
        return a_data.iter().map(|&av| op(av, bv)).collect();
    }

    // Full broadcasting
    let ndim = a_len.max(b_len);
    let mut a_padded = vec![1u32; ndim];
    let mut b_padded = vec![1u32; ndim];

    for i in 0..a_len {
        a_padded[ndim - a_len + i] = a_shape[i];
    }
    for i in 0..b_len {
        b_padded[ndim - b_len + i] = b_shape[i];
    }

    let mut out_shape = vec![0u32; ndim];
    for i in 0..ndim {
        out_shape[i] = if a_padded[i] == b_padded[i] {
            a_padded[i]
        } else if a_padded[i] == 1 {
            b_padded[i]
        } else if b_padded[i] == 1 {
            a_padded[i]
        } else {
            panic!(
                "operands could not be broadcast together with shapes {:?} {:?}",
                a_shape, b_shape
            );
        };
    }

    let total: usize = out_shape.iter().map(|&x| x as usize).product();
    let mut result = Vec::with_capacity(total);

    // Compute strides
    let mut a_strides = vec![0usize; ndim];
    let mut b_strides = vec![0usize; ndim];
    let mut out_strides = vec![0usize; ndim];
    let (mut a_s, mut b_s, mut o_s) = (1usize, 1usize, 1usize);
    for i in (0..ndim).rev() {
        a_strides[i] = if a_padded[i] == 1 { 0 } else { a_s };
        b_strides[i] = if b_padded[i] == 1 { 0 } else { b_s };
        out_strides[i] = o_s;
        a_s *= a_padded[i] as usize;
        b_s *= b_padded[i] as usize;
        o_s *= out_shape[i] as usize;
    }

    for idx in 0..total {
        let mut ai = 0usize;
        let mut bi = 0usize;
        let mut tmp = idx;
        for d in 0..ndim {
            let coord = tmp / out_strides[d];
            tmp %= out_strides[d];
            ai += coord * a_strides[d];
            bi += coord * b_strides[d];
        }
        result.push(op(a_data[ai], b_data[bi]));
    }

    result
}

/// Returns the broadcast output shape as a Vec<u32>.
/// Used by the TypeScript side to know the result shape.
#[wasm_bindgen]
pub fn broadcast_shape(a_shape: &[u32], b_shape: &[u32]) -> Vec<u32> {
    let ndim = a_shape.len().max(b_shape.len());
    let mut a_padded = vec![1u32; ndim];
    let mut b_padded = vec![1u32; ndim];

    for i in 0..a_shape.len() {
        a_padded[ndim - a_shape.len() + i] = a_shape[i];
    }
    for i in 0..b_shape.len() {
        b_padded[ndim - b_shape.len() + i] = b_shape[i];
    }

    let mut out = vec![0u32; ndim];
    for i in 0..ndim {
        out[i] = if a_padded[i] == b_padded[i] {
            a_padded[i]
        } else if a_padded[i] == 1 {
            b_padded[i]
        } else if b_padded[i] == 1 {
            a_padded[i]
        } else {
            panic!(
                "operands could not be broadcast together with shapes {:?} {:?}",
                a_shape, b_shape
            );
        };
    }
    out
}

#[wasm_bindgen]
pub fn binary_add(a_data: &[f64], a_shape: &[u32], b_data: &[f64], b_shape: &[u32]) -> Vec<f64> {
    binary_op(a_data, a_shape, b_data, b_shape, |x, y| x + y)
}

#[wasm_bindgen]
pub fn binary_subtract(
    a_data: &[f64],
    a_shape: &[u32],
    b_data: &[f64],
    b_shape: &[u32],
) -> Vec<f64> {
    binary_op(a_data, a_shape, b_data, b_shape, |x, y| x - y)
}

#[wasm_bindgen]
pub fn binary_multiply(
    a_data: &[f64],
    a_shape: &[u32],
    b_data: &[f64],
    b_shape: &[u32],
) -> Vec<f64> {
    binary_op(a_data, a_shape, b_data, b_shape, |x, y| x * y)
}

#[wasm_bindgen]
pub fn binary_divide(
    a_data: &[f64],
    a_shape: &[u32],
    b_data: &[f64],
    b_shape: &[u32],
) -> Vec<f64> {
    binary_op(a_data, a_shape, b_data, b_shape, |x, y| x / y)
}

#[wasm_bindgen]
pub fn binary_power(
    a_data: &[f64],
    a_shape: &[u32],
    b_data: &[f64],
    b_shape: &[u32],
) -> Vec<f64> {
    binary_op(a_data, a_shape, b_data, b_shape, |x, y| x.powf(y))
}

#[wasm_bindgen]
pub fn binary_maximum(
    a_data: &[f64],
    a_shape: &[u32],
    b_data: &[f64],
    b_shape: &[u32],
) -> Vec<f64> {
    binary_op(a_data, a_shape, b_data, b_shape, |x, y| {
        // NaN propagation like NumPy
        if x.is_nan() || y.is_nan() {
            f64::NAN
        } else {
            x.max(y)
        }
    })
}

#[wasm_bindgen]
pub fn binary_minimum(
    a_data: &[f64],
    a_shape: &[u32],
    b_data: &[f64],
    b_shape: &[u32],
) -> Vec<f64> {
    binary_op(a_data, a_shape, b_data, b_shape, |x, y| {
        if x.is_nan() || y.is_nan() {
            f64::NAN
        } else {
            x.min(y)
        }
    })
}

#[wasm_bindgen]
pub fn binary_mod(
    a_data: &[f64],
    a_shape: &[u32],
    b_data: &[f64],
    b_shape: &[u32],
) -> Vec<f64> {
    binary_op(a_data, a_shape, b_data, b_shape, |x, y| {
        let r = x % y;
        // NumPy mod: result has the sign of the divisor
        if r != 0.0 && r.signum() != y.signum() {
            r + y
        } else {
            r
        }
    })
}

// ============ Reductions ============

#[wasm_bindgen]
pub fn reduce_sum(data: &[f64]) -> f64 {
    data.iter().sum()
}

#[wasm_bindgen]
pub fn reduce_prod(data: &[f64]) -> f64 {
    data.iter().product()
}

#[wasm_bindgen]
pub fn reduce_min(data: &[f64]) -> f64 {
    let mut result = f64::INFINITY;
    for &v in data {
        if v.is_nan() { return f64::NAN; }
        if v < result { result = v; }
    }
    result
}

#[wasm_bindgen]
pub fn reduce_max(data: &[f64]) -> f64 {
    let mut result = f64::NEG_INFINITY;
    for &v in data {
        if v.is_nan() { return f64::NAN; }
        if v > result { result = v; }
    }
    result
}

#[wasm_bindgen]
pub fn reduce_mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let sum: f64 = data.iter().sum();
    sum / data.len() as f64
}

// ============ Axis Reductions ============

/// Generic reduce along a single axis.
fn reduce_axis_impl(
    data: &[f64],
    shape: &[u32],
    axis: usize,
    init: f64,
    fold: impl Fn(f64, f64) -> f64,
    finalize: impl Fn(f64, usize) -> f64,
) -> Vec<f64> {
    let ndim = shape.len();
    assert!(axis < ndim, "axis {} out of bounds for ndim {}", axis, ndim);

    let axis_len = shape[axis] as usize;

    // Compute output shape (shape with axis removed)
    let mut out_shape: Vec<usize> = Vec::with_capacity(ndim - 1);
    for i in 0..ndim {
        if i != axis {
            out_shape.push(shape[i] as usize);
        }
    }

    let out_total: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };

    // Compute input strides
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as usize;
    }

    let axis_stride = strides[axis];

    // For each output element, iterate along the axis
    let mut result = vec![init; out_total];

    // Compute output strides (for mapping output index -> input index)
    let mut out_strides = vec![1usize; out_shape.len()];
    if !out_shape.is_empty() {
        for i in (0..out_shape.len() - 1).rev() {
            out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
        }
    }

    for out_idx in 0..out_total {
        // Convert out_idx to multi-dimensional coords in output space
        let mut tmp = out_idx;
        let mut base_input_idx = 0usize;
        let mut out_dim = 0;
        for d in 0..ndim {
            if d == axis {
                continue;
            }
            let coord = tmp / out_strides[out_dim];
            tmp %= out_strides[out_dim];
            base_input_idx += coord * strides[d];
            out_dim += 1;
        }

        // Reduce along the axis
        let mut acc = init;
        for k in 0..axis_len {
            let input_idx = base_input_idx + k * axis_stride;
            acc = fold(acc, data[input_idx]);
        }
        result[out_idx] = finalize(acc, axis_len);
    }

    result
}

#[wasm_bindgen]
pub fn reduce_sum_axis(data: &[f64], shape: &[u32], axis: u32) -> Vec<f64> {
    reduce_axis_impl(data, shape, axis as usize, 0.0, |a, b| a + b, |v, _| v)
}

#[wasm_bindgen]
pub fn reduce_prod_axis(data: &[f64], shape: &[u32], axis: u32) -> Vec<f64> {
    reduce_axis_impl(data, shape, axis as usize, 1.0, |a, b| a * b, |v, _| v)
}

#[wasm_bindgen]
pub fn reduce_min_axis(data: &[f64], shape: &[u32], axis: u32) -> Vec<f64> {
    reduce_axis_impl(
        data,
        shape,
        axis as usize,
        f64::INFINITY,
        |a, b| if b.is_nan() { f64::NAN } else if a.is_nan() { f64::NAN } else if b < a { b } else { a },
        |v, _| v,
    )
}

#[wasm_bindgen]
pub fn reduce_max_axis(data: &[f64], shape: &[u32], axis: u32) -> Vec<f64> {
    reduce_axis_impl(
        data,
        shape,
        axis as usize,
        f64::NEG_INFINITY,
        |a, b| if b.is_nan() { f64::NAN } else if a.is_nan() { f64::NAN } else if b > a { b } else { a },
        |v, _| v,
    )
}

#[wasm_bindgen]
pub fn reduce_mean_axis(data: &[f64], shape: &[u32], axis: u32) -> Vec<f64> {
    reduce_axis_impl(
        data,
        shape,
        axis as usize,
        0.0,
        |a, b| a + b,
        |v, n| v / n as f64,
    )
}

// ============ Argmin / Argmax Axis Reductions ============

/// Generic argmin/argmax along a single axis.
fn arg_reduce_axis_impl(
    data: &[f64],
    shape: &[u32],
    axis: usize,
    is_max: bool,
) -> Vec<f64> {
    let ndim = shape.len();
    assert!(axis < ndim, "axis {} out of bounds for ndim {}", axis, ndim);

    let axis_len = shape[axis] as usize;

    // Compute output shape (shape with axis removed)
    let mut out_shape: Vec<usize> = Vec::with_capacity(ndim - 1);
    for i in 0..ndim {
        if i != axis {
            out_shape.push(shape[i] as usize);
        }
    }

    let out_total: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };

    // Compute input strides
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as usize;
    }

    let axis_stride = strides[axis];

    // Compute output strides
    let mut out_strides = vec![1usize; out_shape.len()];
    if !out_shape.is_empty() {
        for i in (0..out_shape.len() - 1).rev() {
            out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
        }
    }

    let mut result = vec![0.0f64; out_total];

    for out_idx in 0..out_total {
        let mut tmp = out_idx;
        let mut base_input_idx = 0usize;
        let mut out_dim = 0;
        for d in 0..ndim {
            if d == axis {
                continue;
            }
            let coord = tmp / out_strides[out_dim];
            tmp %= out_strides[out_dim];
            base_input_idx += coord * strides[d];
            out_dim += 1;
        }

        let mut best_idx = 0usize;
        let mut best_val = data[base_input_idx];
        for k in 1..axis_len {
            let val = data[base_input_idx + k * axis_stride];
            let replace = if is_max { val > best_val } else { val < best_val };
            if replace {
                best_val = val;
                best_idx = k;
            }
        }
        result[out_idx] = best_idx as f64;
    }

    result
}

#[wasm_bindgen]
pub fn reduce_argmin_axis(data: &[f64], shape: &[u32], axis: u32) -> Vec<f64> {
    arg_reduce_axis_impl(data, shape, axis as usize, false)
}

#[wasm_bindgen]
pub fn reduce_argmax_axis(data: &[f64], shape: &[u32], axis: u32) -> Vec<f64> {
    arg_reduce_axis_impl(data, shape, axis as usize, true)
}

// ============ Matrix Multiplication ============

#[wasm_bindgen]
pub fn matmul(a: &[f64], m: u32, k: u32, b: &[f64], n: u32) -> Vec<f64> {
    let m = m as usize;
    let k = k as usize;
    let n = n as usize;
    let mut c = vec![0.0; m * n];

    // Cache-friendly loop order: i, p, j
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }

    c
}

// ============ Sorting ============

#[wasm_bindgen]
pub fn sort_f64(data: &[f64]) -> Vec<f64> {
    let mut sorted = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted
}

#[wasm_bindgen]
pub fn argsort_f64(data: &[f64]) -> Vec<u32> {
    let mut indices: Vec<u32> = (0..data.len() as u32).collect();
    indices.sort_unstable_by(|&a, &b| {
        data[a as usize]
            .partial_cmp(&data[b as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
}
