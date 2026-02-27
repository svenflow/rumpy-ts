#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""
Generate test cases from NumPy to verify rumpy-ts correctness.
Outputs JSON that can be loaded by the JS test runner.
"""

import numpy as np
import json

np.random.seed(42)

test_cases = []

def add_test(name, op_name, inputs, expected, shape=None):
    """Add a test case."""
    test_cases.append({
        "name": name,
        "op": op_name,
        "inputs": inputs,
        "expected": expected.flatten().tolist() if hasattr(expected, 'flatten') else expected,
        "expected_shape": list(expected.shape) if hasattr(expected, 'shape') else shape
    })

# ==================== SPRINT 1: Structural Operations ====================

# Test permute/transpose
arr_2x3 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
add_test("permute 2x3 -> 3x2", "permute",
         {"data": arr_2x3.flatten().tolist(), "shape": [2, 3], "axes": [1, 0]},
         arr_2x3.T)

arr_2x3x4 = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
add_test("permute 2x3x4 -> 4x2x3", "permute",
         {"data": arr_2x3x4.flatten().tolist(), "shape": [2, 3, 4], "axes": [2, 0, 1]},
         arr_2x3x4.transpose(2, 0, 1))

# Test sumAxis
arr_3x4 = np.arange(12, dtype=np.float64).reshape(3, 4)
add_test("sumAxis(0) 3x4", "sumAxis",
         {"data": arr_3x4.flatten().tolist(), "shape": [3, 4], "axis": 0, "keepdims": False},
         arr_3x4.sum(axis=0))

add_test("sumAxis(1) 3x4", "sumAxis",
         {"data": arr_3x4.flatten().tolist(), "shape": [3, 4], "axis": 1, "keepdims": False},
         arr_3x4.sum(axis=1))

add_test("sumAxis(0, keepdims) 3x4", "sumAxis",
         {"data": arr_3x4.flatten().tolist(), "shape": [3, 4], "axis": 0, "keepdims": True},
         arr_3x4.sum(axis=0, keepdims=True))

# Test meanAxis
add_test("meanAxis(0) 3x4", "meanAxis",
         {"data": arr_3x4.flatten().tolist(), "shape": [3, 4], "axis": 0, "keepdims": False},
         arr_3x4.mean(axis=0))

add_test("meanAxis(1) 3x4", "meanAxis",
         {"data": arr_3x4.flatten().tolist(), "shape": [3, 4], "axis": 1, "keepdims": False},
         arr_3x4.mean(axis=1))

# Test maxAxis / minAxis
arr_rand = np.random.rand(4, 5).astype(np.float64)
add_test("maxAxis(0) 4x5", "maxAxis",
         {"data": arr_rand.flatten().tolist(), "shape": [4, 5], "axis": 0, "keepdims": False},
         arr_rand.max(axis=0))

add_test("maxAxis(1) 4x5", "maxAxis",
         {"data": arr_rand.flatten().tolist(), "shape": [4, 5], "axis": 1, "keepdims": False},
         arr_rand.max(axis=1))

add_test("minAxis(0) 4x5", "minAxis",
         {"data": arr_rand.flatten().tolist(), "shape": [4, 5], "axis": 0, "keepdims": False},
         arr_rand.min(axis=0))

add_test("minAxis(1) 4x5", "minAxis",
         {"data": arr_rand.flatten().tolist(), "shape": [4, 5], "axis": 1, "keepdims": False},
         arr_rand.min(axis=1))

# Test softmax
def softmax(x, axis):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

arr_softmax = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
add_test("softmax axis=1", "softmax",
         {"data": arr_softmax.flatten().tolist(), "shape": [2, 3], "axis": 1},
         softmax(arr_softmax, axis=1))

add_test("softmax axis=0", "softmax",
         {"data": arr_softmax.flatten().tolist(), "shape": [2, 3], "axis": 0},
         softmax(arr_softmax, axis=0))

# Test relu
arr_relu = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
add_test("relu", "relu",
         {"data": arr_relu.tolist(), "shape": [5]},
         np.maximum(arr_relu, 0))

# Test gelu (approximate)
def gelu_approx(x):
    return x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

arr_gelu = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
add_test("gelu", "gelu",
         {"data": arr_gelu.tolist(), "shape": [5]},
         gelu_approx(arr_gelu))

# Test argmax/argmin
arr_arg = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.float64)
add_test("argmax", "argmax",
         {"data": arr_arg.tolist(), "shape": [8]},
         int(np.argmax(arr_arg)), shape=[])

add_test("argmin", "argmin",
         {"data": arr_arg.tolist(), "shape": [8]},
         int(np.argmin(arr_arg)), shape=[])

# ==================== SPRINT 2: CNN Operations ====================

# Test slice
arr_10 = np.arange(10, dtype=np.float64)
add_test("slice [2:5]", "slice",
         {"data": arr_10.tolist(), "shape": [10], "starts": [2], "stops": [5], "steps": [1]},
         arr_10[2:5])

add_test("slice [-3:]", "slice",
         {"data": arr_10.tolist(), "shape": [10], "starts": [-3], "stops": [2147483647], "steps": [1]},
         arr_10[-3:])

add_test("slice [::2]", "slice",
         {"data": arr_10.tolist(), "shape": [10], "starts": [0], "stops": [2147483647], "steps": [2]},
         arr_10[::2])

# 2D slice
arr_5x5 = np.arange(25, dtype=np.float64).reshape(5, 5)
add_test("slice 2D [1:4, 2:5]", "slice",
         {"data": arr_5x5.flatten().tolist(), "shape": [5, 5],
          "starts": [1, 2], "stops": [4, 5], "steps": [1, 1]},
         arr_5x5[1:4, 2:5])

# Test im2col
# Simple 1x1x4x4 image
img_4x4 = np.arange(16, dtype=np.float64).reshape(1, 1, 4, 4)

def im2col_numpy(img, kh, kw, sh, sw, ph, pw):
    """NumPy reference implementation of im2col."""
    n, c, h, w = img.shape
    h_out = (h + 2*ph - kh) // sh + 1
    w_out = (w + 2*pw - kw) // sw + 1

    # Pad if needed
    if ph > 0 or pw > 0:
        img = np.pad(img, ((0,0), (0,0), (ph,ph), (pw,pw)), mode='constant')

    cols = np.zeros((n * h_out * w_out, c * kh * kw))

    for batch in range(n):
        for oh in range(h_out):
            for ow in range(w_out):
                row_idx = batch * h_out * w_out + oh * w_out + ow
                patch = img[batch, :, oh*sh:oh*sh+kh, ow*sw:ow*sw+kw]
                cols[row_idx] = patch.flatten()

    return cols

add_test("im2col 4x4 k=2 s=1 p=0", "im2col",
         {"data": img_4x4.flatten().tolist(), "shape": [1, 1, 4, 4],
          "kernel_h": 2, "kernel_w": 2, "stride_h": 1, "stride_w": 1, "pad_h": 0, "pad_w": 0},
         im2col_numpy(img_4x4, 2, 2, 1, 1, 0, 0))

add_test("im2col 4x4 k=2 s=2 p=0", "im2col",
         {"data": img_4x4.flatten().tolist(), "shape": [1, 1, 4, 4],
          "kernel_h": 2, "kernel_w": 2, "stride_h": 2, "stride_w": 2, "pad_h": 0, "pad_w": 0},
         im2col_numpy(img_4x4, 2, 2, 2, 2, 0, 0))

# Test maxPool2d
def maxpool2d_numpy(img, kh, kw, sh, sw, ph, pw):
    """NumPy reference implementation of max pooling."""
    n, c, h, w = img.shape
    h_out = (h + 2*ph - kh) // sh + 1
    w_out = (w + 2*pw - kw) // sw + 1

    # Pad if needed
    if ph > 0 or pw > 0:
        img = np.pad(img, ((0,0), (0,0), (ph,ph), (pw,pw)), mode='constant', constant_values=-np.inf)

    out = np.zeros((n, c, h_out, w_out))

    for batch in range(n):
        for ch in range(c):
            for oh in range(h_out):
                for ow in range(w_out):
                    patch = img[batch, ch, oh*sh:oh*sh+kh, ow*sw:ow*sw+kw]
                    out[batch, ch, oh, ow] = np.max(patch)

    return out

add_test("maxPool2d 4x4 k=2 s=2", "maxPool2d",
         {"data": img_4x4.flatten().tolist(), "shape": [1, 1, 4, 4],
          "kernel_h": 2, "kernel_w": 2, "stride_h": 2, "stride_w": 2, "pad_h": 0, "pad_w": 0},
         maxpool2d_numpy(img_4x4, 2, 2, 2, 2, 0, 0))

# Test avgPool2d
def avgpool2d_numpy(img, kh, kw, sh, sw, ph, pw):
    """NumPy reference implementation of average pooling."""
    n, c, h, w = img.shape
    h_out = (h + 2*ph - kh) // sh + 1
    w_out = (w + 2*pw - kw) // sw + 1

    out = np.zeros((n, c, h_out, w_out))

    for batch in range(n):
        for ch in range(c):
            for oh in range(h_out):
                for ow in range(w_out):
                    count = 0
                    total = 0.0
                    for ki in range(kh):
                        for kj in range(kw):
                            ih = oh * sh + ki - ph
                            iw = ow * sw + kj - pw
                            if 0 <= ih < h and 0 <= iw < w:
                                total += img[batch, ch, ih, iw]
                                count += 1
                    out[batch, ch, oh, ow] = total / count if count > 0 else 0

    return out

add_test("avgPool2d 4x4 k=2 s=2", "avgPool2d",
         {"data": img_4x4.flatten().tolist(), "shape": [1, 1, 4, 4],
          "kernel_h": 2, "kernel_w": 2, "stride_h": 2, "stride_w": 2, "pad_h": 0, "pad_w": 0},
         avgpool2d_numpy(img_4x4, 2, 2, 2, 2, 0, 0))

# ==================== SPRINT 3: Boolean Masking ====================

# Test comparisons
arr_cmp = np.array([1, 2, 3, 4, 5], dtype=np.float64)
add_test("gtScalar(3)", "gtScalar",
         {"data": arr_cmp.tolist(), "shape": [5], "scalar": 3},
         (arr_cmp > 3).astype(np.float64))

add_test("ltScalar(3)", "ltScalar",
         {"data": arr_cmp.tolist(), "shape": [5], "scalar": 3},
         (arr_cmp < 3).astype(np.float64))

add_test("eqScalar(3)", "eqScalar",
         {"data": arr_cmp.tolist(), "shape": [5], "scalar": 3},
         (arr_cmp == 3).astype(np.float64))

# Test getByMask
arr_mask_test = np.array([1, 2, 3, 4, 5], dtype=np.float64)
mask = arr_mask_test > 2
add_test("getByMask (x > 2)", "getByMask",
         {"data": arr_mask_test.tolist(), "shape": [5],
          "mask": mask.astype(np.float64).tolist()},
         arr_mask_test[mask])

# Test setByMask
result_set = arr_mask_test.copy()
result_set[mask] = 0
add_test("setByMask (x > 2, 0)", "setByMask",
         {"data": arr_mask_test.tolist(), "shape": [5],
          "mask": mask.astype(np.float64).tolist(), "value": 0},
         result_set)

# Test where
cond = np.array([1, 0, 1, 0, 1], dtype=np.float64)
x = np.array([10, 20, 30, 40, 50], dtype=np.float64)
y = np.array([1, 2, 3, 4, 5], dtype=np.float64)
add_test("where", "where_",
         {"cond": cond.tolist(), "x": x.tolist(), "y": y.tolist(), "shape": [5]},
         np.where(cond.astype(bool), x, y))

# Test clip
arr_clip = np.array([-5, 0, 5, 10, 15], dtype=np.float64)
add_test("clip(0, 10)", "clip",
         {"data": arr_clip.tolist(), "shape": [5], "min": 0, "max": 10},
         np.clip(arr_clip, 0, 10))

# Test isnan/isinf
arr_special = np.array([1, np.nan, np.inf, -np.inf, 0], dtype=np.float64)
add_test("isNan", "isNan",
         {"data": arr_special.tolist(), "shape": [5]},
         np.isnan(arr_special).astype(np.float64))

add_test("isInf", "isInf",
         {"data": arr_special.tolist(), "shape": [5]},
         np.isinf(arr_special).astype(np.float64))

add_test("isFinite", "isFinite",
         {"data": arr_special.tolist(), "shape": [5]},
         np.isfinite(arr_special).astype(np.float64))

# Test countNonzero
arr_nz = np.array([0, 1, 0, 2, 0, 3], dtype=np.float64)
add_test("countNonzero", "countNonzero",
         {"data": arr_nz.tolist(), "shape": [6]},
         int(np.count_nonzero(arr_nz)), shape=[])

# Custom JSON encoder to handle NaN/Inf
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            if np.isnan(obj):
                return "NaN"
            elif np.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
            return float(obj)
        return super().default(obj)

def sanitize_for_json(obj):
    """Replace NaN/Inf with string markers in nested structures."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if np.isnan(obj):
            return "NaN"
        elif np.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
    return obj

# Write output
output = {
    "generated_by": "numpy",
    "numpy_version": np.__version__,
    "test_count": len(test_cases),
    "tests": sanitize_for_json(test_cases)
}

with open("numpy-test-cases.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Generated {len(test_cases)} test cases")
print(f"Saved to numpy-test-cases.json")
