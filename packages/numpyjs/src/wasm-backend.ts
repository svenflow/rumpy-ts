/**
 * WASM backend for numpyjs — compiled from Rust via wasm-pack.
 *
 * Overrides hot-path operations (unary math, binary math with broadcasting,
 * reductions, matmul, sort/argsort) with WASM implementations while
 * delegating everything else to BaseBackend.
 */

import { NDArray, DType, AnyTypedArray, ArrayOrScalar, SortKind } from './types.js';
import { BaseBackend, BaseNDArray } from './base-backend.js';

// Type for the wasm-pack generated module
interface WasmModule {
  // Unary ops
  unary_sin(data: Float64Array): Float64Array;
  unary_cos(data: Float64Array): Float64Array;
  unary_tan(data: Float64Array): Float64Array;
  unary_asin(data: Float64Array): Float64Array;
  unary_acos(data: Float64Array): Float64Array;
  unary_atan(data: Float64Array): Float64Array;
  unary_sinh(data: Float64Array): Float64Array;
  unary_cosh(data: Float64Array): Float64Array;
  unary_tanh(data: Float64Array): Float64Array;
  unary_exp(data: Float64Array): Float64Array;
  unary_log(data: Float64Array): Float64Array;
  unary_log2(data: Float64Array): Float64Array;
  unary_log10(data: Float64Array): Float64Array;
  unary_sqrt(data: Float64Array): Float64Array;
  unary_cbrt(data: Float64Array): Float64Array;
  unary_abs(data: Float64Array): Float64Array;
  unary_ceil(data: Float64Array): Float64Array;
  unary_floor(data: Float64Array): Float64Array;
  unary_round(data: Float64Array): Float64Array;
  unary_sign(data: Float64Array): Float64Array;
  unary_negative(data: Float64Array): Float64Array;
  unary_reciprocal(data: Float64Array): Float64Array;
  unary_square(data: Float64Array): Float64Array;
  unary_expm1(data: Float64Array): Float64Array;
  unary_log1p(data: Float64Array): Float64Array;
  unary_trunc(data: Float64Array): Float64Array;
  unary_asinh(data: Float64Array): Float64Array;
  unary_acosh(data: Float64Array): Float64Array;
  unary_atanh(data: Float64Array): Float64Array;

  // Binary ops (with broadcasting)
  binary_add(
    a: Float64Array,
    aShape: Uint32Array,
    b: Float64Array,
    bShape: Uint32Array
  ): Float64Array;
  binary_subtract(
    a: Float64Array,
    aShape: Uint32Array,
    b: Float64Array,
    bShape: Uint32Array
  ): Float64Array;
  binary_multiply(
    a: Float64Array,
    aShape: Uint32Array,
    b: Float64Array,
    bShape: Uint32Array
  ): Float64Array;
  binary_divide(
    a: Float64Array,
    aShape: Uint32Array,
    b: Float64Array,
    bShape: Uint32Array
  ): Float64Array;
  binary_power(
    a: Float64Array,
    aShape: Uint32Array,
    b: Float64Array,
    bShape: Uint32Array
  ): Float64Array;
  binary_maximum(
    a: Float64Array,
    aShape: Uint32Array,
    b: Float64Array,
    bShape: Uint32Array
  ): Float64Array;
  binary_minimum(
    a: Float64Array,
    aShape: Uint32Array,
    b: Float64Array,
    bShape: Uint32Array
  ): Float64Array;
  binary_mod(
    a: Float64Array,
    aShape: Uint32Array,
    b: Float64Array,
    bShape: Uint32Array
  ): Float64Array;
  broadcast_shape(aShape: Uint32Array, bShape: Uint32Array): Uint32Array;

  // Reductions (full)
  reduce_sum(data: Float64Array): number;
  reduce_prod(data: Float64Array): number;
  reduce_min(data: Float64Array): number;
  reduce_max(data: Float64Array): number;
  reduce_mean(data: Float64Array): number;

  // Axis reductions
  reduce_sum_axis(data: Float64Array, shape: Uint32Array, axis: number): Float64Array;
  reduce_prod_axis(data: Float64Array, shape: Uint32Array, axis: number): Float64Array;
  reduce_min_axis(data: Float64Array, shape: Uint32Array, axis: number): Float64Array;
  reduce_max_axis(data: Float64Array, shape: Uint32Array, axis: number): Float64Array;
  reduce_mean_axis(data: Float64Array, shape: Uint32Array, axis: number): Float64Array;

  // Argmin/Argmax axis reductions
  reduce_argmin_axis(data: Float64Array, shape: Uint32Array, axis: number): Float64Array;
  reduce_argmax_axis(data: Float64Array, shape: Uint32Array, axis: number): Float64Array;

  // Matmul
  matmul(a: Float64Array, m: number, k: number, b: Float64Array, n: number): Float64Array;

  // Sort
  sort_f64(data: Float64Array): Float64Array;
  argsort_f64(data: Float64Array): Uint32Array;
}

// ============ Module-level WASM state ============

let wasmModule: WasmModule | null = null;

export async function initWasmBackend(): Promise<void> {
  // Dynamic import of wasm-pack generated module
  const wasm = await import('../wasm/pkg/numpyjs_wasm.js');
  await (wasm.default as any)();
  wasmModule = wasm as unknown as WasmModule;
}

export function createWasmBackend(): WasmBackend {
  if (!wasmModule) throw new Error('Call initWasmBackend() first');
  return new WasmBackend(wasmModule);
}

// ============ Helper: convert data to Float64Array ============

function toF64(arr: NDArray): Float64Array {
  if (arr.data instanceof Float64Array) return arr.data;
  return new Float64Array(arr.data);
}

function toU32Shape(shape: number[]): Uint32Array {
  return new Uint32Array(shape);
}

// ============ WasmBackend ============

export class WasmBackend extends BaseBackend {
  override name = 'wasm';
  private wasm: WasmModule;

  constructor(wasm: WasmModule) {
    super();
    this.wasm = wasm;
  }

  override createArray(
    data: number[] | Float64Array | AnyTypedArray,
    shape: number[],
    dtype: DType = 'float64'
  ): NDArray {
    if (data instanceof Float64Array) {
      return new BaseNDArray(data, shape, dtype);
    }
    if (ArrayBuffer.isView(data)) {
      return new BaseNDArray(data as AnyTypedArray, shape, dtype);
    }
    return new BaseNDArray(data, shape, dtype);
  }

  // ============ Unary ops (WASM) ============

  private _wasmUnary(arr: NDArray, fn: (d: Float64Array) => Float64Array): NDArray {
    const result = fn(toF64(arr));
    return this.createArray(result, [...arr.shape], arr.dtype);
  }

  override sin(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_sin(d));
  }

  override cos(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_cos(d));
  }

  override tan(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_tan(d));
  }

  override arcsin(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_asin(d));
  }

  override arccos(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_acos(d));
  }

  override arctan(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_atan(d));
  }

  override sinh(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_sinh(d));
  }

  override cosh(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_cosh(d));
  }

  override tanh(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_tanh(d));
  }

  override exp(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_exp(d));
  }

  override log(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_log(d));
  }

  override log2(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_log2(d));
  }

  override log10(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_log10(d));
  }

  override sqrt(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_sqrt(d));
  }

  override cbrt(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_cbrt(d));
  }

  override abs(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_abs(d));
  }

  override absolute(arr: NDArray): NDArray {
    return this.abs(arr);
  }

  override ceil(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_ceil(d));
  }

  override floor(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_floor(d));
  }

  override round(arr: NDArray, decimals: number = 0): NDArray {
    if (decimals === 0) {
      return this._wasmUnary(arr, d => this.wasm.unary_round(d));
    }
    // For non-zero decimals, fall back to BaseBackend
    return super.round(arr, decimals);
  }

  override sign(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_sign(d));
  }

  override negative(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_negative(d));
  }

  override reciprocal(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_reciprocal(d));
  }

  override square(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_square(d));
  }

  override expm1(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_expm1(d));
  }

  override log1p(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_log1p(d));
  }

  override trunc(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_trunc(d));
  }

  override fix(arr: NDArray): NDArray {
    return this.trunc(arr);
  }

  override arcsinh(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_asinh(d));
  }

  override arccosh(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_acosh(d));
  }

  override arctanh(arr: NDArray): NDArray {
    return this._wasmUnary(arr, d => this.wasm.unary_atanh(d));
  }

  // ============ Binary ops (WASM with broadcasting) ============

  private _wasmBinaryOp(
    a: ArrayOrScalar,
    b: ArrayOrScalar,
    fn: (
      aData: Float64Array,
      aShape: Uint32Array,
      bData: Float64Array,
      bShape: Uint32Array
    ) => Float64Array
  ): NDArray {
    const arrA = this._toNDArray(a);
    const arrB = this._toNDArray(b);
    const aShape = toU32Shape(arrA.shape);
    const bShape = toU32Shape(arrB.shape);
    const result = fn(toF64(arrA), aShape, toF64(arrB), bShape);
    const outShape = Array.from(this.wasm.broadcast_shape(aShape, bShape));
    return this.createArray(result, outShape);
  }

  override add(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._wasmBinaryOp(a, b, (ad, as_, bd, bs) => this.wasm.binary_add(ad, as_, bd, bs));
  }

  override subtract(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._wasmBinaryOp(a, b, (ad, as_, bd, bs) =>
      this.wasm.binary_subtract(ad, as_, bd, bs)
    );
  }

  override multiply(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._wasmBinaryOp(a, b, (ad, as_, bd, bs) =>
      this.wasm.binary_multiply(ad, as_, bd, bs)
    );
  }

  override divide(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._wasmBinaryOp(a, b, (ad, as_, bd, bs) => this.wasm.binary_divide(ad, as_, bd, bs));
  }

  override power(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._wasmBinaryOp(a, b, (ad, as_, bd, bs) => this.wasm.binary_power(ad, as_, bd, bs));
  }

  override maximum(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._wasmBinaryOp(a, b, (ad, as_, bd, bs) => this.wasm.binary_maximum(ad, as_, bd, bs));
  }

  override minimum(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._wasmBinaryOp(a, b, (ad, as_, bd, bs) => this.wasm.binary_minimum(ad, as_, bd, bs));
  }

  override mod(a: ArrayOrScalar, b: ArrayOrScalar): NDArray {
    return this._wasmBinaryOp(a, b, (ad, as_, bd, bs) => this.wasm.binary_mod(ad, as_, bd, bs));
  }

  // ============ Reductions (WASM) ============

  private _wasmReduction(
    arr: NDArray,
    axis: number | undefined,
    keepdims: boolean | undefined,
    fullReducer: (data: Float64Array) => number,
    axisReducer: (data: Float64Array, shape: Uint32Array, axis: number) => Float64Array,
    dtype?: DType
  ): number | NDArray {
    if (axis !== undefined) {
      const normAxis = this._normalizeAxis(axis, arr.shape.length);
      const resultData = axisReducer(toF64(arr), toU32Shape(arr.shape), normAxis);
      const outShape = arr.shape.filter((_, i) => i !== normAxis);
      let result: NDArray = this.createArray(resultData, outShape.length > 0 ? outShape : [1]);
      if (dtype) result = this.astype(result, dtype);
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[normAxis] = 1;
        result = this.reshape(result, newShape);
      }
      return result;
    }
    return fullReducer(toF64(arr));
  }

  override sum(arr: NDArray, axis?: number, keepdims?: boolean, dtype?: DType): number | NDArray {
    return this._wasmReduction(
      arr,
      axis,
      keepdims,
      d => this.wasm.reduce_sum(d),
      (d, s, a) => this.wasm.reduce_sum_axis(d, s, a),
      dtype
    );
  }

  override prod(arr: NDArray, axis?: number, keepdims?: boolean, dtype?: DType): number | NDArray {
    return this._wasmReduction(
      arr,
      axis,
      keepdims,
      d => this.wasm.reduce_prod(d),
      (d, s, a) => this.wasm.reduce_prod_axis(d, s, a),
      dtype
    );
  }

  override mean(arr: NDArray, axis?: number, keepdims?: boolean, dtype?: DType): number | NDArray {
    return this._wasmReduction(
      arr,
      axis,
      keepdims,
      d => this.wasm.reduce_mean(d),
      (d, s, a) => this.wasm.reduce_mean_axis(d, s, a),
      dtype
    );
  }

  override min(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    return this._wasmReduction(
      arr,
      axis,
      keepdims,
      d => this.wasm.reduce_min(d),
      (d, s, a) => this.wasm.reduce_min_axis(d, s, a)
    );
  }

  override max(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    return this._wasmReduction(
      arr,
      axis,
      keepdims,
      d => this.wasm.reduce_max(d),
      (d, s, a) => this.wasm.reduce_max_axis(d, s, a)
    );
  }

  override argmin(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    if (axis !== undefined) {
      const normAxis = this._normalizeAxis(axis, arr.shape.length);
      const resultData = this.wasm.reduce_argmin_axis(toF64(arr), toU32Shape(arr.shape), normAxis);
      const outShape = arr.shape.filter((_, i) => i !== normAxis);
      let result: NDArray = this.createArray(resultData, outShape.length > 0 ? outShape : [1]);
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[normAxis] = 1;
        result = this.reshape(result, newShape);
      }
      return result;
    }
    if (arr.data.length === 0) throw new Error('zero-size array');
    let minIdx = 0;
    for (let i = 1; i < arr.data.length; i++) {
      if (arr.data[i] < arr.data[minIdx]) minIdx = i;
    }
    return minIdx;
  }

  override argmax(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    if (axis !== undefined) {
      const normAxis = this._normalizeAxis(axis, arr.shape.length);
      const resultData = this.wasm.reduce_argmax_axis(toF64(arr), toU32Shape(arr.shape), normAxis);
      const outShape = arr.shape.filter((_, i) => i !== normAxis);
      let result: NDArray = this.createArray(resultData, outShape.length > 0 ? outShape : [1]);
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[normAxis] = 1;
        result = this.reshape(result, newShape);
      }
      return result;
    }
    if (arr.data.length === 0) throw new Error('zero-size array');
    let maxIdx = 0;
    for (let i = 1; i < arr.data.length; i++) {
      if (arr.data[i] > arr.data[maxIdx]) maxIdx = i;
    }
    return maxIdx;
  }

  // ============ Matmul (WASM) ============

  override matmul(a: NDArray, b: NDArray): NDArray {
    // Only accelerate 2D matrix multiply; delegate batched/1D to BaseBackend
    if (a.shape.length === 2 && b.shape.length === 2) {
      const m = a.shape[0]!;
      const k = a.shape[1]!;
      const n = b.shape[1]!;
      if (k !== b.shape[0]!) {
        throw new Error(
          `matmul: shapes (${a.shape}) and (${b.shape}) not aligned: ${k} (dim 1) != ${b.shape[0]} (dim 0)`
        );
      }
      const result = this.wasm.matmul(toF64(a), m, k, toF64(b), n);
      return this.createArray(result, [m, n]);
    }
    return super.matmul(a, b);
  }

  // ============ Sort / Argsort (WASM for flat/last-axis) ============

  override sort(arr: NDArray, axis: number = -1, kind?: SortKind): NDArray {
    const ndim = arr.shape.length;
    const normAxis = this._normalizeAxis(axis, ndim);

    // Fast path: 1D sort
    if (ndim === 1) {
      const result = this.wasm.sort_f64(toF64(arr));
      return this.createArray(result, [...arr.shape], arr.dtype);
    }

    // Fast path: sort along last axis for 2D
    if (ndim === 2 && normAxis === ndim - 1) {
      const [rows, cols] = arr.shape as [number, number];
      const data = toF64(arr);
      const result = new Float64Array(data.length);
      for (let r = 0; r < rows; r++) {
        const slice = data.slice(r * cols, (r + 1) * cols);
        const sorted = this.wasm.sort_f64(slice);
        result.set(sorted, r * cols);
      }
      return this.createArray(result, [...arr.shape], arr.dtype);
    }

    // Fall back to BaseBackend for other cases
    return super.sort(arr, axis, kind);
  }

  override argsort(arr: NDArray, axis: number = -1, kind?: SortKind): NDArray {
    const ndim = arr.shape.length;
    const normAxis = this._normalizeAxis(axis, ndim);

    // Fast path: 1D argsort
    if (ndim === 1) {
      const result = this.wasm.argsort_f64(toF64(arr));
      return this.createArray(new Float64Array(result), [...arr.shape], 'float64');
    }

    // Fast path: argsort along last axis for 2D
    if (ndim === 2 && normAxis === ndim - 1) {
      const [rows, cols] = arr.shape as [number, number];
      const data = toF64(arr);
      const result = new Float64Array(data.length);
      for (let r = 0; r < rows; r++) {
        const slice = data.slice(r * cols, (r + 1) * cols);
        const indices = this.wasm.argsort_f64(slice);
        for (let c = 0; c < cols; c++) {
          result[r * cols + c] = indices[c]!;
        }
      }
      return this.createArray(result, [...arr.shape], 'float64');
    }

    return super.argsort(arr, axis, kind);
  }
}
