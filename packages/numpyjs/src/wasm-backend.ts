/**
 * WASM backend for numpyjs — compiled from Rust via wasm-pack.
 *
 * Overrides hot-path operations (unary math, binary math with broadcasting,
 * reductions, matmul, sort/argsort) with WASM implementations while
 * delegating everything else to BaseBackend.
 */

import {
  NDArray,
  DType,
  AnyTypedArray,
  ArrayOrScalar,
  SortKind,
  createTypedArrayFrom,
} from './types.js';
import { BaseBackend } from './base-backend.js';

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

  // Matmul
  matmul(a: Float64Array, m: number, k: number, b: Float64Array, n: number): Float64Array;

  // Sort
  sort_f64(data: Float64Array): Float64Array;
  argsort_f64(data: Float64Array): Uint32Array;
}

// ============ WasmNDArray ============

class WasmNDArray implements NDArray {
  data: AnyTypedArray;
  shape: number[];
  dtype: DType;

  constructor(data: AnyTypedArray | number[], shape: number[], dtype: DType = 'float64') {
    this.dtype = dtype;
    if (Array.isArray(data)) {
      this.data = createTypedArrayFrom(dtype, data);
    } else {
      this.data = data;
    }
    this.shape = shape;
  }

  toArray(): number[] {
    return Array.from(this.data);
  }

  get ndim(): number {
    return this.shape.length;
  }

  get size(): number {
    return this.shape.reduce((a, b) => a * b, 1);
  }

  get T(): NDArray {
    const ndim = this.shape.length;
    if (ndim <= 1) return this;
    const perm = [...Array(ndim).keys()].reverse();
    const newShape = perm.map(i => this.shape[i]!);
    const size = this.data.length;
    const data = new Float64Array(size);

    const oldStrides = new Array<number>(ndim);
    oldStrides[ndim - 1] = 1;
    for (let i = ndim - 2; i >= 0; i--) {
      oldStrides[i] = oldStrides[i + 1]! * this.shape[i + 1]!;
    }

    const newStrides = new Array<number>(ndim);
    newStrides[ndim - 1] = 1;
    for (let i = ndim - 2; i >= 0; i--) {
      newStrides[i] = newStrides[i + 1]! * newShape[i + 1]!;
    }

    for (let newFlat = 0; newFlat < size; newFlat++) {
      let remaining = newFlat;
      let oldFlat = 0;
      for (let d = 0; d < ndim; d++) {
        const coord = Math.floor(remaining / newStrides[d]!);
        remaining -= coord * newStrides[d]!;
        oldFlat += coord * oldStrides[perm[d]!]!;
      }
      data[newFlat] = this.data[oldFlat]!;
    }

    return new WasmNDArray(data, newShape, this.dtype);
  }

  item(): number {
    if (this.data.length !== 1) {
      throw new Error('can only convert an array of size 1 to a scalar');
    }
    return this.data[0]!;
  }
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
      return new WasmNDArray(data, shape, dtype);
    }
    if (ArrayBuffer.isView(data)) {
      return new WasmNDArray(data as AnyTypedArray, shape, dtype);
    }
    return new WasmNDArray(data, shape, dtype);
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

  override sum(arr: NDArray, axis?: number, keepdims?: boolean, dtype?: DType): number | NDArray {
    if (axis !== undefined) {
      axis = this._normalizeAxis(axis, arr.shape.length);
      const shapeU32 = toU32Shape(arr.shape);
      const resultData = this.wasm.reduce_sum_axis(toF64(arr), shapeU32, axis);
      const outShape = arr.shape.filter((_, i) => i !== axis);
      let result: NDArray = this.createArray(resultData, outShape.length > 0 ? outShape : [1]);
      if (dtype) result = this.astype(result, dtype);
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        result = this.reshape(result, newShape);
      }
      return result;
    }
    return this.wasm.reduce_sum(toF64(arr));
  }

  override prod(arr: NDArray, axis?: number, keepdims?: boolean, dtype?: DType): number | NDArray {
    if (axis !== undefined) {
      axis = this._normalizeAxis(axis, arr.shape.length);
      const shapeU32 = toU32Shape(arr.shape);
      const resultData = this.wasm.reduce_prod_axis(toF64(arr), shapeU32, axis);
      const outShape = arr.shape.filter((_, i) => i !== axis);
      let result: NDArray = this.createArray(resultData, outShape.length > 0 ? outShape : [1]);
      if (dtype) result = this.astype(result, dtype);
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        result = this.reshape(result, newShape);
      }
      return result;
    }
    return this.wasm.reduce_prod(toF64(arr));
  }

  override mean(arr: NDArray, axis?: number, keepdims?: boolean, dtype?: DType): number | NDArray {
    if (axis !== undefined) {
      axis = this._normalizeAxis(axis, arr.shape.length);
      const shapeU32 = toU32Shape(arr.shape);
      const resultData = this.wasm.reduce_mean_axis(toF64(arr), shapeU32, axis);
      const outShape = arr.shape.filter((_, i) => i !== axis);
      let result: NDArray = this.createArray(resultData, outShape.length > 0 ? outShape : [1]);
      if (dtype) result = this.astype(result, dtype);
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        result = this.reshape(result, newShape);
      }
      return result;
    }
    return this.wasm.reduce_mean(toF64(arr));
  }

  override min(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    if (axis !== undefined) {
      axis = this._normalizeAxis(axis, arr.shape.length);
      const shapeU32 = toU32Shape(arr.shape);
      const resultData = this.wasm.reduce_min_axis(toF64(arr), shapeU32, axis);
      const outShape = arr.shape.filter((_, i) => i !== axis);
      let result: NDArray = this.createArray(resultData, outShape.length > 0 ? outShape : [1]);
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        result = this.reshape(result, newShape);
      }
      return result;
    }
    return this.wasm.reduce_min(toF64(arr));
  }

  override max(arr: NDArray, axis?: number, keepdims?: boolean): number | NDArray {
    if (axis !== undefined) {
      axis = this._normalizeAxis(axis, arr.shape.length);
      const shapeU32 = toU32Shape(arr.shape);
      const resultData = this.wasm.reduce_max_axis(toF64(arr), shapeU32, axis);
      const outShape = arr.shape.filter((_, i) => i !== axis);
      let result: NDArray = this.createArray(resultData, outShape.length > 0 ? outShape : [1]);
      if (keepdims) {
        const newShape = [...arr.shape];
        newShape[axis] = 1;
        result = this.reshape(result, newShape);
      }
      return result;
    }
    return this.wasm.reduce_max(toF64(arr));
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
