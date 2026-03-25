/**
 * Pure JavaScript reference backend for testing
 *
 * This implements all Backend operations in pure JS, serving as:
 * 1. A reference implementation for testing
 * 2. A fallback when WebGPU is not available
 * 3. A baseline for performance comparisons
 *
 * Extends BaseBackend which provides all the operation implementations.
 * This file only needs to provide the JsNDArray class and createArray().
 */

import { Backend, NDArray, DType, AnyTypedArray, createTypedArrayFrom } from './types.js';
import { BaseBackend } from './base-backend.js';

class JsNDArray implements NDArray {
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
    // Reverse axes for transpose
    const perm = [...Array(ndim).keys()].reverse();
    const newShape = perm.map(i => this.shape[i]);
    const size = this.data.length;
    const data = new Float64Array(size);

    const oldStrides = new Array(ndim);
    oldStrides[ndim - 1] = 1;
    for (let i = ndim - 2; i >= 0; i--) {
      oldStrides[i] = oldStrides[i + 1] * this.shape[i + 1];
    }

    const newStrides = new Array(ndim);
    newStrides[ndim - 1] = 1;
    for (let i = ndim - 2; i >= 0; i--) {
      newStrides[i] = newStrides[i + 1] * newShape[i + 1];
    }

    for (let newFlat = 0; newFlat < size; newFlat++) {
      let remaining = newFlat;
      let oldFlat = 0;
      for (let d = 0; d < ndim; d++) {
        const coord = Math.floor(remaining / newStrides[d]);
        remaining -= coord * newStrides[d];
        oldFlat += coord * oldStrides[perm[d]];
      }
      data[newFlat] = this.data[oldFlat];
    }

    return new JsNDArray(data, newShape, this.dtype);
  }

  item(): number {
    if (this.data.length !== 1) {
      throw new Error('can only convert an array of size 1 to a scalar');
    }
    return this.data[0];
  }
}

export class JsBackend extends BaseBackend {
  override name = 'js';

  override createArray(
    data: number[] | Float64Array | AnyTypedArray,
    shape: number[],
    dtype: DType = 'float64'
  ): NDArray {
    if (data instanceof Float64Array) {
      return new JsNDArray(data, shape, dtype);
    }
    if (ArrayBuffer.isView(data)) {
      // It's some other typed array (Float32Array, Int32Array, etc.)
      return new JsNDArray(data as AnyTypedArray, shape, dtype);
    }
    // It's a number[]
    return new JsNDArray(data, shape, dtype);
  }
}

export function createJsBackend(): Backend {
  return new JsBackend();
}
