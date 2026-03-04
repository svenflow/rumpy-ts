/**
 * WASM Backend adapter for tests
 *
 * This wraps the rumpy-wasm module to implement the Backend interface.
 */

import { Backend, NDArray as IFaceNDArray } from './test-utils';

// These will be populated by initWasmBackend
let wasmModule: any = null;

class WasmNDArray implements IFaceNDArray {
  private _inner: any;

  constructor(inner: any) {
    this._inner = inner;
  }

  get shape(): number[] {
    return Array.from(this._inner.shape);
  }

  get data(): Float64Array {
    return this._inner.toTypedArray();
  }

  toArray(): number[] {
    return Array.from(this._inner.toTypedArray());
  }

  get inner(): any {
    return this._inner;
  }
}

export class WasmBackend implements Backend {
  name = 'wasm';
  private wasm: any;

  constructor(wasm: any) {
    this.wasm = wasm;
  }

  private wrap(inner: any): WasmNDArray {
    return new WasmNDArray(inner);
  }

  private unwrap(arr: IFaceNDArray): any {
    return (arr as WasmNDArray).inner;
  }

  // ============ Creation ============

  zeros(shape: number[]): IFaceNDArray {
    return this.wrap(this.wasm.zeros(new Uint32Array(shape)));
  }

  ones(shape: number[]): IFaceNDArray {
    return this.wrap(this.wasm.ones(new Uint32Array(shape)));
  }

  full(shape: number[], value: number): IFaceNDArray {
    return this.wrap(this.wasm.full(new Uint32Array(shape), value));
  }

  arange(start: number, stop: number, step: number): IFaceNDArray {
    if (step === 0) {
      throw new Error('step cannot be zero');
    }
    return this.wrap(this.wasm.arange(start, stop, step));
  }

  linspace(start: number, stop: number, num: number): IFaceNDArray {
    return this.wrap(this.wasm.linspace(start, stop, num));
  }

  eye(n: number): IFaceNDArray {
    return this.wrap(this.wasm.eye(n));
  }

  diag(arr: IFaceNDArray, k: number = 0): IFaceNDArray {
    return this.wrap(this.unwrap(arr).diag(k));
  }

  array(data: number[], shape?: number[]): IFaceNDArray {
    const s = shape || [data.length];
    return this.wrap(this.wasm.arrayFromTyped(new Float64Array(data), new Uint32Array(s)));
  }

  // ============ Math - Unary ============

  sin(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.sinArr(this.unwrap(arr)));
  }

  cos(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.cosArr(this.unwrap(arr)));
  }

  tan(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.tanArr(this.unwrap(arr)));
  }

  arcsin(arr: IFaceNDArray): IFaceNDArray {
    // Fallback - not directly available in WASM
    const result = Array.from(arr.data).map(Math.asin);
    return this.array(result, arr.shape);
  }

  arccos(arr: IFaceNDArray): IFaceNDArray {
    const result = Array.from(arr.data).map(Math.acos);
    return this.array(result, arr.shape);
  }

  arctan(arr: IFaceNDArray): IFaceNDArray {
    const result = Array.from(arr.data).map(Math.atan);
    return this.array(result, arr.shape);
  }

  sinh(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.sinhArr(this.unwrap(arr)));
  }

  cosh(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.coshArr(this.unwrap(arr)));
  }

  tanh(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.tanhArr(this.unwrap(arr)));
  }

  exp(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.expArr(this.unwrap(arr)));
  }

  log(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.logArr(this.unwrap(arr)));
  }

  log2(arr: IFaceNDArray): IFaceNDArray {
    const result = Array.from(arr.data).map(Math.log2);
    return this.array(result, arr.shape);
  }

  log10(arr: IFaceNDArray): IFaceNDArray {
    const result = Array.from(arr.data).map(Math.log10);
    return this.array(result, arr.shape);
  }

  sqrt(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.sqrtArr(this.unwrap(arr)));
  }

  cbrt(arr: IFaceNDArray): IFaceNDArray {
    const result = Array.from(arr.data).map(Math.cbrt);
    return this.array(result, arr.shape);
  }

  abs(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.absArr(this.unwrap(arr)));
  }

  sign(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.signArr(this.unwrap(arr)));
  }

  floor(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.floorArr(this.unwrap(arr)));
  }

  ceil(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.ceilArr(this.unwrap(arr)));
  }

  round(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.roundArr(this.unwrap(arr)));
  }

  neg(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.negArr(this.unwrap(arr)));
  }

  reciprocal(arr: IFaceNDArray): IFaceNDArray {
    const result = Array.from(arr.data).map(x => 1 / x);
    return this.array(result, arr.shape);
  }

  square(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.squareArr(this.unwrap(arr)));
  }

  // ============ Math - Binary ============

  add(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(a).add(this.unwrap(b)));
  }

  sub(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(a).sub(this.unwrap(b)));
  }

  mul(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(a).mul(this.unwrap(b)));
  }

  div(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(a).div(this.unwrap(b)));
  }

  pow(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    // Element-wise power - use scalar power for each element
    // This is a fallback since element-wise pow isn't directly available
    const aData = a.data;
    const bData = b.data;
    const result = new Float64Array(aData.length);
    for (let i = 0; i < aData.length; i++) {
      result[i] = Math.pow(aData[i], bData[i]);
    }
    return this.array(Array.from(result), a.shape);
  }

  maximum(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.maximum(this.unwrap(a), this.unwrap(b)));
  }

  minimum(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.minimum(this.unwrap(a), this.unwrap(b)));
  }

  // ============ Math - Scalar ============

  addScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).addScalar(scalar));
  }

  subScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).subScalar(scalar));
  }

  mulScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).mulScalar(scalar));
  }

  divScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).divScalar(scalar));
  }

  powScalar(arr: IFaceNDArray, scalar: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).powScalar(scalar));
  }

  clip(arr: IFaceNDArray, min: number, max: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).clip(min, max));
  }

  // ============ Stats ============

  sum(arr: IFaceNDArray): number {
    return this.unwrap(arr).sum();
  }

  prod(arr: IFaceNDArray): number {
    return this.unwrap(arr).prod();
  }

  mean(arr: IFaceNDArray): number {
    return this.unwrap(arr).mean();
  }

  var(arr: IFaceNDArray, ddof: number = 0): number {
    if (ddof === 0) {
      return this.unwrap(arr).var();
    }
    return this.unwrap(arr).varDdof(ddof);
  }

  std(arr: IFaceNDArray, ddof: number = 0): number {
    return this.unwrap(arr).std(ddof);
  }

  min(arr: IFaceNDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    return this.unwrap(arr).min();
  }

  max(arr: IFaceNDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    return this.unwrap(arr).max();
  }

  argmin(arr: IFaceNDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    return this.unwrap(arr).argmin();
  }

  argmax(arr: IFaceNDArray): number {
    if (arr.data.length === 0) throw new Error('zero-size array');
    return this.unwrap(arr).argmax();
  }

  cumsum(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(arr).cumsum(0));
  }

  cumprod(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(arr).cumprod(0));
  }

  all(arr: IFaceNDArray): boolean {
    return this.unwrap(arr).all() !== 0;
  }

  any(arr: IFaceNDArray): boolean {
    return this.unwrap(arr).any() !== 0;
  }

  sumAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).sumAxis(axis, false));
  }

  meanAxis(arr: IFaceNDArray, axis: number): IFaceNDArray {
    return this.wrap(this.unwrap(arr).meanAxis(axis, false));
  }

  // ============ Linalg ============

  matmul(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    // Use optimized f32 parallel matmul for better performance
    // The generic wasm.matmul goes through f64 which is much slower
    const aShape = a.shape;
    const bShape = b.shape;

    // Handle different cases
    if (aShape.length === 2 && bShape.length === 2) {
      // Validate dimensions: a is [m, k], b is [k2, n] - k must equal k2
      const m = aShape[0];
      const k = aShape[1];
      const k2 = bShape[0];
      const n = bShape[1];

      if (k !== k2) {
        throw new Error(`dimension mismatch for matmul: a.shape[1] (${k}) != b.shape[0] (${k2})`);
      }

      // Convert to f32 for optimized kernel
      const aData = new Float32Array(a.data);
      const bData = new Float32Array(b.data);

      // Use the optimized parallel f32 matmul (XNNPACK-style 6x8 kernel with packing)
      const cData = this.wasm.matmulF32OptimizedParallelV3(aData, bData, m, n, k);

      // Convert result back and wrap
      return this.array(Array.from(cData), [m, n]);
    }

    // Fallback to generic matmul for non-2D cases
    return this.wrap(this.wasm.matmul(this.unwrap(a), this.unwrap(b)));
  }

  dot(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.dot(this.unwrap(a), this.unwrap(b)));
  }

  inner(a: IFaceNDArray, b: IFaceNDArray): number {
    // Inner product: sum of element-wise multiplication
    const aData = a.data;
    const bData = b.data;
    let sum = 0;
    for (let i = 0; i < aData.length; i++) {
      sum += aData[i] * bData[i];
    }
    return sum;
  }

  outer(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    // Outer product: a[i] * b[j] for all i, j
    const aData = a.data;
    const bData = b.data;
    const m = aData.length;
    const n = bData.length;
    const result = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        result[i * n + j] = aData[i] * bData[j];
      }
    }
    return this.array(Array.from(result), [m, n]);
  }

  transpose(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.unwrap(arr).transpose());
  }

  trace(arr: IFaceNDArray): number {
    // Trace: sum of diagonal elements
    const shape = arr.shape;
    if (shape.length !== 2) throw new Error('Matrix must be 2D');
    const data = arr.data;
    const n = Math.min(shape[0], shape[1]);
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += data[i * shape[1] + i];
    }
    return sum;
  }

  det(arr: IFaceNDArray): number {
    return this.wasm.det(this.unwrap(arr));
  }

  inv(arr: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.inv(this.unwrap(arr)));
  }

  solve(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    return this.wrap(this.wasm.solve(this.unwrap(a), this.unwrap(b)));
  }

  norm(arr: IFaceNDArray, ord: number = 2): number {
    // WASM norm might have different signature - adjust as needed
    if (ord === Infinity) {
      return Math.max(...Array.from(arr.data).map(Math.abs));
    }
    if (ord === 1) {
      return Array.from(arr.data).reduce((acc, x) => acc + Math.abs(x), 0);
    }
    // Default L2
    return Math.sqrt(Array.from(arr.data).reduce((acc, x) => acc + x * x, 0));
  }

  qr(_arr: IFaceNDArray): { q: IFaceNDArray; r: IFaceNDArray } {
    // QR decomposition not yet implemented in WASM
    throw new Error('QR decomposition not yet implemented in WASM backend');
  }

  svd(_arr: IFaceNDArray): { u: IFaceNDArray; s: IFaceNDArray; vt: IFaceNDArray } {
    // SVD not yet implemented in WASM
    throw new Error('SVD not yet implemented in WASM backend');
  }

  // ============ Creation - Like Functions ============

  zerosLike(arr: IFaceNDArray): IFaceNDArray {
    return this.zeros(arr.shape);
  }

  onesLike(arr: IFaceNDArray): IFaceNDArray {
    return this.ones(arr.shape);
  }

  emptyLike(arr: IFaceNDArray): IFaceNDArray {
    return this.zeros(arr.shape);
  }

  fullLike(arr: IFaceNDArray, value: number): IFaceNDArray {
    return this.full(arr.shape, value);
  }

  // ============ Broadcasting ============

  private _computeStrides(shape: number[]): number[] {
    const strides = new Array(shape.length);
    let stride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= shape[i];
    }
    return strides;
  }

  private _computeBroadcastStrides(srcShape: number[], dstShape: number[]): number[] {
    const strides = new Array(dstShape.length);
    let srcStride = 1;
    for (let i = srcShape.length - 1; i >= 0; i--) {
      strides[i] = srcShape[i] === 1 ? 0 : srcStride;
      srcStride *= srcShape[i];
    }
    return strides;
  }

  broadcastTo(arr: IFaceNDArray, shape: number[]): IFaceNDArray {
    const arrShape = arr.shape;
    if (arrShape.length > shape.length) {
      throw new Error('Cannot broadcast to smaller number of dimensions');
    }

    const paddedShape = new Array(shape.length - arrShape.length).fill(1).concat(arrShape);

    for (let i = 0; i < shape.length; i++) {
      if (paddedShape[i] !== 1 && paddedShape[i] !== shape[i]) {
        throw new Error(`Cannot broadcast shape [${arrShape}] to [${shape}]`);
      }
    }

    const size = shape.reduce((a, b) => a * b, 1);
    const result = new Float64Array(size);
    const strides = this._computeStrides(shape);
    const srcStrides = this._computeBroadcastStrides(paddedShape, shape);

    for (let i = 0; i < size; i++) {
      let srcIdx = 0;
      let remaining = i;
      for (let d = 0; d < shape.length; d++) {
        const coord = Math.floor(remaining / strides[d]);
        remaining = remaining % strides[d];
        srcIdx += coord * srcStrides[d];
      }
      result[i] = arr.data[srcIdx];
    }

    return this.array(Array.from(result), shape);
  }

  broadcastArrays(...arrays: IFaceNDArray[]): IFaceNDArray[] {
    if (arrays.length === 0) return [];
    if (arrays.length === 1) return [this.array(Array.from(arrays[0].data), arrays[0].shape)];

    const shapes = arrays.map(a => a.shape);
    const maxDims = Math.max(...shapes.map(s => s.length));

    const paddedShapes = shapes.map(s => {
      const padded = new Array(maxDims - s.length).fill(1);
      return padded.concat(s);
    });

    const outShape: number[] = [];
    for (let i = 0; i < maxDims; i++) {
      const dims = paddedShapes.map(s => s[i]);
      const maxDim = Math.max(...dims);
      for (const d of dims) {
        if (d !== 1 && d !== maxDim) {
          throw new Error('Shapes are not broadcastable');
        }
      }
      outShape.push(maxDim);
    }

    return arrays.map(arr => this.broadcastTo(arr, outShape));
  }

  // ============ Shape Manipulation ============

  private _normalizeAxis(axis: number, ndim: number): number {
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
    }
    return axis;
  }

  private _transposeGeneral(arr: IFaceNDArray, perm: number[], newShape: number[]): IFaceNDArray {
    const size = arr.data.length;
    const result = new Float64Array(size);

    const oldStrides = this._computeStrides(arr.shape);
    const newStrides = this._computeStrides(newShape);

    for (let i = 0; i < size; i++) {
      const coords = new Array(newShape.length);
      let remaining = i;
      for (let d = 0; d < newShape.length; d++) {
        coords[d] = Math.floor(remaining / newStrides[d]);
        remaining = remaining % newStrides[d];
      }

      let oldIdx = 0;
      for (let d = 0; d < perm.length; d++) {
        oldIdx += coords[d] * oldStrides[perm[d]];
      }

      result[i] = arr.data[oldIdx];
    }

    return this.array(Array.from(result), newShape);
  }

  swapaxes(arr: IFaceNDArray, axis1: number, axis2: number): IFaceNDArray {
    const ndim = arr.shape.length;
    axis1 = this._normalizeAxis(axis1, ndim);
    axis2 = this._normalizeAxis(axis2, ndim);

    if (axis1 === axis2) {
      return this.array(Array.from(arr.data), arr.shape);
    }

    const newShape = [...arr.shape];
    [newShape[axis1], newShape[axis2]] = [newShape[axis2], newShape[axis1]];

    const perm = Array.from({ length: ndim }, (_, i) => i);
    [perm[axis1], perm[axis2]] = [perm[axis2], perm[axis1]];

    return this._transposeGeneral(arr, perm, newShape);
  }

  moveaxis(arr: IFaceNDArray, source: number, destination: number): IFaceNDArray {
    const ndim = arr.shape.length;
    source = this._normalizeAxis(source, ndim);
    destination = this._normalizeAxis(destination, ndim);

    if (source === destination) {
      return this.array(Array.from(arr.data), arr.shape);
    }

    const perm: number[] = [];
    for (let i = 0; i < ndim; i++) {
      if (i !== source) perm.push(i);
    }
    perm.splice(destination, 0, source);

    const newShape = perm.map(i => arr.shape[i]);
    return this._transposeGeneral(arr, perm, newShape);
  }

  squeeze(arr: IFaceNDArray, axis?: number): IFaceNDArray {
    if (axis !== undefined) {
      const normalizedAxis = this._normalizeAxis(axis, arr.shape.length);
      if (arr.shape[normalizedAxis] !== 1) {
        throw new Error(`cannot squeeze axis ${axis} with size ${arr.shape[normalizedAxis]}`);
      }
      const newShape = arr.shape.filter((_, i) => i !== normalizedAxis);
      return this.array(Array.from(arr.data), newShape.length === 0 ? [1] : newShape);
    }

    const newShape = arr.shape.filter(d => d !== 1);
    return this.array(Array.from(arr.data), newShape.length === 0 ? [1] : newShape);
  }

  expandDims(arr: IFaceNDArray, axis: number): IFaceNDArray {
    const ndim = arr.shape.length + 1;
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) {
      throw new Error(`axis ${axis} is out of bounds`);
    }

    const newShape = [...arr.shape];
    newShape.splice(axis, 0, 1);
    return this.array(Array.from(arr.data), newShape);
  }

  reshape(arr: IFaceNDArray, shape: number[]): IFaceNDArray {
    let inferIdx = -1;
    let knownSize = 1;
    for (let i = 0; i < shape.length; i++) {
      if (shape[i] === -1) {
        if (inferIdx !== -1) throw new Error('can only specify one unknown dimension');
        inferIdx = i;
      } else {
        knownSize *= shape[i];
      }
    }

    const newShape = [...shape];
    if (inferIdx !== -1) {
      newShape[inferIdx] = arr.data.length / knownSize;
    }

    const newSize = newShape.reduce((a, b) => a * b, 1);
    if (newSize !== arr.data.length) {
      throw new Error(`cannot reshape array of size ${arr.data.length} into shape [${newShape}]`);
    }

    return this.array(Array.from(arr.data), newShape);
  }

  flatten(arr: IFaceNDArray): IFaceNDArray {
    return this.array(Array.from(arr.data), [arr.data.length]);
  }

  concatenate(arrays: IFaceNDArray[], axis: number = 0): IFaceNDArray {
    if (arrays.length === 0) throw new Error('need at least one array to concatenate');
    if (arrays.length === 1) return this.array(Array.from(arrays[0].data), arrays[0].shape);

    const ndim = arrays[0].shape.length;
    axis = this._normalizeAxis(axis, ndim);

    for (let i = 1; i < arrays.length; i++) {
      if (arrays[i].shape.length !== ndim) {
        throw new Error('all input arrays must have same number of dimensions');
      }
      for (let d = 0; d < ndim; d++) {
        if (d !== axis && arrays[i].shape[d] !== arrays[0].shape[d]) {
          throw new Error('all input array dimensions except concat axis must match');
        }
      }
    }

    const outShape = [...arrays[0].shape];
    outShape[axis] = arrays.reduce((sum, arr) => sum + arr.shape[axis], 0);

    const outSize = outShape.reduce((a, b) => a * b, 1);
    const result = new Float64Array(outSize);

    if (ndim === 1) {
      let offset = 0;
      for (const arr of arrays) {
        result.set(arr.data, offset);
        offset += arr.data.length;
      }
      return this.array(Array.from(result), outShape);
    }

    const outStrides = this._computeStrides(outShape);

    let axisOffset = 0;
    for (const arr of arrays) {
      const srcStrides = this._computeStrides(arr.shape);
      const srcSize = arr.data.length;

      for (let srcIdx = 0; srcIdx < srcSize; srcIdx++) {
        const coords = new Array(ndim);
        let remaining = srcIdx;
        for (let d = 0; d < ndim; d++) {
          coords[d] = Math.floor(remaining / srcStrides[d]);
          remaining = remaining % srcStrides[d];
        }

        coords[axis] += axisOffset;

        let dstIdx = 0;
        for (let d = 0; d < ndim; d++) {
          dstIdx += coords[d] * outStrides[d];
        }

        result[dstIdx] = arr.data[srcIdx];
      }

      axisOffset += arr.shape[axis];
    }

    return this.array(Array.from(result), outShape);
  }

  stack(arrays: IFaceNDArray[], axis: number = 0): IFaceNDArray {
    if (arrays.length === 0) throw new Error('need at least one array to stack');

    const shape = arrays[0].shape;
    for (let i = 1; i < arrays.length; i++) {
      if (arrays[i].shape.length !== shape.length) {
        throw new Error('all input arrays must have the same shape');
      }
      for (let d = 0; d < shape.length; d++) {
        if (arrays[i].shape[d] !== shape[d]) {
          throw new Error('all input arrays must have the same shape');
        }
      }
    }

    const expanded = arrays.map(arr => this.expandDims(arr, axis));
    return this.concatenate(expanded, axis);
  }

  split(arr: IFaceNDArray, indices: number | number[], axis: number = 0): IFaceNDArray[] {
    const ndim = arr.shape.length;
    axis = this._normalizeAxis(axis, ndim);
    const axisSize = arr.shape[axis];

    let splitIndices: number[];
    if (typeof indices === 'number') {
      if (axisSize % indices !== 0) {
        throw new Error(`array of size ${axisSize} cannot be split into ${indices} equal parts`);
      }
      const partSize = axisSize / indices;
      splitIndices = [];
      for (let i = partSize; i < axisSize; i += partSize) {
        splitIndices.push(i);
      }
    } else {
      splitIndices = indices;
    }

    const results: IFaceNDArray[] = [];
    let start = 0;

    const getSlice = (startIdx: number, endIdx: number): IFaceNDArray => {
      const sliceShape = [...arr.shape];
      sliceShape[axis] = endIdx - startIdx;
      const sliceSize = sliceShape.reduce((a, b) => a * b, 1);
      const sliceData = new Float64Array(sliceSize);

      const srcStrides = this._computeStrides(arr.shape);
      const dstStrides = this._computeStrides(sliceShape);

      for (let dstIdx = 0; dstIdx < sliceSize; dstIdx++) {
        const coords = new Array(ndim);
        let remaining = dstIdx;
        for (let d = 0; d < ndim; d++) {
          coords[d] = Math.floor(remaining / dstStrides[d]);
          remaining = remaining % dstStrides[d];
        }

        coords[axis] += startIdx;

        let srcIdx = 0;
        for (let d = 0; d < ndim; d++) {
          srcIdx += coords[d] * srcStrides[d];
        }

        sliceData[dstIdx] = arr.data[srcIdx];
      }

      return this.array(Array.from(sliceData), sliceShape);
    };

    for (const idx of splitIndices) {
      results.push(getSlice(start, idx));
      start = idx;
    }
    results.push(getSlice(start, axisSize));

    return results;
  }

  // ============ Conditional ============

  where(condition: IFaceNDArray, x: IFaceNDArray, y: IFaceNDArray): IFaceNDArray {
    const [condBcast, xBcast, yBcast] = this.broadcastArrays(condition, x, y);
    const size = condBcast.data.length;
    const result = new Float64Array(size);

    for (let i = 0; i < size; i++) {
      result[i] = condBcast.data[i] !== 0 ? xBcast.data[i] : yBcast.data[i];
    }

    return this.array(Array.from(result), condBcast.shape);
  }

  // ============ Advanced Indexing ============

  take(arr: IFaceNDArray, indices: IFaceNDArray | number[], axis?: number): IFaceNDArray {
    const indexArray = Array.isArray(indices) ? indices : Array.from(indices.data);

    if (axis === undefined) {
      const result = new Float64Array(indexArray.length);
      for (let i = 0; i < indexArray.length; i++) {
        let idx = indexArray[i];
        if (idx < 0) idx += arr.data.length;
        result[i] = arr.data[idx];
      }
      return this.array(Array.from(result), [indexArray.length]);
    }

    const ndim = arr.shape.length;
    axis = this._normalizeAxis(axis, ndim);

    const outShape = [...arr.shape];
    outShape[axis] = indexArray.length;

    const outSize = outShape.reduce((a, b) => a * b, 1);
    const result = new Float64Array(outSize);

    const srcStrides = this._computeStrides(arr.shape);
    const dstStrides = this._computeStrides(outShape);

    for (let dstIdx = 0; dstIdx < outSize; dstIdx++) {
      const coords = new Array(ndim);
      let remaining = dstIdx;
      for (let d = 0; d < ndim; d++) {
        coords[d] = Math.floor(remaining / dstStrides[d]);
        remaining = remaining % dstStrides[d];
      }

      let srcAxisCoord = indexArray[coords[axis]];
      if (srcAxisCoord < 0) srcAxisCoord += arr.shape[axis];
      coords[axis] = srcAxisCoord;

      let srcIdx = 0;
      for (let d = 0; d < ndim; d++) {
        srcIdx += coords[d] * srcStrides[d];
      }

      result[dstIdx] = arr.data[srcIdx];
    }

    return this.array(Array.from(result), outShape);
  }

  // ============ Batched Operations ============

  batchedMatmul(a: IFaceNDArray, b: IFaceNDArray): IFaceNDArray {
    if (a.shape.length < 2 || b.shape.length < 2) {
      throw new Error('batchedMatmul requires at least 2D arrays');
    }

    const aM = a.shape[a.shape.length - 2];
    const aK = a.shape[a.shape.length - 1];
    const bK = b.shape[b.shape.length - 2];
    const bN = b.shape[b.shape.length - 1];

    if (aK !== bK) throw new Error('matmul inner dimensions must match');

    const aBatchShape = a.shape.slice(0, -2);
    const bBatchShape = b.shape.slice(0, -2);

    const maxBatchDims = Math.max(aBatchShape.length, bBatchShape.length);
    const paddedABatch = new Array(maxBatchDims - aBatchShape.length).fill(1).concat(aBatchShape);
    const paddedBBatch = new Array(maxBatchDims - bBatchShape.length).fill(1).concat(bBatchShape);

    const outBatchShape: number[] = [];
    for (let i = 0; i < maxBatchDims; i++) {
      const ad = paddedABatch[i];
      const bd = paddedBBatch[i];
      if (ad !== 1 && bd !== 1 && ad !== bd) {
        throw new Error('batch dimensions are not broadcastable');
      }
      outBatchShape.push(Math.max(ad, bd));
    }

    const outShape = [...outBatchShape, aM, bN];
    const batchSize = outBatchShape.length === 0 ? 1 : outBatchShape.reduce((x, y) => x * y, 1);
    const matSize = aM * bN;
    const result = new Float64Array(batchSize * matSize);

    const aBatchStrides = this._computeStrides(paddedABatch);
    const bBatchStrides = this._computeStrides(paddedBBatch);
    const outBatchStrides = this._computeStrides(outBatchShape);

    const aMatStride = aM * aK;
    const bMatStride = bK * bN;

    for (let batch = 0; batch < batchSize; batch++) {
      const coords = new Array(maxBatchDims);
      let remaining = batch;
      for (let d = 0; d < maxBatchDims; d++) {
        coords[d] = Math.floor(remaining / outBatchStrides[d]);
        remaining = remaining % outBatchStrides[d];
      }

      let aOffset = 0;
      let bOffset = 0;
      for (let d = 0; d < maxBatchDims; d++) {
        const aCoord = paddedABatch[d] === 1 ? 0 : coords[d];
        const bCoord = paddedBBatch[d] === 1 ? 0 : coords[d];
        aOffset += aCoord * aBatchStrides[d];
        bOffset += bCoord * bBatchStrides[d];
      }
      aOffset *= aMatStride;
      bOffset *= bMatStride;

      const outOffset = batch * matSize;
      for (let i = 0; i < aM; i++) {
        for (let j = 0; j < bN; j++) {
          let sum = 0;
          for (let k = 0; k < aK; k++) {
            sum += a.data[aOffset + i * aK + k] * b.data[bOffset + k * bN + j];
          }
          result[outOffset + i * bN + j] = sum;
        }
      }
    }

    return this.array(Array.from(result), outShape);
  }

  // ============ Einstein Summation ============

  einsum(subscripts: string, ...operands: IFaceNDArray[]): IFaceNDArray {
    const [inputStr, outputStr] = subscripts.split('->').map(s => s.trim());
    const inputSubscripts = inputStr.split(',').map(s => s.trim());

    if (inputSubscripts.length !== operands.length) {
      throw new Error(`einsum: expected ${inputSubscripts.length} operands, got ${operands.length}`);
    }

    const labelSizes: Map<string, number> = new Map();
    const inputLabels: string[][] = [];

    for (let i = 0; i < operands.length; i++) {
      const labels = inputSubscripts[i].split('');
      inputLabels.push(labels);
      if (labels.length !== operands[i].shape.length) {
        throw new Error(`einsum: operand ${i} has ${operands[i].shape.length} dimensions but subscripts specify ${labels.length}`);
      }
      for (let j = 0; j < labels.length; j++) {
        const label = labels[j];
        const size = operands[i].shape[j];
        if (labelSizes.has(label)) {
          if (labelSizes.get(label) !== size) {
            throw new Error(`einsum: inconsistent size for label '${label}'`);
          }
        } else {
          labelSizes.set(label, size);
        }
      }
    }

    let outputLabels: string[];
    if (outputStr !== undefined) {
      outputLabels = outputStr.split('');
    } else {
      const labelCounts: Map<string, number> = new Map();
      for (const labels of inputLabels) {
        for (const label of labels) {
          labelCounts.set(label, (labelCounts.get(label) || 0) + 1);
        }
      }
      outputLabels = [];
      const allLabels = Array.from(labelSizes.keys()).sort();
      for (const label of allLabels) {
        if (labelCounts.get(label) === 1) {
          outputLabels.push(label);
        }
      }
    }

    const outputShape = outputLabels.map(l => labelSizes.get(l)!);
    const outputSize = outputShape.length === 0 ? 1 : outputShape.reduce((a, b) => a * b, 1);

    const outputSet = new Set(outputLabels);
    const allLabels = Array.from(labelSizes.keys());
    const contractedLabels = allLabels.filter(l => !outputSet.has(l));

    const contractedSizes = contractedLabels.map(l => labelSizes.get(l)!);
    const contractedTotal = contractedSizes.length === 0 ? 1 : contractedSizes.reduce((a, b) => a * b, 1);

    const result = new Float64Array(outputSize);

    const inputStrides = operands.map(op => this._computeStrides(op.shape));

    const outputStrides = outputShape.length === 0 ? [] : this._computeStrides(outputShape);

    for (let outIdx = 0; outIdx < outputSize; outIdx++) {
      const outCoords: Map<string, number> = new Map();
      let remaining = outIdx;
      for (let d = 0; d < outputLabels.length; d++) {
        const coord = Math.floor(remaining / outputStrides[d]);
        remaining = remaining % outputStrides[d];
        outCoords.set(outputLabels[d], coord);
      }

      let sum = 0;
      for (let contrIdx = 0; contrIdx < contractedTotal; contrIdx++) {
        const contrCoords: Map<string, number> = new Map();
        let contrRemaining = contrIdx;
        for (let d = 0; d < contractedLabels.length; d++) {
          const size = contractedSizes[d];
          const stride = d < contractedSizes.length - 1
            ? contractedSizes.slice(d + 1).reduce((a, b) => a * b, 1)
            : 1;
          const coord = Math.floor(contrRemaining / stride);
          contrRemaining = contrRemaining % stride;
          contrCoords.set(contractedLabels[d], coord);
        }

        const allCoords = new Map([...outCoords, ...contrCoords]);

        let prod = 1;
        for (let i = 0; i < operands.length; i++) {
          const labels = inputLabels[i];
          const strides = inputStrides[i];
          let idx = 0;
          for (let d = 0; d < labels.length; d++) {
            idx += allCoords.get(labels[d])! * strides[d];
          }
          prod *= operands[i].data[idx];
        }
        sum += prod;
      }

      result[outIdx] = sum;
    }

    return this.array(Array.from(result), outputShape.length === 0 ? [1] : outputShape);
  }
}

/**
 * Initialize the WASM backend
 *
 * Must be called before createWasmBackend()
 */
export async function initWasmBackend(): Promise<void> {
  // Import the WASM module
  const module = await import('./wasm-pkg/rumpy_wasm.js');

  // In browser context, fetch the WASM file
  const wasmUrl = new URL('./wasm-pkg/rumpy_wasm_bg.wasm', import.meta.url);
  const wasmResponse = await fetch(wasmUrl);
  const wasmBytes = await wasmResponse.arrayBuffer();

  // Initialize the module with wasm bytes
  await module.default(wasmBytes);

  // Initialize thread pool (wasm-bindgen-rayon)
  // In browser, this sets up Web Workers
  await module.initThreadPool(navigator.hardwareConcurrency || 4);

  wasmModule = module;
}

/**
 * Create a WASM backend instance
 *
 * Requires initWasmBackend() to have been called first
 */
export function createWasmBackend(): Backend {
  if (!wasmModule) {
    throw new Error('WASM module not initialized. Call initWasmBackend() first.');
  }
  return new WasmBackend(wasmModule);
}
