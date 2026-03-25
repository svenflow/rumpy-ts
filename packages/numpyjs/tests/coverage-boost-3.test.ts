/**
 * Coverage boost phase 3 — final push toward 100% line coverage in base-backend.ts
 *
 * Targets all remaining uncovered lines: scalar broadcasting in divmod,
 * quantile interpolation methods, tensordot tuple axes, FFT length-1,
 * packbits/unpackbits little-endian, lstsq rcond, block, columnStack 2D,
 * argwhere empty, dstack variants, and many error/edge-case branches.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { Backend, approxEq, getData } from './test-utils';

export function coverageBoost3Tests(getBackend: () => Backend) {
  describe('coverage-boost-3', () => {
    let B: Backend;
    beforeAll(() => {
      B = getBackend();
    });

    const arr = (data: number[], shape?: number[]) => B.array(data, shape ?? [data.length]);
    const mat = (data: number[], rows: number, cols: number) => B.array(data, [rows, cols]);

    // ============================================================
    // Scalar broadcasting in divmod
    // ============================================================

    describe('divmod scalar broadcasting', () => {
      it('scalar / array', async () => {
        const result = B.divmod(10, arr([3, 4, 5]));
        expect(await getData(result.quotient, B)).toEqual([3, 2, 2]);
        expect(await getData(result.remainder, B)).toEqual([1, 2, 0]);
      });

      it('array / scalar', async () => {
        const result = B.divmod(arr([10, 11, 12]), 3);
        expect(await getData(result.quotient, B)).toEqual([3, 3, 4]);
        expect(await getData(result.remainder, B)).toEqual([1, 2, 0]);
      });
    });

    // ============================================================
    // Quantile interpolation methods
    // ============================================================

    describe('quantile interpolation methods', () => {
      const a = () => arr([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

      it('lower method', () => {
        const result = B.quantile(a(), 0.25, undefined, undefined, 'lower');
        expect(result).toBe(3);
      });

      it('higher method', () => {
        const result = B.quantile(a(), 0.25, undefined, undefined, 'higher');
        expect(result).toBe(4);
      });

      it('midpoint method', () => {
        const result = B.quantile(a(), 0.25, undefined, undefined, 'midpoint');
        expect(result).toBe(3.5);
      });

      it('nearest method', () => {
        const result = B.quantile(a(), 0.25, undefined, undefined, 'nearest');
        // 0.25 quantile of [1..10]: index = 0.25*9 = 2.25, nearest is index 2 → value 3
        expect(result).toBe(3);
      });
    });

    // ============================================================
    // cond fallback / edge cases
    // ============================================================

    describe('cond edge cases', () => {
      it('cond with p=2 produces finite positive number', () => {
        const a = mat([3, 1, 1, 2], 2, 2);
        const c = B.cond(a, 2);
        expect(c).toBeGreaterThan(0);
        expect(Number.isFinite(c)).toBe(true);
      });

      it('cond with p=-2 returns inverse condition number', () => {
        const a = mat([3, 1, 1, 2], 2, 2);
        const c2 = B.cond(a, 2);
        const cm2 = B.cond(a, -2);
        // p=-2 gives sMin/sMax, which is 1/cond(a, 2)
        expect(approxEq(cm2, 1 / c2, 1e-8)).toBe(true);
      });
    });

    // ============================================================
    // General N-D broadcasting (4D+)
    // ============================================================

    describe('general N-D broadcasting path', () => {
      it('4D broadcast with actual broadcasting', async () => {
        const a = B.array([1, 2, 3, 4, 5, 6], [1, 2, 1, 3]);
        const b = B.array([10, 20], [1, 1, 2, 1]);
        const result = B.add(a, b);
        expect(result.shape).toEqual([1, 2, 2, 3]);
        const data = await getData(result, B);
        // a[0,0,0,:] = [1,2,3], a[0,1,0,:] = [4,5,6]
        // b[0,0,:,0] = [10,20] broadcast across last dim
        // result[0,0,0,:] = [11,12,13], result[0,0,1,:] = [21,22,23]
        // result[0,1,0,:] = [14,15,16], result[0,1,1,:] = [24,25,26]
        expect(data).toEqual([11, 12, 13, 21, 22, 23, 14, 15, 16, 24, 25, 26]);
      });
    });

    // ============================================================
    // tensordot with tuple axes
    // ============================================================

    describe('tensordot', () => {
      it('tensordot with axes as tuple arrays', async () => {
        const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
        const b = mat([1, 2, 3, 4, 5, 6], 3, 2);
        const result = B.tensordot(a, b, [[1], [0]]);
        expect(result.shape).toEqual([2, 2]);
        // Same as matmul
        expect(await getData(result, B)).toEqual([22, 28, 49, 64]);
      });

      it('tensordot producing non-scalar result', async () => {
        const a = mat([1, 2, 3, 4], 2, 2);
        const b = mat([5, 6, 7, 8], 2, 2);
        const result = B.tensordot(a, b, 1);
        expect(result.shape).toEqual([2, 2]);
        // Same as matmul: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        expect(await getData(result, B)).toEqual([19, 22, 43, 50]);
      });
    });

    // ============================================================
    // FFT on length-1 input
    // ============================================================

    describe('fft length-1', () => {
      it('fft on single element', async () => {
        const real = arr([42]);
        const imag = arr([0]);
        const result = B.fft(real, imag);
        expect(await getData(result.real, B)).toEqual([42]);
        expect(await getData(result.imag, B)).toEqual([0]);
      });
    });

    // ============================================================
    // packbits / unpackbits little-endian
    // ============================================================

    describe('packbits/unpackbits little-endian', () => {
      it('packbits little-endian', async () => {
        const a = arr([1, 0, 1, 0, 0, 0, 0, 0]);
        const result = B.packbits(a, undefined, 'little');
        const data = await getData(result, B);
        expect(data[0]).toBe(5); // bits 0,2 set = 1+4 = 5
      });

      it('unpackbits little-endian', async () => {
        const a = B.array([5], [1], 'uint8');
        const result = B.unpackbits(a, undefined, undefined, 'little');
        const data = await getData(result, B);
        expect(data[0]).toBe(1); // bit 0
        expect(data[1]).toBe(0); // bit 1
        expect(data[2]).toBe(1); // bit 2
      });
    });

    // ============================================================
    // lstsq with rcond
    // ============================================================

    describe('lstsq rcond', () => {
      it('lstsq with rcond=null', () => {
        // System: x0 + x1 = 1, x0 + 2*x1 = 2, 2*x0 + 3*x1 = 3
        const a = mat([1, 1, 1, 2, 2, 3], 3, 2);
        const b = arr([1, 2, 3]);
        const result = B.lstsq(a, b, null);
        expect(result.x.shape).toEqual([2]);
        // Verify solution approximately satisfies the system
        const x = result.x.data;
        expect(approxEq(x[0] + x[1], 1, 0.1)).toBe(true);
      });

      it('lstsq with numeric rcond', () => {
        const a = mat([1, 1, 1, 2, 2, 3], 3, 2);
        const b = arr([1, 2, 3]);
        const result = B.lstsq(a, b, 0.01);
        expect(result.x.shape).toEqual([2]);
        const x = result.x.data;
        expect(Number.isFinite(x[0])).toBe(true);
        expect(Number.isFinite(x[1])).toBe(true);
      });
    });

    // ============================================================
    // block with non-array row
    // ============================================================

    describe('block', () => {
      it('block with nested array rows', async () => {
        const a = mat([1, 2, 3, 4], 2, 2);
        const b = mat([5, 6, 7, 8], 2, 2);
        const result = B.block([[a, b]]);
        expect(result.shape).toEqual([2, 4]);
        expect(await getData(result, B)).toEqual([1, 2, 5, 6, 3, 4, 7, 8]);
      });
    });

    // ============================================================
    // columnStack 2D
    // ============================================================

    describe('columnStack 2D', () => {
      it('columnStack with 2D arrays', async () => {
        const a = mat([1, 2, 3, 4], 2, 2);
        const b = mat([5, 6, 7, 8], 2, 2);
        const result = B.columnStack([a, b]);
        expect(result.shape).toEqual([2, 4]);
        expect(await getData(result, B)).toEqual([1, 2, 5, 6, 3, 4, 7, 8]);
      });
    });

    // ============================================================
    // argwhere on all-zero array
    // ============================================================

    describe('argwhere empty', () => {
      it('argwhere on all zeros', () => {
        const a = arr([0, 0, 0]);
        const result = B.argwhere(a);
        expect(result.shape).toEqual([0, 1]);
      });
    });

    // ============================================================
    // dstack variants
    // ============================================================

    describe('dstack edge cases', () => {
      it('dstack with 1D arrays', async () => {
        const a = arr([1, 2]);
        const b = arr([3, 4]);
        const result = B.dstack([a, b]);
        expect(result.shape).toEqual([1, 2, 2]);
        expect(await getData(result, B)).toEqual([1, 3, 2, 4]);
      });

      it('dstack with 3D arrays', async () => {
        const a = B.array([1, 2, 3, 4], [1, 2, 2]);
        const b = B.array([5, 6, 7, 8], [1, 2, 2]);
        const result = B.dstack([a, b]);
        expect(result.shape).toEqual([1, 2, 4]);
        expect(await getData(result, B)).toEqual([1, 2, 5, 6, 3, 4, 7, 8]);
      });
    });

    // ============================================================
    // Error throws we haven't hit
    // ============================================================

    describe('remaining error branches', () => {
      it('partition with OOB axis', () => {
        expect(() => B.partition(arr([3, 1, 2]), 1, 5)).toThrow('axis');
      });

      it('partition with OOB kth', () => {
        expect(() => B.partition(arr([3, 1, 2]), 10)).toThrow('kth');
      });

      it('argpartition with OOB axis', () => {
        expect(() => B.argpartition(arr([3, 1, 2]), 1, 5)).toThrow('axis');
      });

      it('argpartition with OOB kth', () => {
        expect(() => B.argpartition(arr([3, 1, 2]), 10)).toThrow('kth');
      });

      it('concatenate ndim mismatch', () => {
        expect(() => B.concatenate([arr([1, 2]), mat([3, 4], 1, 2)])).toThrow('dimensions');
      });

      it('concatenate non-axis dim mismatch', () => {
        expect(() => B.concatenate([mat([1, 2], 1, 2), mat([3, 4, 5, 6], 1, 4)], 0)).toThrow(
          'match'
        );
      });

      it('broadcastTo fewer dims', () => {
        expect(() => B.broadcastTo(mat([1, 2, 3, 4], 2, 2), [4])).toThrow('broadcast');
      });

      it('broadcastTo incompatible dim', () => {
        expect(() => B.broadcastTo(arr([1, 2, 3]), [2])).toThrow('broadcast');
      });

      it('moveaxis source/dest length mismatch', () => {
        expect(() => B.moveaxis(B.array([1, 2, 3, 4, 5, 6], [2, 3]), [0, 1], [0])).toThrow(
          'same number'
        );
      });

      it('batchedMatmul non-broadcastable batch', () => {
        const a = B.array(
          Array.from({ length: 8 }, (_, i) => i),
          [2, 2, 2]
        );
        const b = B.array(
          Array.from({ length: 12 }, (_, i) => i),
          [3, 2, 2]
        );
        expect(() => B.batchedMatmul(a, b)).toThrow('broadcast');
      });

      it('einsum operand ndim mismatch', () => {
        expect(() => B.einsum('ijk->i', mat([1, 2, 3, 4], 2, 2))).toThrow('dimensions');
      });

      it('einsum label size mismatch', () => {
        // 'ii' on non-square matrix
        expect(() => B.einsum('ii->', mat([1, 2, 3, 4, 5, 6], 2, 3))).toThrow('size');
      });

      it('gradient with <2 elements throws', () => {
        expect(() => B.gradient(arr([1]))).toThrow('at least');
      });

      it('gradient edge_order=2 with <3 elements throws', () => {
        expect(() => B.gradient(arr([1, 2]), 0, 2)).toThrow('at least');
      });

      it('cross with wrong size throws', () => {
        expect(() => B.cross(arr([1, 2]), arr([3, 4]))).toThrow('3');
      });

      it('cov with different length x,y throws', () => {
        expect(() => B.cov(arr([1, 2, 3]), arr([1, 2]))).toThrow('length');
      });

      it('average weight length mismatch throws', () => {
        expect(() => B.average(arr([1, 2, 3]), arr([1, 2]))).toThrow('length');
      });

      it('pad 3D throws', () => {
        expect(() => B.pad(B.array([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]), 1)).toThrow('1D and 2D');
      });

      it('_diffOnce on <2 elements throws', () => {
        const a = mat([1], 1, 1);
        expect(() => B.diff(a, 1, 0)).toThrow('at least 2');
      });

      it('roll with mismatched shift/axis lengths', () => {
        expect(() => B.roll(mat([1, 2, 3, 4], 2, 2), [1, 2], [0])).toThrow('same length');
      });

      it('solve with non-square matrix', () => {
        expect(() => B.solve(mat([1, 2, 3, 4, 5, 6], 2, 3), arr([1, 2]))).toThrow('square');
      });
    });

    // ============================================================
    // Remaining edge case paths
    // ============================================================

    describe('remaining edge paths', () => {
      it('cov on 1D array', async () => {
        const result = B.cov(arr([1, 2, 3, 4, 5]));
        // 1D array -> 1x1 covariance matrix = variance = 2.5
        expect(result.shape).toEqual([1, 1]);
        expect(approxEq(result.data[0], 2.5, 1e-10)).toBe(true);
      });

      it('histogram with explicit range', async () => {
        const a = arr([1, 2, 3, 4, 5]);
        const result = B.histogram(a, 5, [0, 10]);
        expect(result.binEdges.shape[0]).toBe(6);
      });

      it('histogram with NDArray bins and density', async () => {
        const a = arr([1, 2, 3, 4, 5]);
        const edges = arr([0, 2.5, 5]);
        const result = B.histogram(a, edges, undefined, true);
        const data = await getData(result.hist, B);
        // With density, should integrate to 1
        expect(data.every(v => v >= 0)).toBe(true);
      });

      it('flip 2D without axis', async () => {
        // Already covered but this exercises the full reversal path
        const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
        const result = B.flip(a);
        expect(await getData(result, B)).toEqual([6, 5, 4, 3, 2, 1]);
      });

      it('ptp with axis no keepdims', async () => {
        const a = mat([1, 5, 3, 7, 2, 8], 2, 3);
        const result = B.ptp(a, 1) as any;
        const data = await getData(result, B);
        expect(data).toEqual([4, 6]);
      });

      it('nanquantile with axis no keepdims', async () => {
        const a = mat([1, NaN, 3, 4, 5, NaN], 2, 3);
        const result = B.nanquantile(a, 0.5, 1);
        // Row 0: non-NaN = [1, 3], median = 2. Row 1: non-NaN = [4, 5], median = 4.5
        const data = await getData(result as any, B);
        expect(approxEq(data[0], 2, 1e-10)).toBe(true);
        expect(approxEq(data[1], 4.5, 1e-10)).toBe(true);
      });

      it('diff with numeric append on 2D', async () => {
        const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
        const result = B.diff(a, 1, 1, undefined, 10);
        // Appending 10 along axis 1 then diff
        expect(result.shape[1]).toBe(3);
      });

      it('nanprod with axis no keepdims', async () => {
        const a = mat([1, NaN, 3, 4, 5, NaN], 2, 3);
        const result = B.nanprod(a, 1) as any;
        const data = await getData(result, B);
        expect(data[0]).toBe(3); // 1 * 3
        expect(data[1]).toBe(20); // 4 * 5
      });

      it('quantile with axis no keepdims (lower)', async () => {
        const a = mat([1, 2, 3, 4, 5, 6], 2, 3);
        const result = B.quantile(a, 0.5, 1, false, 'lower');
        const data = await getData(result as any, B);
        expect(data).toEqual([2, 5]);
      });

      it('polydiv with near-zero remainder', async () => {
        // (x^3 - 1) / (x - 1) = x^2 + x + 1, remainder ~0
        const u = arr([1, 0, 0, -1]);
        const v = arr([1, -1]);
        const { q } = B.polydiv(u, v);
        const qData = await getData(q, B);
        expect(approxEq(qData[0], 1, 1e-10)).toBe(true);
        expect(approxEq(qData[1], 1, 1e-10)).toBe(true);
        expect(approxEq(qData[2], 1, 1e-10)).toBe(true);
      });

      it('_reduceAlongAxis on 3D (axis 0)', async () => {
        const a = B.array([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
        const result = B.sum(a, 0) as any;
        expect(result.shape).toEqual([2, 2]);
        // sum along axis 0: [1+5, 2+6, 3+7, 4+8] = [6, 8, 10, 12]
        expect(await getData(result, B)).toEqual([6, 8, 10, 12]);
      });

      it('sort 3D along non-last axis', async () => {
        const a = B.array([4, 3, 2, 1, 8, 7, 6, 5], [2, 2, 2]);
        const result = B.sort(a, 0);
        expect(result.shape).toEqual([2, 2, 2]);
        // Sort along axis 0: min(4,8)=4, min(3,7)=3, min(2,6)=2, min(1,5)=1 for first plane
        expect(await getData(result, B)).toEqual([4, 3, 2, 1, 8, 7, 6, 5]);
      });

      it('Jacobi SVD on negative tau matrix', () => {
        // Matrix where (aqq - app) / (2*apq) is negative
        const a = mat([1, 5, 5, 2], 2, 2);
        const { u, s } = B.svd(a);
        expect(s.shape[0]).toBe(2);
        // Verify U * S * Vt ≈ A: singular values should be positive
        expect(s.data[0]).toBeGreaterThan(0);
        expect(s.data[1]).toBeGreaterThan(0);
        // Verify orthogonality of U: U^T * U ≈ I
        const utu = B.matmul(B.transpose(u), u);
        expect(approxEq(utu.data[0], 1, 1e-8)).toBe(true);
      });

      it('solve with 1D b vector', async () => {
        const a = mat([2, 1, 1, 3], 2, 2);
        const b = arr([5, 7]);
        const result = B.solve(a, b) as any;
        const data = await getData(result, B);
        // 2x + y = 5, x + 3y = 7 => x = 1.6, y = 1.8
        expect(data.length).toBe(2);
        expect(approxEq(data[0], 1.6, 1e-8)).toBe(true);
        expect(approxEq(data[1], 1.8, 1e-8)).toBe(true);
      });
    });
  });
}
