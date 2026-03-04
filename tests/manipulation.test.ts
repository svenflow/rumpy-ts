/**
 * Array manipulation and high-priority ops tests
 * Covers: zeros_like, broadcast_to, swapaxes, where, einsum, batched matmul, etc.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { Backend, DEFAULT_TOL, RELAXED_TOL, approxEq, arraysApproxEq } from './test-utils';

export function manipulationTests(getBackend: () => Backend) {
  describe('manipulation', () => {
    let B: Backend;
    beforeAll(() => {
      B = getBackend();
    });

    // ============ Like Functions ============

    describe('zerosLike', () => {
      it('creates zeros with same shape', () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [2, 3]);
        const zeros = B.zerosLike(arr);
        expect(zeros.shape).toEqual([2, 3]);
        expect(zeros.data.every(x => x === 0)).toBe(true);
      });
    });

    describe('onesLike', () => {
      it('creates ones with same shape', () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [2, 3]);
        const ones = B.onesLike(arr);
        expect(ones.shape).toEqual([2, 3]);
        expect(ones.data.every(x => x === 1)).toBe(true);
      });
    });

    describe('fullLike', () => {
      it('creates full array with same shape', () => {
        const arr = B.array([1, 2, 3, 4], [2, 2]);
        const full = B.fullLike(arr, 7.5);
        expect(full.shape).toEqual([2, 2]);
        expect(full.data.every(x => x === 7.5)).toBe(true);
      });
    });

    // ============ Broadcasting ============

    describe('broadcastTo', () => {
      it('broadcasts scalar to shape', () => {
        const arr = B.array([5], [1]);
        const result = B.broadcastTo(arr, [3]);
        expect(result.shape).toEqual([3]);
        expect(result.toArray()).toEqual([5, 5, 5]);
      });

      it('broadcasts 1D to 2D', () => {
        const arr = B.array([1, 2, 3], [3]);
        const result = B.broadcastTo(arr, [2, 3]);
        expect(result.shape).toEqual([2, 3]);
        expect(result.toArray()).toEqual([1, 2, 3, 1, 2, 3]);
      });

      it('broadcasts with leading dimensions', () => {
        const arr = B.array([1, 2], [1, 2]);
        const result = B.broadcastTo(arr, [3, 2]);
        expect(result.shape).toEqual([3, 2]);
        expect(result.toArray()).toEqual([1, 2, 1, 2, 1, 2]);
      });
    });

    describe('broadcastArrays', () => {
      it('broadcasts two arrays to common shape', () => {
        const a = B.array([1, 2, 3], [3]);
        const b = B.array([4], [1]);
        const [aBcast, bBcast] = B.broadcastArrays(a, b);
        expect(aBcast.shape).toEqual([3]);
        expect(bBcast.shape).toEqual([3]);
        expect(aBcast.toArray()).toEqual([1, 2, 3]);
        expect(bBcast.toArray()).toEqual([4, 4, 4]);
      });

      it('broadcasts 1D and 2D arrays', () => {
        const a = B.array([1, 2, 3], [3]);
        const b = B.array([10, 20], [2, 1]);
        const [aBcast, bBcast] = B.broadcastArrays(a, b);
        expect(aBcast.shape).toEqual([2, 3]);
        expect(bBcast.shape).toEqual([2, 3]);
      });
    });

    // ============ Shape Manipulation ============

    describe('swapaxes', () => {
      it('swaps axes in 2D array', () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [2, 3]);
        const result = B.swapaxes(arr, 0, 1);
        expect(result.shape).toEqual([3, 2]);
        // Original: [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
        expect(result.toArray()).toEqual([1, 4, 2, 5, 3, 6]);
      });

      it('handles negative axis', () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [2, 3]);
        const result = B.swapaxes(arr, 0, -1);
        expect(result.shape).toEqual([3, 2]);
      });

      it('same axis returns copy', () => {
        const arr = B.array([1, 2, 3, 4], [2, 2]);
        const result = B.swapaxes(arr, 0, 0);
        expect(result.shape).toEqual([2, 2]);
        expect(result.toArray()).toEqual([1, 2, 3, 4]);
      });
    });

    describe('moveaxis', () => {
      it('moves axis to new position', () => {
        const arr = B.array(Array.from({ length: 24 }, (_, i) => i), [2, 3, 4]);
        const result = B.moveaxis(arr, 0, -1);
        expect(result.shape).toEqual([3, 4, 2]);
      });

      it('moves axis forward', () => {
        const arr = B.array(Array.from({ length: 24 }, (_, i) => i), [2, 3, 4]);
        const result = B.moveaxis(arr, 2, 0);
        expect(result.shape).toEqual([4, 2, 3]);
      });
    });

    describe('squeeze', () => {
      it('removes all size-1 dimensions', () => {
        const arr = B.array([1, 2, 3], [1, 3, 1]);
        const result = B.squeeze(arr);
        expect(result.shape).toEqual([3]);
      });

      it('removes specific axis', () => {
        const arr = B.array([1, 2, 3], [1, 3]);
        const result = B.squeeze(arr, 0);
        expect(result.shape).toEqual([3]);
      });

      it('handles negative axis', () => {
        const arr = B.array([1, 2, 3], [3, 1]);
        const result = B.squeeze(arr, -1);
        expect(result.shape).toEqual([3]);
      });
    });

    describe('expandDims', () => {
      it('adds dimension at start', () => {
        const arr = B.array([1, 2, 3], [3]);
        const result = B.expandDims(arr, 0);
        expect(result.shape).toEqual([1, 3]);
      });

      it('adds dimension at end', () => {
        const arr = B.array([1, 2, 3], [3]);
        const result = B.expandDims(arr, 1);
        expect(result.shape).toEqual([3, 1]);
      });

      it('handles negative axis', () => {
        const arr = B.array([1, 2, 3], [3]);
        const result = B.expandDims(arr, -1);
        expect(result.shape).toEqual([3, 1]);
      });
    });

    describe('reshape', () => {
      it('reshapes to new shape', () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [6]);
        const result = B.reshape(arr, [2, 3]);
        expect(result.shape).toEqual([2, 3]);
        expect(result.toArray()).toEqual([1, 2, 3, 4, 5, 6]);
      });

      it('handles -1 dimension inference', () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [6]);
        const result = B.reshape(arr, [2, -1]);
        expect(result.shape).toEqual([2, 3]);
      });
    });

    describe('flatten', () => {
      it('flattens 2D array', () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [2, 3]);
        const result = B.flatten(arr);
        expect(result.shape).toEqual([6]);
        expect(result.toArray()).toEqual([1, 2, 3, 4, 5, 6]);
      });
    });

    describe('concatenate', () => {
      it('concatenates along axis 0', () => {
        const a = B.array([1, 2], [2]);
        const b = B.array([3, 4, 5], [3]);
        const result = B.concatenate([a, b], 0);
        expect(result.shape).toEqual([5]);
        expect(result.toArray()).toEqual([1, 2, 3, 4, 5]);
      });

      it('concatenates 2D arrays along axis 0', () => {
        const a = B.array([1, 2, 3, 4], [2, 2]);
        const b = B.array([5, 6], [1, 2]);
        const result = B.concatenate([a, b], 0);
        expect(result.shape).toEqual([3, 2]);
        expect(result.toArray()).toEqual([1, 2, 3, 4, 5, 6]);
      });

      it('concatenates along axis 1', () => {
        const a = B.array([1, 2, 3, 4], [2, 2]);
        const b = B.array([5, 6], [2, 1]);
        const result = B.concatenate([a, b], 1);
        expect(result.shape).toEqual([2, 3]);
        expect(result.toArray()).toEqual([1, 2, 5, 3, 4, 6]);
      });
    });

    describe('stack', () => {
      it('stacks along axis 0', () => {
        const a = B.array([1, 2, 3], [3]);
        const b = B.array([4, 5, 6], [3]);
        const result = B.stack([a, b], 0);
        expect(result.shape).toEqual([2, 3]);
        expect(result.toArray()).toEqual([1, 2, 3, 4, 5, 6]);
      });

      it('stacks along axis 1', () => {
        const a = B.array([1, 2, 3], [3]);
        const b = B.array([4, 5, 6], [3]);
        const result = B.stack([a, b], 1);
        expect(result.shape).toEqual([3, 2]);
        expect(result.toArray()).toEqual([1, 4, 2, 5, 3, 6]);
      });
    });

    describe('split', () => {
      it('splits into equal parts', () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [6]);
        const parts = B.split(arr, 3, 0);
        expect(parts.length).toBe(3);
        expect(parts[0].toArray()).toEqual([1, 2]);
        expect(parts[1].toArray()).toEqual([3, 4]);
        expect(parts[2].toArray()).toEqual([5, 6]);
      });

      it('splits at indices', () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [6]);
        const parts = B.split(arr, [2, 4], 0);
        expect(parts.length).toBe(3);
        expect(parts[0].toArray()).toEqual([1, 2]);
        expect(parts[1].toArray()).toEqual([3, 4]);
        expect(parts[2].toArray()).toEqual([5, 6]);
      });
    });

    // ============ Conditional ============

    describe('where', () => {
      it('selects based on condition', () => {
        const cond = B.array([1, 0, 1, 0], [4]); // truthy, falsy, truthy, falsy
        const x = B.array([1, 2, 3, 4], [4]);
        const y = B.array([10, 20, 30, 40], [4]);
        const result = B.where(cond, x, y);
        expect(result.toArray()).toEqual([1, 20, 3, 40]);
      });

      it('broadcasts condition', () => {
        const cond = B.array([1, 0], [2]);
        const x = B.array([1, 2, 3, 4, 5, 6], [3, 2]);
        const y = B.array([10, 20, 30, 40, 50, 60], [3, 2]);
        const result = B.where(cond, x, y);
        expect(result.shape).toEqual([3, 2]);
        expect(result.toArray()).toEqual([1, 20, 3, 40, 5, 60]);
      });
    });

    // ============ Advanced Indexing ============

    describe('take', () => {
      it('takes elements by indices', () => {
        const arr = B.array([10, 20, 30, 40, 50], [5]);
        const result = B.take(arr, [0, 2, 4]);
        expect(result.toArray()).toEqual([10, 30, 50]);
      });

      it('takes along axis', () => {
        const arr = B.array([1, 2, 3, 4, 5, 6], [2, 3]);
        const result = B.take(arr, [0, 2], 1);
        expect(result.shape).toEqual([2, 2]);
        expect(result.toArray()).toEqual([1, 3, 4, 6]);
      });

      it('handles negative indices', () => {
        const arr = B.array([10, 20, 30, 40, 50], [5]);
        const result = B.take(arr, [-1, -2]);
        expect(result.toArray()).toEqual([50, 40]);
      });
    });

    // ============ Batched Matmul ============

    describe('batchedMatmul', () => {
      it('performs batched matrix multiplication', () => {
        // Two 2x2 matrices in batch
        const a = B.array([
          1, 2, 3, 4,  // First 2x2
          5, 6, 7, 8   // Second 2x2
        ], [2, 2, 2]);
        const b = B.array([
          1, 0, 0, 1,  // Identity
          1, 0, 0, 1   // Identity
        ], [2, 2, 2]);
        const result = B.batchedMatmul(a, b);
        expect(result.shape).toEqual([2, 2, 2]);
        // A @ I = A
        expect(result.toArray()).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
      });

      it('broadcasts batch dimensions', () => {
        // (2, 2, 2) @ (2, 2) -> (2, 2, 2)
        const a = B.array([
          1, 2, 3, 4,
          5, 6, 7, 8
        ], [2, 2, 2]);
        const b = B.array([1, 0, 0, 1], [2, 2]); // Single 2x2 identity
        const result = B.batchedMatmul(a, b);
        expect(result.shape).toEqual([2, 2, 2]);
      });
    });

    // ============ Einstein Summation ============

    describe('einsum', () => {
      it('computes matrix multiplication: ij,jk->ik', () => {
        const a = B.array([1, 2, 3, 4], [2, 2]);
        const b = B.array([1, 0, 0, 1], [2, 2]);
        const result = B.einsum('ij,jk->ik', a, b);
        expect(result.shape).toEqual([2, 2]);
        // A @ I = A
        expect(result.toArray()).toEqual([1, 2, 3, 4]);
      });

      it('computes trace: ii->', () => {
        const a = B.array([1, 2, 3, 4], [2, 2]);
        const result = B.einsum('ii->', a);
        expect(result.toArray()).toEqual([5]); // 1 + 4
      });

      it('computes transpose: ij->ji', () => {
        const a = B.array([1, 2, 3, 4, 5, 6], [2, 3]);
        const result = B.einsum('ij->ji', a);
        expect(result.shape).toEqual([3, 2]);
        expect(result.toArray()).toEqual([1, 4, 2, 5, 3, 6]);
      });

      it('computes outer product: i,j->ij', () => {
        const a = B.array([1, 2, 3], [3]);
        const b = B.array([4, 5], [2]);
        const result = B.einsum('i,j->ij', a, b);
        expect(result.shape).toEqual([3, 2]);
        expect(result.toArray()).toEqual([4, 5, 8, 10, 12, 15]);
      });

      it('computes dot product: i,i->', () => {
        const a = B.array([1, 2, 3], [3]);
        const b = B.array([4, 5, 6], [3]);
        const result = B.einsum('i,i->', a, b);
        expect(result.toArray()).toEqual([32]); // 1*4 + 2*5 + 3*6
      });

      it('computes batch matmul: bij,bjk->bik', () => {
        // Two 2x2 matrices in batch
        const a = B.array([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
        const b = B.array([1, 0, 0, 1, 1, 0, 0, 1], [2, 2, 2]); // Two identity matrices
        const result = B.einsum('bij,bjk->bik', a, b);
        expect(result.shape).toEqual([2, 2, 2]);
        expect(result.toArray()).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
      });

      it('computes element-wise multiply and sum: ij,ij->', () => {
        const a = B.array([1, 2, 3, 4], [2, 2]);
        const b = B.array([1, 1, 1, 1], [2, 2]);
        const result = B.einsum('ij,ij->', a, b);
        expect(result.toArray()).toEqual([10]); // 1+2+3+4
      });

      it('implicit output (sum repeated indices)', () => {
        // ij,jk with no explicit output -> ik
        const a = B.array([1, 2, 3, 4], [2, 2]);
        const b = B.array([1, 0, 0, 1], [2, 2]);
        const result = B.einsum('ij,jk', a, b);
        expect(result.shape).toEqual([2, 2]);
      });
    });
  });
}
