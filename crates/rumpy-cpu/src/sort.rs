//! Sorting and searching operations for CPU backend

use crate::{CpuArray, CpuBackend};
use ndarray::{ArrayD, IxDyn};
use rumpy_core::{ops::SortOps, Array};
use std::cmp::Ordering;

/// NaN-safe comparison for sorting: NaN values sort to the end (NumPy behavior)
fn nan_safe_cmp(a: &f64, b: &f64) -> Ordering {
    a.partial_cmp(b).unwrap_or_else(|| {
        // Handle NaN: NaN is "greater" than all values, so sorts to end
        match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => unreachable!(), // partial_cmp only returns None for NaN
        }
    })
}

impl SortOps for CpuBackend {
    type Array = CpuArray;

    fn sort(arr: &CpuArray, _axis: Option<usize>) -> CpuArray {
        // For simplicity, always sort flattened array
        // TODO: Implement axis-aware sorting
        let mut data = arr.as_f64_slice();
        data.sort_by(nan_safe_cmp);
        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(arr.shape()), data).unwrap())
    }

    fn argsort(arr: &CpuArray, _axis: Option<usize>) -> CpuArray {
        let data = arr.as_f64_slice();
        let mut indices: Vec<usize> = (0..data.len()).collect();
        indices.sort_by(|&a, &b| nan_safe_cmp(&data[a], &data[b]));

        let result: Vec<f64> = indices.iter().map(|&i| i as f64).collect();
        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[result.len()]), result).unwrap())
    }

    fn searchsorted(arr: &CpuArray, values: &CpuArray) -> CpuArray {
        let data = arr.as_f64_slice();
        let vals = values.as_f64_slice();

        let result: Vec<f64> = vals
            .iter()
            .map(|&v| {
                // Binary search for insertion point with NaN-safe comparison
                // NaN values in the search array sort to the end
                data.binary_search_by(|probe| nan_safe_cmp(probe, &v))
                    .unwrap_or_else(|i| i) as f64
            })
            .collect();

        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[result.len()]), result).unwrap())
    }

    fn unique(arr: &CpuArray) -> CpuArray {
        let mut data = arr.as_f64_slice();
        data.sort_by(nan_safe_cmp);
        // dedup removes consecutive duplicates, but NaN != NaN so we need special handling
        // Keep only one NaN at the end
        let had_nan = data.iter().any(|x| x.is_nan());
        data.retain(|x| !x.is_nan());
        data.dedup();
        if had_nan {
            data.push(f64::NAN);
        }

        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap())
    }

    fn nonzero(arr: &CpuArray) -> Vec<CpuArray> {
        let data = arr.as_ndarray();
        let shape = data.shape();
        let ndim = shape.len();

        // Collect indices where value != 0
        let mut indices: Vec<Vec<usize>> = vec![Vec::new(); ndim];

        for (flat_idx, &val) in data.iter().enumerate() {
            if val != 0.0 {
                // Convert flat index to multi-dimensional indices
                let mut idx = flat_idx;
                let mut coords = vec![0usize; ndim];
                for d in (0..ndim).rev() {
                    coords[d] = idx % shape[d];
                    idx /= shape[d];
                }
                // Push coordinates in the correct order
                for (d, &coord) in coords.iter().enumerate() {
                    indices[d].push(coord);
                }
            }
        }

        // Convert to CpuArrays
        indices
            .into_iter()
            .map(|idx| {
                let data: Vec<f64> = idx.iter().map(|&i| i as f64).collect();
                CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(data: Vec<f64>) -> CpuArray {
        CpuArray::from_f64_vec(data.clone(), vec![data.len()]).unwrap()
    }

    #[test]
    fn test_sort() {
        let a = arr(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        let sorted = CpuBackend::sort(&a, None);
        assert_eq!(
            sorted.as_f64_slice(),
            vec![1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0]
        );
    }

    #[test]
    fn test_argsort() {
        let a = arr(vec![3.0, 1.0, 4.0, 1.0, 5.0]);
        let indices = CpuBackend::argsort(&a, None);
        // Values at indices 1, 3 are smallest (1.0), then 0 (3.0), then 2 (4.0), then 4 (5.0)
        assert_eq!(indices.as_f64_slice(), vec![1.0, 3.0, 0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_searchsorted() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let values = arr(vec![2.5, 0.0, 5.0, 6.0]);
        let indices = CpuBackend::searchsorted(&a, &values);
        // 2.5 -> index 2, 0.0 -> index 0, 5.0 -> index 4, 6.0 -> index 5
        assert_eq!(indices.as_f64_slice(), vec![2.0, 0.0, 4.0, 5.0]);
    }

    #[test]
    fn test_unique() {
        let a = arr(vec![3.0, 1.0, 2.0, 1.0, 3.0, 2.0]);
        let unique = CpuBackend::unique(&a);
        assert_eq!(unique.as_f64_slice(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_nonzero_1d() {
        let a = arr(vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0]);
        let indices = CpuBackend::nonzero(&a);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].as_f64_slice(), vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_nonzero_2d() {
        let a = CpuArray::from_f64_vec(vec![0.0, 1.0, 2.0, 0.0, 3.0, 0.0], vec![2, 3]).unwrap();
        let indices = CpuBackend::nonzero(&a);
        assert_eq!(indices.len(), 2);
        // Non-zero at (0,1), (0,2), (1,1)
        assert_eq!(indices[0].as_f64_slice(), vec![0.0, 0.0, 1.0]); // row indices
        assert_eq!(indices[1].as_f64_slice(), vec![1.0, 2.0, 1.0]); // col indices
    }
}
