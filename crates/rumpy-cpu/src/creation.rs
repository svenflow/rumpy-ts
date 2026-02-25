//! Array creation operations for CPU backend

use crate::{CpuArray, CpuBackend};
use ndarray::{Array2, ArrayD, IxDyn};
use rumpy_core::{ops::CreationOps, Result, RumpyError};

impl CreationOps for CpuBackend {
    type Array = CpuArray;

    fn zeros(shape: Vec<usize>) -> CpuArray {
        CpuArray::zeros(shape)
    }

    fn ones(shape: Vec<usize>) -> CpuArray {
        CpuArray::ones(shape)
    }

    fn full(shape: Vec<usize>, value: f64) -> CpuArray {
        CpuArray::full(shape, value)
    }

    fn arange(start: f64, stop: f64, step: f64) -> Result<CpuArray> {
        if step == 0.0 {
            return Err(RumpyError::InvalidArgument(
                "Step cannot be zero".to_string(),
            ));
        }

        if (step > 0.0 && start >= stop) || (step < 0.0 && start <= stop) {
            return Ok(CpuArray::from_ndarray(ArrayD::zeros(IxDyn(&[0]))));
        }

        let n = ((stop - start) / step).ceil() as usize;
        let values: Vec<f64> = (0..n).map(|i| start + (i as f64) * step).collect();

        Ok(CpuArray::from_ndarray(
            ArrayD::from_shape_vec(IxDyn(&[values.len()]), values).unwrap(),
        ))
    }

    fn linspace(start: f64, stop: f64, num: usize) -> CpuArray {
        if num == 0 {
            return CpuArray::from_ndarray(ArrayD::zeros(IxDyn(&[0])));
        }
        if num == 1 {
            return CpuArray::from_ndarray(
                ArrayD::from_shape_vec(IxDyn(&[1]), vec![start]).unwrap(),
            );
        }

        let step = (stop - start) / (num - 1) as f64;
        let values: Vec<f64> = (0..num).map(|i| start + (i as f64) * step).collect();

        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[num]), values).unwrap())
    }

    fn eye(n: usize) -> CpuArray {
        let mut arr = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            arr[[i, i]] = 1.0;
        }
        CpuArray::from_ndarray(arr.into_dyn())
    }

    fn diag(arr: &CpuArray, k: i32) -> Result<CpuArray> {
        let data = arr.as_ndarray();
        let shape = data.shape();

        if shape.len() == 1 {
            // Create diagonal matrix from 1D array
            let n = shape[0];
            let size = n + k.unsigned_abs() as usize;
            let mut result = Array2::<f64>::zeros((size, size));

            for i in 0..n {
                let row = if k >= 0 { i } else { i + (-k) as usize };
                let col = if k >= 0 { i + k as usize } else { i };
                if row < size && col < size {
                    result[[row, col]] = data[IxDyn(&[i])];
                }
            }

            Ok(CpuArray::from_ndarray(result.into_dyn()))
        } else if shape.len() == 2 {
            // Extract diagonal from 2D array
            let (m, n) = (shape[0], shape[1]);
            let start_row = if k >= 0 { 0 } else { (-k) as usize };
            let start_col = if k >= 0 { k as usize } else { 0 };

            let diag_len = std::cmp::min(m.saturating_sub(start_row), n.saturating_sub(start_col));

            let values: Vec<f64> = (0..diag_len)
                .map(|i| data[IxDyn(&[start_row + i, start_col + i])])
                .collect();

            Ok(CpuArray::from_ndarray(
                ArrayD::from_shape_vec(IxDyn(&[diag_len]), values).unwrap(),
            ))
        } else {
            Err(RumpyError::InvalidArgument(
                "diag requires 1D or 2D array".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rumpy_core::Array;

    #[test]
    fn test_zeros() {
        let arr = CpuBackend::zeros(vec![3, 4]);
        assert_eq!(arr.shape(), &[3, 4]);
        assert!(arr.as_f64_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let arr = CpuBackend::ones(vec![2, 3]);
        assert_eq!(arr.shape(), &[2, 3]);
        assert!(arr.as_f64_slice().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_full() {
        let arr = CpuBackend::full(vec![2, 2], 5.0);
        assert!(arr.as_f64_slice().iter().all(|&x| x == 5.0));
    }

    #[test]
    fn test_arange() {
        let arr = CpuBackend::arange(0.0, 5.0, 1.0).unwrap();
        assert_eq!(arr.as_f64_slice(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_arange_step() {
        let arr = CpuBackend::arange(0.0, 10.0, 2.0).unwrap();
        assert_eq!(arr.as_f64_slice(), vec![0.0, 2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_arange_negative_step() {
        let arr = CpuBackend::arange(5.0, 0.0, -1.0).unwrap();
        assert_eq!(arr.as_f64_slice(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_arange_zero_step() {
        let result = CpuBackend::arange(0.0, 5.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_linspace() {
        let arr = CpuBackend::linspace(0.0, 1.0, 5);
        let data = arr.as_f64_slice();
        assert_eq!(data.len(), 5);
        assert!((data[0] - 0.0).abs() < 1e-10);
        assert!((data[4] - 1.0).abs() < 1e-10);
        assert!((data[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_eye() {
        let arr = CpuBackend::eye(3);
        assert_eq!(arr.shape(), &[3, 3]);
        let data = arr.as_f64_slice();
        assert_eq!(data[0], 1.0); // [0,0]
        assert_eq!(data[1], 0.0); // [0,1]
        assert_eq!(data[4], 1.0); // [1,1]
        assert_eq!(data[8], 1.0); // [2,2]
    }

    #[test]
    fn test_diag_create() {
        let vec = CpuArray::from_f64_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let arr = CpuBackend::diag(&vec, 0).unwrap();
        assert_eq!(arr.shape(), &[3, 3]);
        let data = arr.as_f64_slice();
        assert_eq!(data[0], 1.0);
        assert_eq!(data[4], 2.0);
        assert_eq!(data[8], 3.0);
    }

    #[test]
    fn test_diag_extract() {
        let mat = CpuArray::from_f64_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        )
        .unwrap();
        let diag = CpuBackend::diag(&mat, 0).unwrap();
        assert_eq!(diag.as_f64_slice(), vec![1.0, 5.0, 9.0]);
    }
}
