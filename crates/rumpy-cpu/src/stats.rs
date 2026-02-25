//! Statistical operations for CPU backend

use crate::{CpuArray, CpuBackend};
use ndarray::{ArrayD, Axis, IxDyn};
use rumpy_core::{ops::StatsOps, Array, Result, RumpyError};

impl StatsOps for CpuBackend {
    type Array = CpuArray;

    fn sum(arr: &CpuArray) -> f64 {
        arr.as_ndarray().sum()
    }

    fn prod(arr: &CpuArray) -> f64 {
        arr.as_ndarray().product()
    }

    fn mean(arr: &CpuArray) -> f64 {
        let data = arr.as_ndarray();
        if data.is_empty() {
            return f64::NAN;
        }
        data.sum() / data.len() as f64
    }

    fn var(arr: &CpuArray) -> f64 {
        let data = arr.as_ndarray();
        if data.is_empty() {
            return f64::NAN;
        }
        let mean = data.sum() / data.len() as f64;
        data.mapv(|x| (x - mean).powi(2)).sum() / data.len() as f64
    }

    fn std(arr: &CpuArray) -> f64 {
        Self::var(arr).sqrt()
    }

    fn min(arr: &CpuArray) -> f64 {
        arr.as_ndarray()
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
    }

    fn max(arr: &CpuArray) -> f64 {
        arr.as_ndarray()
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    fn argmin(arr: &CpuArray) -> usize {
        arr.as_ndarray()
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn argmax(arr: &CpuArray) -> usize {
        arr.as_ndarray()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn sum_axis(arr: &CpuArray, axis: usize) -> Result<CpuArray> {
        let data = arr.as_ndarray();
        if axis >= data.ndim() {
            return Err(RumpyError::InvalidAxis {
                axis,
                ndim: data.ndim(),
            });
        }
        Ok(CpuArray::from_ndarray(data.sum_axis(Axis(axis))))
    }

    fn mean_axis(arr: &CpuArray, axis: usize) -> Result<CpuArray> {
        let data = arr.as_ndarray();
        if axis >= data.ndim() {
            return Err(RumpyError::InvalidAxis {
                axis,
                ndim: data.ndim(),
            });
        }
        Ok(CpuArray::from_ndarray(data.mean_axis(Axis(axis)).unwrap()))
    }

    fn min_axis(arr: &CpuArray, axis: usize) -> Result<CpuArray> {
        let data = arr.as_ndarray();
        if axis >= data.ndim() {
            return Err(RumpyError::InvalidAxis {
                axis,
                ndim: data.ndim(),
            });
        }
        // ndarray doesn't have min_axis directly, we need to implement it
        let shape = data.shape();
        let new_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &s)| s)
            .collect();

        let result_size: usize = new_shape.iter().product();
        let mut result = vec![f64::INFINITY; result_size];

        for (i, &val) in data.iter().enumerate() {
            // Calculate result index by removing the axis dimension
            let mut idx = i;
            let mut result_idx = 0;
            let mut multiplier = 1;

            for d in (0..shape.len()).rev() {
                let coord = idx % shape[d];
                idx /= shape[d];

                if d != axis {
                    result_idx += coord * multiplier;
                    multiplier *= shape[d];
                }
            }

            result[result_idx] = result[result_idx].min(val);
        }

        Ok(CpuArray::from_ndarray(
            ArrayD::from_shape_vec(IxDyn(&new_shape), result).unwrap(),
        ))
    }

    fn max_axis(arr: &CpuArray, axis: usize) -> Result<CpuArray> {
        let data = arr.as_ndarray();
        if axis >= data.ndim() {
            return Err(RumpyError::InvalidAxis {
                axis,
                ndim: data.ndim(),
            });
        }

        let shape = data.shape();
        let new_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &s)| s)
            .collect();

        let result_size: usize = new_shape.iter().product();
        let mut result = vec![f64::NEG_INFINITY; result_size];

        for (i, &val) in data.iter().enumerate() {
            let mut idx = i;
            let mut result_idx = 0;
            let mut multiplier = 1;

            for d in (0..shape.len()).rev() {
                let coord = idx % shape[d];
                idx /= shape[d];

                if d != axis {
                    result_idx += coord * multiplier;
                    multiplier *= shape[d];
                }
            }

            result[result_idx] = result[result_idx].max(val);
        }

        Ok(CpuArray::from_ndarray(
            ArrayD::from_shape_vec(IxDyn(&new_shape), result).unwrap(),
        ))
    }

    fn cumsum(arr: &CpuArray) -> CpuArray {
        let data = arr.as_f64_slice();
        let mut result = Vec::with_capacity(data.len());
        let mut acc = 0.0;
        for &x in &data {
            acc += x;
            result.push(acc);
        }
        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[data.len()]), result).unwrap())
    }

    fn cumprod(arr: &CpuArray) -> CpuArray {
        let data = arr.as_f64_slice();
        let mut result = Vec::with_capacity(data.len());
        let mut acc = 1.0;
        for &x in &data {
            acc *= x;
            result.push(acc);
        }
        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[data.len()]), result).unwrap())
    }

    fn all(arr: &CpuArray) -> bool {
        arr.as_ndarray().iter().all(|&x| x != 0.0)
    }

    fn any(arr: &CpuArray) -> bool {
        arr.as_ndarray().iter().any(|&x| x != 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(data: Vec<f64>) -> CpuArray {
        CpuArray::from_f64_vec(data.clone(), vec![data.len()]).unwrap()
    }

    fn mat(data: Vec<f64>, rows: usize, cols: usize) -> CpuArray {
        CpuArray::from_f64_vec(data, vec![rows, cols]).unwrap()
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn test_sum() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(CpuBackend::sum(&a), 15.0);
    }

    #[test]
    fn test_prod() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(CpuBackend::prod(&a), 24.0);
    }

    #[test]
    fn test_mean() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(CpuBackend::mean(&a), 3.0);
    }

    #[test]
    fn test_var() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(approx_eq(CpuBackend::var(&a), 2.0));
    }

    #[test]
    fn test_std() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(approx_eq(CpuBackend::std(&a), 2.0_f64.sqrt()));
    }

    #[test]
    fn test_min_max() {
        let a = arr(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        assert_eq!(CpuBackend::min(&a), 1.0);
        assert_eq!(CpuBackend::max(&a), 9.0);
    }

    #[test]
    fn test_argmin_argmax() {
        let a = arr(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        assert_eq!(CpuBackend::argmin(&a), 1);
        assert_eq!(CpuBackend::argmax(&a), 5);
    }

    #[test]
    fn test_sum_axis() {
        // [[1, 2, 3], [4, 5, 6]]
        let m = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);

        // Sum along axis 0 (rows) -> [5, 7, 9]
        let result = CpuBackend::sum_axis(&m, 0).unwrap();
        assert_eq!(result.as_f64_slice(), vec![5.0, 7.0, 9.0]);

        // Sum along axis 1 (cols) -> [6, 15]
        let result = CpuBackend::sum_axis(&m, 1).unwrap();
        assert_eq!(result.as_f64_slice(), vec![6.0, 15.0]);
    }

    #[test]
    fn test_cumsum() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0]);
        let result = CpuBackend::cumsum(&a);
        assert_eq!(result.as_f64_slice(), vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_cumprod() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0]);
        let result = CpuBackend::cumprod(&a);
        assert_eq!(result.as_f64_slice(), vec![1.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn test_all_any() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        assert!(CpuBackend::all(&a));
        assert!(CpuBackend::any(&a));

        let b = arr(vec![0.0, 0.0, 0.0]);
        assert!(!CpuBackend::all(&b));
        assert!(!CpuBackend::any(&b));

        let c = arr(vec![1.0, 0.0, 1.0]);
        assert!(!CpuBackend::all(&c));
        assert!(CpuBackend::any(&c));
    }
}
