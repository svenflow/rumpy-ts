//! Comparison operations for CPU backend

use crate::{CpuArray, CpuBackend};
use rumpy_core::{ops::CompareOps, Array, Result, RumpyError};

macro_rules! impl_compare_op {
    ($name:ident, $op:tt) => {
        fn $name(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
            if a.shape() != b.shape() {
                return Err(RumpyError::IncompatibleShapes(
                    a.shape().to_vec(),
                    b.shape().to_vec(),
                ));
            }
            let result: Vec<f64> = a
                .as_f64_slice()
                .iter()
                .zip(b.as_f64_slice().iter())
                .map(|(&x, &y)| if x $op y { 1.0 } else { 0.0 })
                .collect();
            Ok(CpuArray::from_f64_vec(result, a.shape().to_vec())?)
        }
    };
}

macro_rules! impl_compare_scalar_op {
    ($name:ident, $op:tt) => {
        fn $name(arr: &CpuArray, scalar: f64) -> CpuArray {
            let result: Vec<f64> = arr
                .as_f64_slice()
                .iter()
                .map(|&x| if x $op scalar { 1.0 } else { 0.0 })
                .collect();
            CpuArray::from_f64_vec(result, arr.shape().to_vec()).unwrap()
        }
    };
}

impl CompareOps for CpuBackend {
    type Array = CpuArray;

    impl_compare_op!(eq, ==);
    impl_compare_op!(ne, !=);
    impl_compare_op!(lt, <);
    impl_compare_op!(le, <=);
    impl_compare_op!(gt, >);
    impl_compare_op!(ge, >=);

    impl_compare_scalar_op!(eq_scalar, ==);
    impl_compare_scalar_op!(ne_scalar, !=);
    impl_compare_scalar_op!(lt_scalar, <);
    impl_compare_scalar_op!(le_scalar, <=);
    impl_compare_scalar_op!(gt_scalar, >);
    impl_compare_scalar_op!(ge_scalar, >=);

    fn isnan(arr: &CpuArray) -> CpuArray {
        let result: Vec<f64> = arr
            .as_f64_slice()
            .iter()
            .map(|&x| if x.is_nan() { 1.0 } else { 0.0 })
            .collect();
        CpuArray::from_f64_vec(result, arr.shape().to_vec()).unwrap()
    }

    fn isinf(arr: &CpuArray) -> CpuArray {
        let result: Vec<f64> = arr
            .as_f64_slice()
            .iter()
            .map(|&x| if x.is_infinite() { 1.0 } else { 0.0 })
            .collect();
        CpuArray::from_f64_vec(result, arr.shape().to_vec()).unwrap()
    }

    fn isfinite(arr: &CpuArray) -> CpuArray {
        let result: Vec<f64> = arr
            .as_f64_slice()
            .iter()
            .map(|&x| if x.is_finite() { 1.0 } else { 0.0 })
            .collect();
        CpuArray::from_f64_vec(result, arr.shape().to_vec()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(data: Vec<f64>) -> CpuArray {
        CpuArray::from_f64_vec(data.clone(), vec![data.len()]).unwrap()
    }

    #[test]
    fn test_eq() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![1.0, 5.0, 3.0]);
        let result = CpuBackend::eq(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_ne() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![1.0, 5.0, 3.0]);
        let result = CpuBackend::ne(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_lt() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![2.0, 2.0, 2.0]);
        let result = CpuBackend::lt(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_le() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![2.0, 2.0, 2.0]);
        let result = CpuBackend::le(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_gt() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![2.0, 2.0, 2.0]);
        let result = CpuBackend::gt(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_ge() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![2.0, 2.0, 2.0]);
        let result = CpuBackend::ge(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_eq_scalar() {
        let a = arr(vec![1.0, 2.0, 3.0, 2.0]);
        let result = CpuBackend::eq_scalar(&a, 2.0);
        assert_eq!(result.as_f64_slice(), vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_lt_scalar() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0]);
        let result = CpuBackend::lt_scalar(&a, 3.0);
        assert_eq!(result.as_f64_slice(), vec![1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_isnan() {
        let a = arr(vec![1.0, f64::NAN, 3.0, f64::NAN]);
        let result = CpuBackend::isnan(&a);
        assert_eq!(result.as_f64_slice(), vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_isinf() {
        let a = arr(vec![1.0, f64::INFINITY, 3.0, f64::NEG_INFINITY]);
        let result = CpuBackend::isinf(&a);
        assert_eq!(result.as_f64_slice(), vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_isfinite() {
        let a = arr(vec![1.0, f64::INFINITY, f64::NAN, 4.0]);
        let result = CpuBackend::isfinite(&a);
        assert_eq!(result.as_f64_slice(), vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_shape_mismatch() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![1.0, 2.0]);
        let result = CpuBackend::eq(&a, &b);
        assert!(result.is_err());
    }
}
