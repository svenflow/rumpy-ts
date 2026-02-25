//! Element-wise math operations for CPU backend

use crate::{CpuArray, CpuBackend};
use rumpy_core::{ops::MathOps, Array, Result, RumpyError};

macro_rules! impl_unary_op {
    ($name:ident, $op:expr) => {
        fn $name(arr: &CpuArray) -> CpuArray {
            CpuArray::from_ndarray(arr.as_ndarray().mapv($op))
        }
    };
}

macro_rules! impl_binary_op {
    ($name:ident, $op:tt) => {
        fn $name(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
            if a.shape() != b.shape() {
                return Err(RumpyError::IncompatibleShapes(
                    a.shape().to_vec(),
                    b.shape().to_vec(),
                ));
            }
            Ok(CpuArray::from_ndarray(a.as_ndarray() $op b.as_ndarray()))
        }
    };
}

macro_rules! impl_scalar_op {
    ($name:ident, $op:tt) => {
        fn $name(arr: &CpuArray, scalar: f64) -> CpuArray {
            CpuArray::from_ndarray(arr.as_ndarray().mapv(|x| x $op scalar))
        }
    };
}

impl MathOps for CpuBackend {
    type Array = CpuArray;

    // Trigonometric
    impl_unary_op!(sin, |x: f64| x.sin());
    impl_unary_op!(cos, |x: f64| x.cos());
    impl_unary_op!(tan, |x: f64| x.tan());
    impl_unary_op!(arcsin, |x: f64| x.asin());
    impl_unary_op!(arccos, |x: f64| x.acos());
    impl_unary_op!(arctan, |x: f64| x.atan());

    // Hyperbolic
    impl_unary_op!(sinh, |x: f64| x.sinh());
    impl_unary_op!(cosh, |x: f64| x.cosh());
    impl_unary_op!(tanh, |x: f64| x.tanh());

    // Exponential and logarithmic
    impl_unary_op!(exp, |x: f64| x.exp());
    impl_unary_op!(exp2, |x: f64| x.exp2());
    impl_unary_op!(log, |x: f64| x.ln());
    impl_unary_op!(log2, |x: f64| x.log2());
    impl_unary_op!(log10, |x: f64| x.log10());

    // Power and roots
    impl_unary_op!(sqrt, |x: f64| x.sqrt());
    impl_unary_op!(cbrt, |x: f64| x.cbrt());
    impl_unary_op!(square, |x: f64| x * x);

    // Rounding
    impl_unary_op!(floor, |x: f64| x.floor());
    impl_unary_op!(ceil, |x: f64| x.ceil());
    impl_unary_op!(round, |x: f64| x.round());

    // Other unary
    impl_unary_op!(abs, |x: f64| x.abs());
    impl_unary_op!(neg, |x: f64| -x);
    impl_unary_op!(reciprocal, |x: f64| 1.0 / x);

    fn sign(arr: &CpuArray) -> CpuArray {
        CpuArray::from_ndarray(arr.as_ndarray().mapv(|x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }))
    }

    // Binary operations
    impl_binary_op!(add, +);
    impl_binary_op!(sub, -);
    impl_binary_op!(mul, *);
    impl_binary_op!(div, /);

    fn pow(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        if a.shape() != b.shape() {
            return Err(RumpyError::IncompatibleShapes(
                a.shape().to_vec(),
                b.shape().to_vec(),
            ));
        }
        let a_data = a.as_ndarray();
        let b_data = b.as_ndarray();
        let result = ndarray::Zip::from(a_data)
            .and(b_data)
            .map_collect(|&x, &y| x.powf(y));
        Ok(CpuArray::from_ndarray(result))
    }

    fn maximum(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        if a.shape() != b.shape() {
            return Err(RumpyError::IncompatibleShapes(
                a.shape().to_vec(),
                b.shape().to_vec(),
            ));
        }
        let a_data = a.as_ndarray();
        let b_data = b.as_ndarray();
        let result = ndarray::Zip::from(a_data)
            .and(b_data)
            .map_collect(|&x, &y| x.max(y));
        Ok(CpuArray::from_ndarray(result))
    }

    fn minimum(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        if a.shape() != b.shape() {
            return Err(RumpyError::IncompatibleShapes(
                a.shape().to_vec(),
                b.shape().to_vec(),
            ));
        }
        let a_data = a.as_ndarray();
        let b_data = b.as_ndarray();
        let result = ndarray::Zip::from(a_data)
            .and(b_data)
            .map_collect(|&x, &y| x.min(y));
        Ok(CpuArray::from_ndarray(result))
    }

    // Scalar operations
    impl_scalar_op!(add_scalar, +);
    impl_scalar_op!(sub_scalar, -);
    impl_scalar_op!(mul_scalar, *);
    impl_scalar_op!(div_scalar, /);

    fn pow_scalar(arr: &CpuArray, scalar: f64) -> CpuArray {
        CpuArray::from_ndarray(arr.as_ndarray().mapv(|x| x.powf(scalar)))
    }

    fn clip(arr: &CpuArray, min: f64, max: f64) -> CpuArray {
        CpuArray::from_ndarray(arr.as_ndarray().mapv(|x| x.clamp(min, max)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    fn arr(data: Vec<f64>) -> CpuArray {
        CpuArray::from_f64_vec(data.clone(), vec![data.len()]).unwrap()
    }

    #[test]
    fn test_sin() {
        let a = arr(vec![0.0, PI / 2.0, PI]);
        let result = CpuBackend::sin(&a);
        let data = result.as_f64_slice();
        assert!(approx_eq(data[0], 0.0));
        assert!(approx_eq(data[1], 1.0));
        assert!(approx_eq(data[2], 0.0));
    }

    #[test]
    fn test_cos() {
        let a = arr(vec![0.0, PI / 2.0, PI]);
        let result = CpuBackend::cos(&a);
        let data = result.as_f64_slice();
        assert!(approx_eq(data[0], 1.0));
        assert!(approx_eq(data[1], 0.0));
        assert!(approx_eq(data[2], -1.0));
    }

    #[test]
    fn test_exp_log() {
        let a = arr(vec![0.0, 1.0, 2.0]);
        let exp_a = CpuBackend::exp(&a);
        let log_exp_a = CpuBackend::log(&exp_a);
        for (x, y) in a.as_f64_slice().iter().zip(log_exp_a.as_f64_slice().iter()) {
            assert!(approx_eq(*x, *y));
        }
    }

    #[test]
    fn test_sqrt() {
        let a = arr(vec![0.0, 1.0, 4.0, 9.0, 16.0]);
        let result = CpuBackend::sqrt(&a);
        assert_eq!(result.as_f64_slice(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_abs() {
        let a = arr(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = CpuBackend::abs(&a);
        assert_eq!(result.as_f64_slice(), vec![2.0, 1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sign() {
        let a = arr(vec![-2.0, -0.5, 0.0, 0.5, 2.0]);
        let result = CpuBackend::sign(&a);
        assert_eq!(result.as_f64_slice(), vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_add() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![4.0, 5.0, 6.0]);
        let result = CpuBackend::add(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub() {
        let a = arr(vec![5.0, 7.0, 9.0]);
        let b = arr(vec![1.0, 2.0, 3.0]);
        let result = CpuBackend::sub(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_mul() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![2.0, 3.0, 4.0]);
        let result = CpuBackend::mul(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_div() {
        let a = arr(vec![4.0, 9.0, 16.0]);
        let b = arr(vec![2.0, 3.0, 4.0]);
        let result = CpuBackend::div(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_pow() {
        let a = arr(vec![2.0, 3.0, 4.0]);
        let b = arr(vec![2.0, 2.0, 2.0]);
        let result = CpuBackend::pow(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_add_scalar() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let result = CpuBackend::add_scalar(&a, 10.0);
        assert_eq!(result.as_f64_slice(), vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_clip() {
        let a = arr(vec![-5.0, 0.0, 5.0, 10.0, 15.0]);
        let result = CpuBackend::clip(&a, 0.0, 10.0);
        assert_eq!(result.as_f64_slice(), vec![0.0, 0.0, 5.0, 10.0, 10.0]);
    }

    #[test]
    fn test_maximum() {
        let a = arr(vec![1.0, 5.0, 3.0]);
        let b = arr(vec![2.0, 3.0, 4.0]);
        let result = CpuBackend::maximum(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![2.0, 5.0, 4.0]);
    }

    #[test]
    fn test_minimum() {
        let a = arr(vec![1.0, 5.0, 3.0]);
        let b = arr(vec![2.0, 3.0, 4.0]);
        let result = CpuBackend::minimum(&a, &b).unwrap();
        assert_eq!(result.as_f64_slice(), vec![1.0, 3.0, 3.0]);
    }

    #[test]
    fn test_shape_mismatch() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![1.0, 2.0]);
        let result = CpuBackend::add(&a, &b);
        assert!(result.is_err());
    }
}
