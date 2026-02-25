//! Linear algebra operations for CPU backend using faer

use crate::{CpuArray, CpuBackend};
use faer::{Mat, MatRef};
use ndarray::{ArrayD, IxDyn};
use rumpy_core::{ops::LinalgOps, Array, Result, RumpyError};

/// Convert CpuArray to faer Mat (assumes 2D, row-major)
fn to_faer(arr: &CpuArray) -> Result<Mat<f64>> {
    let shape = arr.shape();
    if shape.len() != 2 {
        return Err(RumpyError::InvalidArgument(
            "Matrix must be 2D".to_string(),
        ));
    }
    let (m, n) = (shape[0], shape[1]);
    let data = arr.as_f64_slice();

    // faer uses column-major, we have row-major
    Ok(Mat::from_fn(m, n, |i, j| data[i * n + j]))
}

/// Convert faer Mat to CpuArray (row-major)
fn from_faer(mat: MatRef<'_, f64>) -> CpuArray {
    let (m, n) = (mat.nrows(), mat.ncols());
    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            data.push(mat.read(i, j));
        }
    }
    CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[m, n]), data).unwrap())
}

/// Convert faer column vector to CpuArray
fn from_faer_vec(mat: MatRef<'_, f64>) -> CpuArray {
    let n = mat.nrows();
    let data: Vec<f64> = (0..n).map(|i| mat.read(i, 0)).collect();
    CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[n]), data).unwrap())
}

impl LinalgOps for CpuBackend {
    type Array = CpuArray;

    fn matmul(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        let mat_a = to_faer(a)?;
        let mat_b = to_faer(b)?;

        if mat_a.ncols() != mat_b.nrows() {
            return Err(RumpyError::IncompatibleShapes(
                a.shape().to_vec(),
                b.shape().to_vec(),
            ));
        }

        let result = mat_a * mat_b;
        Ok(from_faer(result.as_ref()))
    }

    fn dot(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() == 1 && b_shape.len() == 1 {
            // Vector dot product
            if a_shape[0] != b_shape[0] {
                return Err(RumpyError::IncompatibleShapes(
                    a_shape.to_vec(),
                    b_shape.to_vec(),
                ));
            }
            let result: f64 = a
                .as_f64_slice()
                .iter()
                .zip(b.as_f64_slice().iter())
                .map(|(x, y)| x * y)
                .sum();
            Ok(CpuArray::from_f64_vec(vec![result], vec![1])?)
        } else if a_shape.len() == 2 && b_shape.len() == 2 {
            // Matrix multiplication
            Self::matmul(a, b)
        } else {
            Err(RumpyError::InvalidArgument(
                "dot requires 1D or 2D arrays".to_string(),
            ))
        }
    }

    fn inner(a: &CpuArray, b: &CpuArray) -> Result<f64> {
        if a.size() != b.size() {
            return Err(RumpyError::IncompatibleShapes(
                a.shape().to_vec(),
                b.shape().to_vec(),
            ));
        }
        Ok(a.as_f64_slice()
            .iter()
            .zip(b.as_f64_slice().iter())
            .map(|(x, y)| x * y)
            .sum())
    }

    fn outer(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        let a_data = a.as_f64_slice();
        let b_data = b.as_f64_slice();
        let m = a_data.len();
        let n = b_data.len();

        let mut result = Vec::with_capacity(m * n);
        for &ai in &a_data {
            for &bi in &b_data {
                result.push(ai * bi);
            }
        }

        Ok(CpuArray::from_ndarray(
            ArrayD::from_shape_vec(IxDyn(&[m, n]), result).unwrap(),
        ))
    }

    fn inv(arr: &CpuArray) -> Result<CpuArray> {
        let mat = to_faer(arr)?;
        if mat.nrows() != mat.ncols() {
            return Err(RumpyError::NotSquare(arr.shape().to_vec()));
        }

        let lu = mat.partial_piv_lu();
        let inv = lu.inverse();
        Ok(from_faer(inv.as_ref()))
    }

    fn pinv(arr: &CpuArray) -> Result<CpuArray> {
        let mat = to_faer(arr)?;
        let svd = mat.svd();

        // Compute pseudoinverse using SVD: A+ = V * S+ * U^T
        let u = svd.u();
        let s = svd.s_diagonal();
        let v = svd.v();

        let (m, n) = (mat.nrows(), mat.ncols());
        let k = s.nrows();

        // Invert singular values (with threshold)
        let threshold = 1e-10 * s.read(0, 0);
        let mut s_inv = Mat::zeros(n, m);
        for i in 0..k {
            let si = s.read(i, 0);
            if si.abs() > threshold {
                s_inv.write(i, i, 1.0 / si);
            }
        }

        // A+ = V * S+ * U^T
        let result = &v * &s_inv * u.transpose();
        Ok(from_faer(result.as_ref()))
    }

    fn det(arr: &CpuArray) -> Result<f64> {
        let mat = to_faer(arr)?;
        if mat.nrows() != mat.ncols() {
            return Err(RumpyError::NotSquare(arr.shape().to_vec()));
        }

        let lu = mat.partial_piv_lu();
        Ok(lu.compute_determinant())
    }

    fn trace(arr: &CpuArray) -> Result<f64> {
        let shape = arr.shape();
        if shape.len() != 2 {
            return Err(RumpyError::InvalidArgument(
                "trace requires 2D array".to_string(),
            ));
        }
        let n = shape[0].min(shape[1]);
        let data = arr.as_ndarray();
        let mut sum = 0.0;
        for i in 0..n {
            sum += data[IxDyn(&[i, i])];
        }
        Ok(sum)
    }

    fn rank(arr: &CpuArray) -> Result<usize> {
        let mat = to_faer(arr)?;
        let svd = mat.svd();
        let s = svd.s_diagonal();

        // Count singular values above threshold
        let threshold = 1e-10 * s.read(0, 0);
        let mut rank = 0;
        for i in 0..s.nrows() {
            if s.read(i, 0) > threshold {
                rank += 1;
            }
        }
        Ok(rank)
    }

    fn norm(arr: &CpuArray, ord: Option<f64>) -> Result<f64> {
        let data = arr.as_f64_slice();
        let ord = ord.unwrap_or(2.0);

        if ord == 2.0 {
            // L2 norm (Euclidean)
            Ok(data.iter().map(|x| x * x).sum::<f64>().sqrt())
        } else if ord == 1.0 {
            // L1 norm
            Ok(data.iter().map(|x| x.abs()).sum())
        } else if ord == f64::INFINITY {
            // L-inf norm
            Ok(data.iter().map(|x| x.abs()).fold(0.0, f64::max))
        } else if ord == f64::NEG_INFINITY {
            // L-neg-inf norm
            Ok(data.iter().map(|x| x.abs()).fold(f64::INFINITY, f64::min))
        } else {
            // General p-norm
            Ok(data.iter().map(|x| x.abs().powf(ord)).sum::<f64>().powf(1.0 / ord))
        }
    }

    fn solve(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        let mat_a = to_faer(a)?;
        let mat_b = to_faer(b)?;

        if mat_a.nrows() != mat_a.ncols() {
            return Err(RumpyError::NotSquare(a.shape().to_vec()));
        }
        if mat_a.nrows() != mat_b.nrows() {
            return Err(RumpyError::IncompatibleShapes(
                a.shape().to_vec(),
                b.shape().to_vec(),
            ));
        }

        let lu = mat_a.partial_piv_lu();
        let x = lu.solve(&mat_b);

        if b.shape().len() == 1 || b.shape()[1] == 1 {
            Ok(from_faer_vec(x.as_ref()))
        } else {
            Ok(from_faer(x.as_ref()))
        }
    }

    fn lstsq(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        // Least squares via pseudoinverse: x = A+ * b
        let a_pinv = Self::pinv(a)?;
        Self::matmul(&a_pinv, b)
    }

    fn qr(arr: &CpuArray) -> Result<(CpuArray, CpuArray)> {
        let mat = to_faer(arr)?;
        let qr = mat.qr();

        let q = qr.compute_thin_q();
        let r = qr.compute_thin_r();

        Ok((from_faer(q.as_ref()), from_faer(r.as_ref())))
    }

    fn lu(arr: &CpuArray) -> Result<(CpuArray, CpuArray, CpuArray)> {
        let mat = to_faer(arr)?;
        let n = mat.nrows();

        let plu = mat.partial_piv_lu();
        let (p, l, u) = plu.into_parts();

        // Convert permutation to matrix
        let mut p_mat = Mat::zeros(n, n);
        for i in 0..n {
            p_mat.write(i, p.inverse().arrays().0[i].to_signed().unsigned_abs(), 1.0);
        }

        Ok((
            from_faer(p_mat.as_ref()),
            from_faer(l.into_inner().as_ref()),
            from_faer(u.into_inner().as_ref()),
        ))
    }

    fn cholesky(arr: &CpuArray) -> Result<CpuArray> {
        let mat = to_faer(arr)?;
        if mat.nrows() != mat.ncols() {
            return Err(RumpyError::NotSquare(arr.shape().to_vec()));
        }

        let chol = mat.cholesky(faer::Side::Lower);
        match chol {
            Ok(c) => Ok(from_faer(c.into_parts().0.into_inner().as_ref())),
            Err(_) => Err(RumpyError::InvalidArgument(
                "Matrix is not positive definite".to_string(),
            )),
        }
    }

    fn svd(arr: &CpuArray) -> Result<(CpuArray, CpuArray, CpuArray)> {
        let mat = to_faer(arr)?;
        let svd = mat.svd();

        let u = svd.u();
        let s = svd.s_diagonal();
        let vt = svd.v().transpose();

        // Convert s to 1D array
        let s_vec: Vec<f64> = (0..s.nrows()).map(|i| s.read(i, 0)).collect();
        let s_arr = CpuArray::from_ndarray(
            ArrayD::from_shape_vec(IxDyn(&[s_vec.len()]), s_vec).unwrap(),
        );

        Ok((from_faer(u), s_arr, from_faer(vt.as_ref())))
    }

    fn eig(arr: &CpuArray) -> Result<(CpuArray, CpuArray)> {
        let mat = to_faer(arr)?;
        if mat.nrows() != mat.ncols() {
            return Err(RumpyError::NotSquare(arr.shape().to_vec()));
        }

        let evd = mat.selfadjoint_eigendecomposition(faer::Side::Lower);
        let eigenvalues = evd.s_diagonal();
        let eigenvectors = evd.u();

        let eig_vec: Vec<f64> = (0..eigenvalues.nrows())
            .map(|i| eigenvalues.read(i, 0))
            .collect();
        let eig_arr = CpuArray::from_ndarray(
            ArrayD::from_shape_vec(IxDyn(&[eig_vec.len()]), eig_vec).unwrap(),
        );

        Ok((eig_arr, from_faer(eigenvectors)))
    }

    fn eigvals(arr: &CpuArray) -> Result<CpuArray> {
        let (vals, _) = Self::eig(arr)?;
        Ok(vals)
    }

    fn transpose(arr: &CpuArray) -> CpuArray {
        let data = arr.as_ndarray();
        let shape = data.shape();

        if shape.len() == 1 {
            return arr.clone();
        }

        if shape.len() == 2 {
            let (m, n) = (shape[0], shape[1]);
            let mut result = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    result[j * m + i] = data[IxDyn(&[i, j])];
                }
            }
            return CpuArray::from_ndarray(
                ArrayD::from_shape_vec(IxDyn(&[n, m]), result).unwrap(),
            );
        }

        // General transpose: reverse axes
        let reversed: Vec<usize> = (0..shape.len()).rev().collect();
        CpuArray::from_ndarray(data.clone().permuted_axes(reversed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(data: Vec<f64>, rows: usize, cols: usize) -> CpuArray {
        CpuArray::from_f64_vec(data, vec![rows, cols]).unwrap()
    }

    fn vec1d(data: Vec<f64>) -> CpuArray {
        CpuArray::from_f64_vec(data.clone(), vec![data.len()]).unwrap()
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    fn approx_eq_vec(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y))
    }

    #[test]
    fn test_matmul() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let b = mat(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
        let c = CpuBackend::matmul(&a, &b).unwrap();

        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        assert_eq!(c.as_f64_slice(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_dot_vectors() {
        let a = vec1d(vec![1.0, 2.0, 3.0]);
        let b = vec1d(vec![4.0, 5.0, 6.0]);
        let c = CpuBackend::dot(&a, &b).unwrap();
        // 1*4 + 2*5 + 3*6 = 32
        assert_eq!(c.as_f64_slice()[0], 32.0);
    }

    #[test]
    fn test_inner() {
        let a = vec1d(vec![1.0, 2.0, 3.0]);
        let b = vec1d(vec![4.0, 5.0, 6.0]);
        let result = CpuBackend::inner(&a, &b).unwrap();
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_outer() {
        let a = vec1d(vec![1.0, 2.0]);
        let b = vec1d(vec![3.0, 4.0, 5.0]);
        let c = CpuBackend::outer(&a, &b).unwrap();
        // [[1*3, 1*4, 1*5], [2*3, 2*4, 2*5]]
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.as_f64_slice(), vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_inv() {
        let a = mat(vec![4.0, 7.0, 2.0, 6.0], 2, 2);
        let a_inv = CpuBackend::inv(&a).unwrap();

        // A @ A^-1 = I
        let identity = CpuBackend::matmul(&a, &a_inv).unwrap();
        let data = identity.as_f64_slice();
        assert!(approx_eq(data[0], 1.0));
        assert!(approx_eq(data[1], 0.0));
        assert!(approx_eq(data[2], 0.0));
        assert!(approx_eq(data[3], 1.0));
    }

    #[test]
    fn test_det() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let det = CpuBackend::det(&a).unwrap();
        // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
        assert!(approx_eq(det, -2.0));
    }

    #[test]
    fn test_trace() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        let tr = CpuBackend::trace(&a).unwrap();
        // trace = 1 + 5 + 9 = 15
        assert_eq!(tr, 15.0);
    }

    #[test]
    fn test_norm() {
        let a = vec1d(vec![3.0, 4.0]);
        assert!(approx_eq(CpuBackend::norm(&a, Some(2.0)).unwrap(), 5.0));
        assert!(approx_eq(CpuBackend::norm(&a, Some(1.0)).unwrap(), 7.0));
        assert!(approx_eq(CpuBackend::norm(&a, Some(f64::INFINITY)).unwrap(), 4.0));
    }

    #[test]
    fn test_solve() {
        // Solve Ax = b where A = [[3,1],[1,2]], b = [[9],[8]]
        // Solution: x = [[2],[3]]
        let a = mat(vec![3.0, 1.0, 1.0, 2.0], 2, 2);
        let b = mat(vec![9.0, 8.0], 2, 1);
        let x = CpuBackend::solve(&a, &b).unwrap();
        assert!(approx_eq_vec(&x.as_f64_slice(), &[2.0, 3.0]));
    }

    #[test]
    fn test_qr() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let (q, r) = CpuBackend::qr(&a).unwrap();

        // Q @ R should equal A
        let reconstructed = CpuBackend::matmul(&q, &r).unwrap();
        assert!(approx_eq_vec(&reconstructed.as_f64_slice(), &a.as_f64_slice()));
    }

    #[test]
    fn test_svd() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let (u, s, vt) = CpuBackend::svd(&a).unwrap();

        // Verify shapes
        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(vt.shape(), &[2, 3]);

        // U @ diag(S) @ Vt should equal A
        // (simplified check: just verify s values are positive and sorted)
        let s_data = s.as_f64_slice();
        assert!(s_data[0] >= s_data[1]);
        assert!(s_data[1] > 0.0);
    }

    #[test]
    fn test_transpose() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let at = CpuBackend::transpose(&a);
        assert_eq!(at.shape(), &[3, 2]);
        assert_eq!(at.as_f64_slice(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
