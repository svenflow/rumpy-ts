//! Linear algebra operations for CPU backend using faer
//! On WASM targets with simd128, uses hand-optimized SIMD GEMM kernels.

use crate::{simd_gemm, CpuArray, CpuBackend};
use faer::Mat;
use ndarray::{ArrayD, IxDyn};
use rumpy_core::{ops::LinalgOps, Array, Result, RumpyError};

/// Convert CpuArray to faer Mat (assumes 2D, row-major)
fn to_faer(arr: &CpuArray) -> Result<Mat<f64>> {
    let shape = arr.shape();
    if shape.len() != 2 {
        return Err(RumpyError::InvalidArgument("Matrix must be 2D".to_string()));
    }
    let (m, n) = (shape[0], shape[1]);
    let data = arr.as_f64_slice();

    // faer uses column-major, we have row-major
    Ok(Mat::from_fn(m, n, |i, j| data[i * n + j]))
}

/// Convert faer Mat to CpuArray (row-major)
fn from_faer(mat: &Mat<f64>) -> CpuArray {
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
fn from_faer_vec(mat: &Mat<f64>) -> CpuArray {
    let n = mat.nrows();
    let data: Vec<f64> = (0..n).map(|i| mat.read(i, 0)).collect();
    CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[n]), data).unwrap())
}

impl LinalgOps for CpuBackend {
    type Array = CpuArray;

    fn matmul(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(RumpyError::InvalidArgument("Matrix must be 2D".to_string()));
        }

        let (m, k1) = (a_shape[0], a_shape[1]);
        let (k2, n) = (b_shape[0], b_shape[1]);

        if k1 != k2 {
            return Err(RumpyError::IncompatibleShapes(
                a.shape().to_vec(),
                b.shape().to_vec(),
            ));
        }

        let k = k1;

        // Use SIMD-optimized GEMM on WASM (simd128 is enabled via .cargo/config.toml rustflags)
        // Fall back to faer for native targets
        #[cfg(target_arch = "wasm32")]
        {
            let a_data = a.as_f64_slice();
            let b_data = b.as_f64_slice();
            let c_data = simd_gemm::matmul_dispatch_f64(&a_data, &b_data, m, n, k);
            Ok(CpuArray::from_ndarray(
                ArrayD::from_shape_vec(IxDyn(&[m, n]), c_data).unwrap(),
            ))
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let mat_a = to_faer(a)?;
            let mat_b = to_faer(b)?;
            let result = &mat_a * &mat_b;
            Ok(from_faer(&result))
        }
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

        // Use Gauss-Jordan elimination for inverse
        let n = mat.nrows();
        let mut aug = Mat::zeros(n, 2 * n);

        // Copy matrix to left half, identity to right half
        for i in 0..n {
            for j in 0..n {
                aug.write(i, j, mat.read(i, j));
                aug.write(i, n + j, if i == j { 1.0 } else { 0.0 });
            }
        }

        // Gauss-Jordan elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug.read(k, i).abs() > aug.read(max_row, i).abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..(2 * n) {
                    let tmp = aug.read(i, j);
                    aug.write(i, j, aug.read(max_row, j));
                    aug.write(max_row, j, tmp);
                }
            }

            let pivot = aug.read(i, i);
            if pivot.abs() < 1e-14 {
                return Err(RumpyError::SingularMatrix);
            }

            // Scale pivot row
            for j in 0..(2 * n) {
                aug.write(i, j, aug.read(i, j) / pivot);
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = aug.read(k, i);
                    for j in 0..(2 * n) {
                        aug.write(k, j, aug.read(k, j) - factor * aug.read(i, j));
                    }
                }
            }
        }

        // Extract inverse from right half
        let mut inv = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                inv.write(i, j, aug.read(i, n + j));
            }
        }

        Ok(from_faer(&inv))
    }

    fn pinv(arr: &CpuArray) -> Result<CpuArray> {
        // Use SVD-based pseudoinverse
        // For simplicity, use the formula: A+ = (A^T A)^-1 A^T for full column rank
        // This is a simplified implementation
        let mat = to_faer(arr)?;
        let mt = mat.transpose();
        let mta = mt * &mat;

        // Solve (A^T A) X = A^T to get X = A+
        let mta_inv = {
            let arr_tmp = from_faer(&mta);
            Self::inv(&arr_tmp)?
        };
        let mta_inv_mat = to_faer(&mta_inv)?;
        let result = &mta_inv_mat * mt;
        Ok(from_faer(&result))
    }

    fn det(arr: &CpuArray) -> Result<f64> {
        let mat = to_faer(arr)?;
        let n = mat.nrows();
        if n != mat.ncols() {
            return Err(RumpyError::NotSquare(arr.shape().to_vec()));
        }

        // LU decomposition for determinant
        let mut work = mat.clone();
        let mut det = 1.0;
        let mut swaps = 0;

        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if work.read(k, i).abs() > work.read(max_row, i).abs() {
                    max_row = k;
                }
            }

            if max_row != i {
                for j in 0..n {
                    let tmp = work.read(i, j);
                    work.write(i, j, work.read(max_row, j));
                    work.write(max_row, j, tmp);
                }
                swaps += 1;
            }

            let pivot = work.read(i, i);
            if pivot.abs() < 1e-14 {
                return Ok(0.0);
            }

            det *= pivot;

            for k in (i + 1)..n {
                let factor = work.read(k, i) / pivot;
                for j in i..n {
                    work.write(k, j, work.read(k, j) - factor * work.read(i, j));
                }
            }
        }

        if swaps % 2 == 1 {
            det = -det;
        }

        Ok(det)
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
        // Simple rank estimation via row echelon form
        let mat = to_faer(arr)?;
        let (m, n) = (mat.nrows(), mat.ncols());
        let mut work = mat.clone();

        let mut rank = 0;
        let mut col = 0;

        for row in 0..m {
            if col >= n {
                break;
            }

            // Find pivot
            let mut max_row = row;
            for k in (row + 1)..m {
                if work.read(k, col).abs() > work.read(max_row, col).abs() {
                    max_row = k;
                }
            }

            if work.read(max_row, col).abs() < 1e-10 {
                col += 1;
                continue;
            }

            // Swap rows
            if max_row != row {
                for j in 0..n {
                    let tmp = work.read(row, j);
                    work.write(row, j, work.read(max_row, j));
                    work.write(max_row, j, tmp);
                }
            }

            // Eliminate
            let pivot = work.read(row, col);
            for k in (row + 1)..m {
                let factor = work.read(k, col) / pivot;
                for j in col..n {
                    work.write(k, j, work.read(k, j) - factor * work.read(row, j));
                }
            }

            rank += 1;
            col += 1;
        }

        Ok(rank)
    }

    fn norm(arr: &CpuArray, ord: Option<f64>) -> Result<f64> {
        let data = arr.as_f64_slice();
        let ord = ord.unwrap_or(2.0);

        if ord == 2.0 {
            Ok(data.iter().map(|x| x * x).sum::<f64>().sqrt())
        } else if ord == 1.0 {
            Ok(data.iter().map(|x| x.abs()).sum())
        } else if ord == f64::INFINITY {
            Ok(data.iter().map(|x| x.abs()).fold(0.0, f64::max))
        } else if ord == f64::NEG_INFINITY {
            Ok(data.iter().map(|x| x.abs()).fold(f64::INFINITY, f64::min))
        } else {
            Ok(data
                .iter()
                .map(|x| x.abs().powf(ord))
                .sum::<f64>()
                .powf(1.0 / ord))
        }
    }

    fn solve(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        // Solve Ax = b using Gaussian elimination with partial pivoting
        let mat_a = to_faer(a)?;
        let mat_b = to_faer(b)?;

        let n = mat_a.nrows();
        if n != mat_a.ncols() {
            return Err(RumpyError::NotSquare(a.shape().to_vec()));
        }
        if n != mat_b.nrows() {
            return Err(RumpyError::IncompatibleShapes(
                a.shape().to_vec(),
                b.shape().to_vec(),
            ));
        }

        let m = mat_b.ncols();

        // Augmented matrix [A | b]
        let mut aug = Mat::zeros(n, n + m);
        for i in 0..n {
            for j in 0..n {
                aug.write(i, j, mat_a.read(i, j));
            }
            for j in 0..m {
                aug.write(i, n + j, mat_b.read(i, j));
            }
        }

        // Forward elimination
        for i in 0..n {
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug.read(k, i).abs() > aug.read(max_row, i).abs() {
                    max_row = k;
                }
            }

            if max_row != i {
                for j in 0..(n + m) {
                    let tmp = aug.read(i, j);
                    aug.write(i, j, aug.read(max_row, j));
                    aug.write(max_row, j, tmp);
                }
            }

            let pivot = aug.read(i, i);
            if pivot.abs() < 1e-14 {
                return Err(RumpyError::SingularMatrix);
            }

            for k in (i + 1)..n {
                let factor = aug.read(k, i) / pivot;
                for j in i..(n + m) {
                    aug.write(k, j, aug.read(k, j) - factor * aug.read(i, j));
                }
            }
        }

        // Back substitution
        let mut x = Mat::zeros(n, m);
        for i in (0..n).rev() {
            for j in 0..m {
                let mut sum = aug.read(i, n + j);
                for k in (i + 1)..n {
                    sum -= aug.read(i, k) * x.read(k, j);
                }
                x.write(i, j, sum / aug.read(i, i));
            }
        }

        if b.shape().len() == 1 || b.shape()[1] == 1 {
            Ok(from_faer_vec(&x))
        } else {
            Ok(from_faer(&x))
        }
    }

    fn lstsq(a: &CpuArray, b: &CpuArray) -> Result<CpuArray> {
        let a_pinv = Self::pinv(a)?;
        Self::matmul(&a_pinv, b)
    }

    fn qr(arr: &CpuArray) -> Result<(CpuArray, CpuArray)> {
        // Gram-Schmidt QR decomposition
        let mat = to_faer(arr)?;
        let (m, n) = (mat.nrows(), mat.ncols());
        let k = m.min(n);

        let mut q = Mat::zeros(m, k);
        let mut r = Mat::zeros(k, n);

        for j in 0..k {
            // Start with column j of A
            let mut v: Vec<f64> = (0..m).map(|i| mat.read(i, j)).collect();

            // Orthogonalize against previous columns
            for i in 0..j {
                let dot: f64 = (0..m).map(|row| q.read(row, i) * mat.read(row, j)).sum();
                r.write(i, j, dot);
                for (row, v_elem) in v.iter_mut().enumerate().take(m) {
                    *v_elem -= dot * q.read(row, i);
                }
            }

            // Normalize
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            r.write(j, j, norm);

            if norm > 1e-14 {
                for (row, &v_elem) in v.iter().enumerate().take(m) {
                    q.write(row, j, v_elem / norm);
                }
            }
        }

        // Fill remaining R columns
        for j in k..n {
            for i in 0..k {
                let mut dot = 0.0;
                for row in 0..m {
                    dot += q.read(row, i) * mat.read(row, j);
                }
                r.write(i, j, dot);
            }
        }

        Ok((from_faer(&q), from_faer(&r)))
    }

    fn lu(arr: &CpuArray) -> Result<(CpuArray, CpuArray, CpuArray)> {
        let mat = to_faer(arr)?;
        let n = mat.nrows();
        if n != mat.ncols() {
            return Err(RumpyError::NotSquare(arr.shape().to_vec()));
        }

        let mut l = Mat::zeros(n, n);
        let mut u = mat.clone();
        let mut p = Mat::zeros(n, n);

        // Initialize P as identity
        for i in 0..n {
            p.write(i, i, 1.0);
        }

        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if u.read(k, i).abs() > u.read(max_row, i).abs() {
                    max_row = k;
                }
            }

            // Swap in U, L, and P
            if max_row != i {
                for j in 0..n {
                    let tmp = u.read(i, j);
                    u.write(i, j, u.read(max_row, j));
                    u.write(max_row, j, tmp);

                    let tmp = p.read(i, j);
                    p.write(i, j, p.read(max_row, j));
                    p.write(max_row, j, tmp);
                }
                for j in 0..i {
                    let tmp = l.read(i, j);
                    l.write(i, j, l.read(max_row, j));
                    l.write(max_row, j, tmp);
                }
            }

            l.write(i, i, 1.0);

            let pivot = u.read(i, i);
            if pivot.abs() < 1e-14 {
                continue;
            }

            for k in (i + 1)..n {
                let factor = u.read(k, i) / pivot;
                l.write(k, i, factor);
                for j in i..n {
                    u.write(k, j, u.read(k, j) - factor * u.read(i, j));
                }
            }
        }

        Ok((from_faer(&p), from_faer(&l), from_faer(&u)))
    }

    fn cholesky(arr: &CpuArray) -> Result<CpuArray> {
        let mat = to_faer(arr)?;
        let n = mat.nrows();
        if n != mat.ncols() {
            return Err(RumpyError::NotSquare(arr.shape().to_vec()));
        }

        let mut l = Mat::zeros(n, n);

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                if j == i {
                    for k in 0..j {
                        sum += l.read(j, k) * l.read(j, k);
                    }
                    let diag = mat.read(j, j) - sum;
                    if diag <= 0.0 {
                        return Err(RumpyError::InvalidArgument(
                            "Matrix is not positive definite".to_string(),
                        ));
                    }
                    l.write(j, j, diag.sqrt());
                } else {
                    for k in 0..j {
                        sum += l.read(i, k) * l.read(j, k);
                    }
                    l.write(i, j, (mat.read(i, j) - sum) / l.read(j, j));
                }
            }
        }

        Ok(from_faer(&l))
    }

    fn svd(arr: &CpuArray) -> Result<(CpuArray, CpuArray, CpuArray)> {
        // Simple power iteration SVD for the first singular value
        // This is a placeholder - real SVD needs more sophisticated algorithm
        let mat = to_faer(arr)?;
        let (m, n) = (mat.nrows(), mat.ncols());
        let k = m.min(n);

        // For now, compute via eigendecomposition of A^T A
        let mt = mat.transpose();
        let ata = mt * &mat;

        // Simple eigenvalue computation for symmetric matrix
        let ata_arr = from_faer(&ata);
        let (eigenvalues, v) = Self::eig(&ata_arr)?;

        // Singular values are sqrt of eigenvalues (take only first k)
        let s_data: Vec<f64> = eigenvalues
            .as_f64_slice()
            .iter()
            .take(k)
            .map(|&x| x.max(0.0).sqrt())
            .collect();
        let s = CpuArray::from_f64_vec(s_data, vec![k])?;

        // V is n x n, we need only first k columns for V_k (n x k)
        let v_mat = to_faer(&v)?;

        // U = A V_k S^-1 (m x k)
        let mut u = Mat::zeros(m, k);
        for j in 0..k {
            let sigma = s.get_flat(j);
            if sigma.abs() > 1e-14 {
                // Compute (A * v_j) / sigma
                for i in 0..m {
                    let mut sum = 0.0;
                    for l in 0..n {
                        sum += mat.read(i, l) * v_mat.read(l, j);
                    }
                    u.write(i, j, sum / sigma);
                }
            }
        }

        // V^T is k x n (first k rows of V transposed)
        let mut vt_data = vec![0.0; k * n];
        for i in 0..k {
            for j in 0..n {
                vt_data[i * n + j] = v_mat.read(j, i);
            }
        }
        let vt_arr = CpuArray::from_f64_vec(vt_data, vec![k, n])?;

        Ok((from_faer(&u), s, vt_arr))
    }

    fn eig(arr: &CpuArray) -> Result<(CpuArray, CpuArray)> {
        // Simple power iteration for dominant eigenvalue
        // This is a placeholder - full eigendecomposition needs QR algorithm
        let mat = to_faer(arr)?;
        let n = mat.nrows();
        if n != mat.ncols() {
            return Err(RumpyError::NotSquare(arr.shape().to_vec()));
        }

        // For symmetric matrices, use simple iteration
        // This is a very basic implementation
        let mut eigenvalues = Vec::with_capacity(n);
        let mut eigenvectors = Mat::zeros(n, n);

        let mut work = mat.clone();

        for k in 0..n {
            // Power iteration for dominant eigenvalue
            let mut v: Vec<f64> = (0..n).map(|_| 1.0).collect();
            let mut eigenvalue = 0.0;

            for _ in 0..100 {
                // Multiply
                let mut new_v = vec![0.0; n];
                for (i, nv) in new_v.iter_mut().enumerate().take(n) {
                    for (j, &vj) in v.iter().enumerate().take(n) {
                        *nv += work.read(i, j) * vj;
                    }
                }

                // Normalize
                let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm < 1e-14 {
                    break;
                }

                eigenvalue = new_v.iter().zip(v.iter()).map(|(a, b)| a * b).sum::<f64>()
                    / v.iter().map(|x| x * x).sum::<f64>();

                for (v_elem, &nv) in v.iter_mut().zip(new_v.iter()).take(n) {
                    *v_elem = nv / norm;
                }
            }

            eigenvalues.push(eigenvalue);
            for (i, &v_elem) in v.iter().enumerate().take(n) {
                eigenvectors.write(i, k, v_elem);
            }

            // Deflate: A = A - λ v v^T
            for i in 0..n {
                for j in 0..n {
                    work.write(i, j, work.read(i, j) - eigenvalue * v[i] * v[j]);
                }
            }
        }

        let eig_arr = CpuArray::from_f64_vec(eigenvalues, vec![n])?;
        Ok((eig_arr, from_faer(&eigenvectors)))
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
            return CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[n, m]), result).unwrap());
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
    fn test_inv() {
        let a = mat(vec![4.0, 7.0, 2.0, 6.0], 2, 2);
        let a_inv = CpuBackend::inv(&a).unwrap();

        // A @ A^-1 should be identity
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
    fn test_transpose() {
        let a = mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let at = CpuBackend::transpose(&a);
        assert_eq!(at.shape(), &[3, 2]);
        assert_eq!(at.as_f64_slice(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
