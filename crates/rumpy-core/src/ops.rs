//! Operation traits that backends implement

use crate::array::Array;
use crate::Result;

/// Array creation operations
pub trait CreationOps: Sized {
    type Array: Array;

    /// Create array of zeros
    fn zeros(shape: Vec<usize>) -> Self::Array;

    /// Create array of ones
    fn ones(shape: Vec<usize>) -> Self::Array;

    /// Create array filled with value
    fn full(shape: Vec<usize>, value: f64) -> Self::Array;

    /// Create array from range [start, stop) with step
    fn arange(start: f64, stop: f64, step: f64) -> Result<Self::Array>;

    /// Create array of evenly spaced values
    fn linspace(start: f64, stop: f64, num: usize) -> Self::Array;

    /// Create identity matrix
    fn eye(n: usize) -> Self::Array;

    /// Create diagonal matrix
    fn diag(arr: &Self::Array, k: i32) -> Result<Self::Array>;
}

/// Element-wise math operations
pub trait MathOps {
    type Array: Array;

    // Unary operations
    fn sin(arr: &Self::Array) -> Self::Array;
    fn cos(arr: &Self::Array) -> Self::Array;
    fn tan(arr: &Self::Array) -> Self::Array;
    fn arcsin(arr: &Self::Array) -> Self::Array;
    fn arccos(arr: &Self::Array) -> Self::Array;
    fn arctan(arr: &Self::Array) -> Self::Array;
    fn sinh(arr: &Self::Array) -> Self::Array;
    fn cosh(arr: &Self::Array) -> Self::Array;
    fn tanh(arr: &Self::Array) -> Self::Array;
    fn exp(arr: &Self::Array) -> Self::Array;
    fn exp2(arr: &Self::Array) -> Self::Array;
    fn log(arr: &Self::Array) -> Self::Array;
    fn log2(arr: &Self::Array) -> Self::Array;
    fn log10(arr: &Self::Array) -> Self::Array;
    fn sqrt(arr: &Self::Array) -> Self::Array;
    fn cbrt(arr: &Self::Array) -> Self::Array;
    fn abs(arr: &Self::Array) -> Self::Array;
    fn sign(arr: &Self::Array) -> Self::Array;
    fn floor(arr: &Self::Array) -> Self::Array;
    fn ceil(arr: &Self::Array) -> Self::Array;
    fn round(arr: &Self::Array) -> Self::Array;
    fn neg(arr: &Self::Array) -> Self::Array;
    fn reciprocal(arr: &Self::Array) -> Self::Array;
    fn square(arr: &Self::Array) -> Self::Array;

    // Binary operations (element-wise, same shape)
    fn add(a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;
    fn sub(a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;
    fn mul(a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;
    fn div(a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;
    fn pow(a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;
    fn maximum(a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;
    fn minimum(a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;

    // Scalar operations
    fn add_scalar(arr: &Self::Array, scalar: f64) -> Self::Array;
    fn sub_scalar(arr: &Self::Array, scalar: f64) -> Self::Array;
    fn mul_scalar(arr: &Self::Array, scalar: f64) -> Self::Array;
    fn div_scalar(arr: &Self::Array, scalar: f64) -> Self::Array;
    fn pow_scalar(arr: &Self::Array, scalar: f64) -> Self::Array;

    // Clip
    fn clip(arr: &Self::Array, min: f64, max: f64) -> Self::Array;
}

/// Statistical/reduction operations
pub trait StatsOps {
    type Array: Array;

    // Full array reductions
    fn sum(arr: &Self::Array) -> f64;
    fn prod(arr: &Self::Array) -> f64;
    fn mean(arr: &Self::Array) -> f64;
    fn var(arr: &Self::Array) -> f64;
    fn std(arr: &Self::Array) -> f64;
    fn min(arr: &Self::Array) -> f64;
    fn max(arr: &Self::Array) -> f64;
    fn argmin(arr: &Self::Array) -> usize;
    fn argmax(arr: &Self::Array) -> usize;

    // Axis reductions
    fn sum_axis(arr: &Self::Array, axis: usize) -> Result<Self::Array>;
    fn mean_axis(arr: &Self::Array, axis: usize) -> Result<Self::Array>;
    fn min_axis(arr: &Self::Array, axis: usize) -> Result<Self::Array>;
    fn max_axis(arr: &Self::Array, axis: usize) -> Result<Self::Array>;

    // Cumulative
    fn cumsum(arr: &Self::Array) -> Self::Array;
    fn cumprod(arr: &Self::Array) -> Self::Array;

    // Boolean
    fn all(arr: &Self::Array) -> bool;
    fn any(arr: &Self::Array) -> bool;
}

/// Linear algebra operations
pub trait LinalgOps {
    type Array: Array;

    /// Matrix multiplication
    fn matmul(a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;

    /// Dot product (1D) or matrix multiply (2D)
    fn dot(a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;

    /// Inner product
    fn inner(a: &Self::Array, b: &Self::Array) -> Result<f64>;

    /// Outer product
    fn outer(a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;

    /// Matrix inverse
    fn inv(arr: &Self::Array) -> Result<Self::Array>;

    /// Pseudo-inverse
    fn pinv(arr: &Self::Array) -> Result<Self::Array>;

    /// Determinant
    fn det(arr: &Self::Array) -> Result<f64>;

    /// Matrix trace
    fn trace(arr: &Self::Array) -> Result<f64>;

    /// Matrix rank
    fn rank(arr: &Self::Array) -> Result<usize>;

    /// Matrix/vector norm
    fn norm(arr: &Self::Array, ord: Option<f64>) -> Result<f64>;

    /// Solve linear system Ax = b
    fn solve(a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;

    /// Least squares solution
    fn lstsq(a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;

    /// QR decomposition
    fn qr(arr: &Self::Array) -> Result<(Self::Array, Self::Array)>;

    /// LU decomposition
    fn lu(arr: &Self::Array) -> Result<(Self::Array, Self::Array, Self::Array)>;

    /// Cholesky decomposition
    fn cholesky(arr: &Self::Array) -> Result<Self::Array>;

    /// Singular value decomposition
    fn svd(arr: &Self::Array) -> Result<(Self::Array, Self::Array, Self::Array)>;

    /// Eigenvalues and eigenvectors
    fn eig(arr: &Self::Array) -> Result<(Self::Array, Self::Array)>;

    /// Eigenvalues only
    fn eigvals(arr: &Self::Array) -> Result<Self::Array>;

    /// Transpose
    fn transpose(arr: &Self::Array) -> Self::Array;
}

/// Array manipulation operations
pub trait ManipulationOps {
    type Array: Array;

    /// Reshape array
    fn reshape(arr: &Self::Array, shape: Vec<usize>) -> Result<Self::Array>;

    /// Flatten to 1D
    fn flatten(arr: &Self::Array) -> Self::Array;

    /// Ravel (flatten, but may return view)
    fn ravel(arr: &Self::Array) -> Self::Array;

    /// Squeeze (remove axes of length 1)
    fn squeeze(arr: &Self::Array) -> Self::Array;

    /// Expand dims
    fn expand_dims(arr: &Self::Array, axis: usize) -> Result<Self::Array>;

    /// Concatenate arrays along axis
    fn concatenate(arrays: &[&Self::Array], axis: usize) -> Result<Self::Array>;

    /// Stack arrays along new axis
    fn stack(arrays: &[&Self::Array], axis: usize) -> Result<Self::Array>;

    /// Vertical stack
    fn vstack(arrays: &[&Self::Array]) -> Result<Self::Array>;

    /// Horizontal stack
    fn hstack(arrays: &[&Self::Array]) -> Result<Self::Array>;

    /// Split array
    fn split(arr: &Self::Array, indices: &[usize], axis: usize) -> Result<Vec<Self::Array>>;

    /// Tile array
    fn tile(arr: &Self::Array, reps: &[usize]) -> Self::Array;

    /// Repeat elements
    fn repeat(arr: &Self::Array, repeats: usize, axis: Option<usize>) -> Result<Self::Array>;

    /// Flip array
    fn flip(arr: &Self::Array, axis: Option<usize>) -> Self::Array;

    /// Roll array
    fn roll(arr: &Self::Array, shift: i64, axis: Option<usize>) -> Self::Array;

    /// Rotate 90 degrees
    fn rot90(arr: &Self::Array, k: i32) -> Result<Self::Array>;
}

/// Random number generation
pub trait RandomOps {
    type Array: Array;

    /// Set random seed
    fn seed(s: u64);

    /// Uniform random [0, 1)
    fn rand(shape: Vec<usize>) -> Self::Array;

    /// Standard normal distribution
    fn randn(shape: Vec<usize>) -> Self::Array;

    /// Random integers in [low, high)
    fn randint(low: i64, high: i64, shape: Vec<usize>) -> Self::Array;

    /// Uniform distribution [low, high)
    fn uniform(low: f64, high: f64, shape: Vec<usize>) -> Self::Array;

    /// Normal distribution
    fn normal(loc: f64, scale: f64, shape: Vec<usize>) -> Self::Array;

    /// Shuffle array in-place
    fn shuffle(arr: &mut Self::Array);

    /// Random permutation
    fn permutation(n: usize) -> Self::Array;

    /// Random choice
    fn choice(arr: &Self::Array, size: usize, replace: bool) -> Self::Array;
}

/// Sorting and searching
pub trait SortOps {
    type Array: Array;

    /// Sort array
    fn sort(arr: &Self::Array, axis: Option<usize>) -> Self::Array;

    /// Argsort (indices that would sort)
    fn argsort(arr: &Self::Array, axis: Option<usize>) -> Self::Array;

    /// Search sorted
    fn searchsorted(arr: &Self::Array, values: &Self::Array) -> Self::Array;

    /// Unique elements
    fn unique(arr: &Self::Array) -> Self::Array;

    /// Where (condition indices)
    fn nonzero(arr: &Self::Array) -> Vec<Self::Array>;
}

/// Comparison operations
pub trait CompareOps {
    type Array: Array;

    fn eq(a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;
    fn ne(a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;
    fn lt(a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;
    fn le(a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;
    fn gt(a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;
    fn ge(a: &Self::Array, b: &Self::Array) -> Result<Self::Array>;

    fn eq_scalar(arr: &Self::Array, scalar: f64) -> Self::Array;
    fn ne_scalar(arr: &Self::Array, scalar: f64) -> Self::Array;
    fn lt_scalar(arr: &Self::Array, scalar: f64) -> Self::Array;
    fn le_scalar(arr: &Self::Array, scalar: f64) -> Self::Array;
    fn gt_scalar(arr: &Self::Array, scalar: f64) -> Self::Array;
    fn ge_scalar(arr: &Self::Array, scalar: f64) -> Self::Array;

    fn isnan(arr: &Self::Array) -> Self::Array;
    fn isinf(arr: &Self::Array) -> Self::Array;
    fn isfinite(arr: &Self::Array) -> Self::Array;
}
