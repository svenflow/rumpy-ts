//! Array manipulation operations for CPU backend

use crate::{CpuArray, CpuBackend};
use ndarray::{ArrayD, Axis, IxDyn};
use rumpy_core::{ops::LinalgOps, ops::ManipulationOps, Array, Result, RumpyError};

impl ManipulationOps for CpuBackend {
    type Array = CpuArray;

    fn reshape(arr: &CpuArray, shape: Vec<usize>) -> Result<CpuArray> {
        let expected_size: usize = shape.iter().product();
        if arr.size() != expected_size {
            return Err(RumpyError::InvalidShape(format!(
                "Cannot reshape array of size {} into shape {:?}",
                arr.size(),
                shape
            )));
        }

        let data = arr.as_f64_slice();
        Ok(CpuArray::from_ndarray(
            ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap(),
        ))
    }

    fn flatten(arr: &CpuArray) -> CpuArray {
        let data = arr.as_f64_slice();
        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap())
    }

    fn ravel(arr: &CpuArray) -> CpuArray {
        Self::flatten(arr)
    }

    fn squeeze(arr: &CpuArray) -> CpuArray {
        let shape: Vec<usize> = arr.shape().iter().filter(|&&s| s != 1).cloned().collect();
        if shape.is_empty() {
            return CpuArray::from_ndarray(
                ArrayD::from_shape_vec(IxDyn(&[1]), arr.as_f64_slice()).unwrap(),
            );
        }
        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&shape), arr.as_f64_slice()).unwrap())
    }

    fn expand_dims(arr: &CpuArray, axis: usize) -> Result<CpuArray> {
        let mut shape = arr.shape().to_vec();
        if axis > shape.len() {
            return Err(RumpyError::InvalidAxis {
                axis,
                ndim: shape.len(),
            });
        }
        shape.insert(axis, 1);
        Ok(CpuArray::from_ndarray(
            ArrayD::from_shape_vec(IxDyn(&shape), arr.as_f64_slice()).unwrap(),
        ))
    }

    fn concatenate(arrays: &[&CpuArray], axis: usize) -> Result<CpuArray> {
        if arrays.is_empty() {
            return Err(RumpyError::InvalidArgument(
                "Cannot concatenate empty array list".to_string(),
            ));
        }

        let first_shape = arrays[0].shape();
        if axis >= first_shape.len() {
            return Err(RumpyError::InvalidAxis {
                axis,
                ndim: first_shape.len(),
            });
        }

        // Verify all arrays have compatible shapes
        for arr in arrays.iter().skip(1) {
            let shape = arr.shape();
            if shape.len() != first_shape.len() {
                return Err(RumpyError::IncompatibleShapes(
                    first_shape.to_vec(),
                    shape.to_vec(),
                ));
            }
            for (i, (&s1, &s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if i != axis && s1 != s2 {
                    return Err(RumpyError::IncompatibleShapes(
                        first_shape.to_vec(),
                        shape.to_vec(),
                    ));
                }
            }
        }

        // Use ndarray's concatenate
        let nd_arrays: Vec<_> = arrays.iter().map(|a| a.as_ndarray().view()).collect();
        let result = ndarray::concatenate(Axis(axis), &nd_arrays)
            .map_err(|e| RumpyError::InvalidArgument(e.to_string()))?;

        Ok(CpuArray::from_ndarray(result))
    }

    fn stack(arrays: &[&CpuArray], axis: usize) -> Result<CpuArray> {
        if arrays.is_empty() {
            return Err(RumpyError::InvalidArgument(
                "Cannot stack empty array list".to_string(),
            ));
        }

        let first_shape = arrays[0].shape();
        for arr in arrays.iter().skip(1) {
            if arr.shape() != first_shape {
                return Err(RumpyError::IncompatibleShapes(
                    first_shape.to_vec(),
                    arr.shape().to_vec(),
                ));
            }
        }

        // Expand dims then concatenate
        let expanded: Result<Vec<CpuArray>> =
            arrays.iter().map(|a| Self::expand_dims(a, axis)).collect();
        let expanded = expanded?;
        let refs: Vec<&CpuArray> = expanded.iter().collect();
        Self::concatenate(&refs, axis)
    }

    fn vstack(arrays: &[&CpuArray]) -> Result<CpuArray> {
        Self::concatenate(arrays, 0)
    }

    fn hstack(arrays: &[&CpuArray]) -> Result<CpuArray> {
        if arrays.is_empty() {
            return Err(RumpyError::InvalidArgument(
                "Cannot hstack empty array list".to_string(),
            ));
        }

        if arrays[0].ndim() == 1 {
            // For 1D arrays, concatenate along axis 0
            Self::concatenate(arrays, 0)
        } else {
            // For 2D+ arrays, concatenate along axis 1
            Self::concatenate(arrays, 1)
        }
    }

    fn split(arr: &CpuArray, indices: &[usize], axis: usize) -> Result<Vec<CpuArray>> {
        let shape = arr.shape();
        if axis >= shape.len() {
            return Err(RumpyError::InvalidAxis {
                axis,
                ndim: shape.len(),
            });
        }

        let axis_len = shape[axis];
        let mut result = Vec::new();
        let mut prev = 0;

        for &idx in indices {
            if idx > axis_len {
                return Err(RumpyError::IndexOutOfBounds {
                    index: idx,
                    size: axis_len,
                });
            }

            let slice = arr
                .as_ndarray()
                .slice_axis(Axis(axis), ndarray::Slice::from(prev..idx));
            result.push(CpuArray::from_ndarray(slice.to_owned()));
            prev = idx;
        }

        // Last segment
        let slice = arr
            .as_ndarray()
            .slice_axis(Axis(axis), ndarray::Slice::from(prev..));
        result.push(CpuArray::from_ndarray(slice.to_owned()));

        Ok(result)
    }

    fn tile(arr: &CpuArray, reps: &[usize]) -> CpuArray {
        let shape = arr.shape();
        let data = arr.as_f64_slice();

        // Pad reps to match dimensions
        let full_len = shape.len().max(reps.len());
        let mut full_reps = vec![1; full_len];
        let offset = full_len - reps.len();
        for (i, &r) in reps.iter().enumerate() {
            full_reps[offset + i] = r;
        }

        // Calculate new shape
        let new_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .map(|(i, &s)| s * full_reps.get(i).unwrap_or(&1))
            .collect();

        let new_size: usize = new_shape.iter().product();
        let mut result = vec![0.0; new_size];

        // Fill by iterating over output indices
        for (out_idx, result_elem) in result.iter_mut().enumerate().take(new_size) {
            let mut idx = out_idx;
            let mut in_idx = 0;
            let mut in_mult = 1;

            for d in (0..shape.len()).rev() {
                let coord = idx % new_shape[d];
                idx /= new_shape[d];

                let in_coord = coord % shape[d];
                in_idx += in_coord * in_mult;
                in_mult *= shape[d];
            }

            *result_elem = data[in_idx];
        }

        CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(&new_shape), result).unwrap())
    }

    fn repeat(arr: &CpuArray, repeats: usize, axis: Option<usize>) -> Result<CpuArray> {
        match axis {
            None => {
                // Flatten and repeat each element
                let data = arr.as_f64_slice();
                let result: Vec<f64> = data.iter().flat_map(|&x| vec![x; repeats]).collect();
                Ok(CpuArray::from_ndarray(
                    ArrayD::from_shape_vec(IxDyn(&[result.len()]), result).unwrap(),
                ))
            }
            Some(ax) => {
                let shape = arr.shape();
                if ax >= shape.len() {
                    return Err(RumpyError::InvalidAxis {
                        axis: ax,
                        ndim: shape.len(),
                    });
                }

                // Repeat along axis
                let repeated: Vec<CpuArray> = (0..repeats).map(|_| arr.clone()).collect();
                let refs: Vec<&CpuArray> = repeated.iter().collect();
                Self::concatenate(&refs, ax)
            }
        }
    }

    fn flip(arr: &CpuArray, axis: Option<usize>) -> CpuArray {
        let data = arr.as_ndarray();

        match axis {
            None => {
                // Flip all elements (reverse flattened)
                let flat: Vec<f64> = data.iter().cloned().collect();
                let reversed: Vec<f64> = flat.into_iter().rev().collect();
                CpuArray::from_ndarray(
                    ArrayD::from_shape_vec(IxDyn(arr.shape()), reversed).unwrap(),
                )
            }
            Some(ax) => {
                // Flip along specific axis
                let shape = data.shape();
                if ax >= shape.len() {
                    return arr.clone();
                }

                let mut result = data.clone();
                let axis_len = shape[ax];

                // Use ndarray's slice_axis_inplace with reversed slice
                for i in 0..axis_len / 2 {
                    let j = axis_len - 1 - i;
                    // Swap slices at positions i and j along axis
                    let slice_i = result.slice_axis(Axis(ax), ndarray::Slice::from(i..=i));
                    let slice_j = result.slice_axis(Axis(ax), ndarray::Slice::from(j..=j));
                    let temp_i: Vec<f64> = slice_i.iter().cloned().collect();
                    let temp_j: Vec<f64> = slice_j.iter().cloned().collect();

                    // Write back swapped
                    let mut iter_i = temp_j.into_iter();
                    let mut iter_j = temp_i.into_iter();

                    for val in result
                        .slice_axis_mut(Axis(ax), ndarray::Slice::from(i..=i))
                        .iter_mut()
                    {
                        *val = iter_i.next().unwrap();
                    }
                    for val in result
                        .slice_axis_mut(Axis(ax), ndarray::Slice::from(j..=j))
                        .iter_mut()
                    {
                        *val = iter_j.next().unwrap();
                    }
                }

                CpuArray::from_ndarray(result)
            }
        }
    }

    fn roll(arr: &CpuArray, shift: i64, axis: Option<usize>) -> CpuArray {
        let data = arr.as_f64_slice();
        let n = data.len() as i64;

        if n == 0 {
            return arr.clone();
        }

        match axis {
            None => {
                // Roll flattened array
                let shift = ((shift % n) + n) % n;
                let shift = shift as usize;

                let mut result = vec![0.0; data.len()];
                for (i, &val) in data.iter().enumerate() {
                    let new_i = (i + shift) % data.len();
                    result[new_i] = val;
                }

                CpuArray::from_ndarray(ArrayD::from_shape_vec(IxDyn(arr.shape()), result).unwrap())
            }
            Some(_ax) => {
                // For simplicity, flatten, roll, reshape
                // TODO: Proper axis-aware rolling
                let flat = Self::flatten(arr);
                let rolled = Self::roll(&flat, shift, None);
                Self::reshape(&rolled, arr.shape().to_vec()).unwrap()
            }
        }
    }

    fn rot90(arr: &CpuArray, k: i32) -> Result<CpuArray> {
        let shape = arr.shape();
        if shape.len() < 2 {
            return Err(RumpyError::InvalidArgument(
                "rot90 requires at least 2D array".to_string(),
            ));
        }

        let k = k.rem_euclid(4);
        let mut result = arr.clone();

        for _ in 0..k {
            // Rotate 90 degrees: transpose then flip horizontally
            result = crate::CpuBackend::transpose(&result);
            result = Self::flip(&result, Some(1));
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(data: Vec<f64>, shape: Vec<usize>) -> CpuArray {
        CpuArray::from_f64_vec(data, shape).unwrap()
    }

    #[test]
    fn test_reshape() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = CpuBackend::reshape(&a, vec![3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.as_f64_slice(), a.as_f64_slice());
    }

    #[test]
    fn test_reshape_invalid() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = CpuBackend::reshape(&a, vec![3, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_flatten() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let flat = CpuBackend::flatten(&a);
        assert_eq!(flat.shape(), &[6]);
        assert_eq!(flat.as_f64_slice(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_squeeze() {
        let a = arr(vec![1.0, 2.0, 3.0], vec![1, 3, 1]);
        let squeezed = CpuBackend::squeeze(&a);
        assert_eq!(squeezed.shape(), &[3]);
    }

    #[test]
    fn test_expand_dims() {
        let a = arr(vec![1.0, 2.0, 3.0], vec![3]);
        let expanded = CpuBackend::expand_dims(&a, 0).unwrap();
        assert_eq!(expanded.shape(), &[1, 3]);

        let expanded = CpuBackend::expand_dims(&a, 1).unwrap();
        assert_eq!(expanded.shape(), &[3, 1]);
    }

    #[test]
    fn test_concatenate() {
        let a = arr(vec![1.0, 2.0], vec![2]);
        let b = arr(vec![3.0, 4.0], vec![2]);
        let c = CpuBackend::concatenate(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape(), &[4]);
        assert_eq!(c.as_f64_slice(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_stack() {
        let a = arr(vec![1.0, 2.0], vec![2]);
        let b = arr(vec![3.0, 4.0], vec![2]);
        let c = CpuBackend::stack(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_flip() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let flipped = CpuBackend::flip(&a, None);
        assert_eq!(flipped.as_f64_slice(), vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_roll() {
        let a = arr(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let rolled = CpuBackend::roll(&a, 2, None);
        assert_eq!(rolled.as_f64_slice(), vec![4.0, 5.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_tile() {
        let a = arr(vec![1.0, 2.0], vec![2]);
        let tiled = CpuBackend::tile(&a, &[3]);
        assert_eq!(tiled.as_f64_slice(), vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn test_repeat() {
        let a = arr(vec![1.0, 2.0, 3.0], vec![3]);
        let repeated = CpuBackend::repeat(&a, 2, None).unwrap();
        assert_eq!(repeated.as_f64_slice(), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }
}
