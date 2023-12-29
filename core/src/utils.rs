/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

use ndarray::prelude::*;
use ndarray::{concatenate, IntoDimension, RemoveAxis, ShapeError};
use num::cast::AsPrimitive;
use num::{Float, Num, NumCast, Zero};

pub fn arange<T>(a: T, b: T, h: T) -> Array1<T>
where
    T: AsPrimitive<usize> + Float,
{
    let n: usize = ((b - a) / h).as_();
    let mut res = Array1::<T>::zeros(n);
    res[0] = a;
    for i in 1..n {
        res[i] = res[i - 1] + h;
    }
    res
}

pub fn cauchy_dot<T, D>(a: &Array<T, D>, lambda: &Array<T, D>, omega: &Array<T, D>) -> T
where
    D: Dimension,
    T: NdFloat,
{
    (a / (omega - lambda)).sum()
}

pub fn compute_inverse<T: NdFloat>(matrix: &Array2<T>) -> Option<Array2<T>> {
    let (rows, cols) = matrix.dim();

    if rows != cols {
        return None; // Matrix must be square for inversion
    }

    let identity = Array2::eye(rows);

    // Concatenate the original matrix with an identity matrix
    let mut augmented_matrix = Array2::zeros((rows, 2 * cols));
    augmented_matrix.slice_mut(s![.., ..cols]).assign(matrix);
    augmented_matrix.slice_mut(s![.., cols..]).assign(&identity);

    // Perform Gaussian elimination to reduce the left half to the identity matrix
    for i in 0..rows {
        let pivot = augmented_matrix[[i, i]];

        if pivot == T::zero() {
            return None; // Matrix is singular
        }

        augmented_matrix
            .slice_mut(s![i, ..])
            .mapv_inplace(|x| x / pivot);

        for j in 0..rows {
            if i != j {
                let am = augmented_matrix.clone();
                let factor = augmented_matrix[[j, i]];
                let rhs = am.slice(s![i, ..]);
                augmented_matrix
                    .slice_mut(s![j, ..])
                    .zip_mut_with(&rhs, |x, &y| *x -= y * factor);
            }
        }
    }

    // Extract the inverted matrix from the augmented matrix
    let inverted = augmented_matrix.slice(s![.., cols..]);

    Some(inverted.to_owned())
}

pub fn concat_iter<D, T>(axis: usize, iter: impl IntoIterator<Item = Array<T, D>>) -> Array<T, D>
where
    D: RemoveAxis,
    T: Clone,
{
    let mut arr = iter.into_iter().collect::<Vec<_>>();
    let mut out = arr.pop().unwrap();
    for i in arr {
        out = concatenate!(Axis(axis), out, i);
    }
    out
}

pub fn linarr<T, D>(dim: impl IntoDimension<Dim = D>) -> Result<Array<T, D>, ShapeError>
where
    D: Dimension,
    T: Float,
{
    let dim = dim.into_dimension();
    let n = dim.as_array_view().product();
    Array::linspace(T::one(), T::from(n).unwrap(), n).into_shape(dim)
}

pub fn linspace<T, D>(dim: impl IntoDimension<Dim = D>) -> Result<Array<T, D>, ShapeError>
where
    D: Dimension,
    T: Float,
{
    let dim = dim.into_dimension();
    let n = dim.as_array_view().product();
    Array::linspace(T::zero(), T::from(n - 1).unwrap(), n).into_shape(dim)
}

/// creates a matrix from the given shape filled with numerical elements [0, n) spaced evenly by 1
pub fn rangespace<T, D>(dim: impl IntoDimension<Dim = D>) -> Array<T, D>
where
    D: Dimension,
    T: Num + NumCast,
{
    let dim = dim.into_dimension();
    let iter = (0..dim.size()).map(|i| T::from(i).unwrap());
    Array::from_shape_vec(dim, iter.collect()).unwrap()
}

pub fn round_to<T: Float>(val: T, decimals: usize) -> T {
    let factor = T::from(10).expect("").powi(decimals as i32);
    (val * factor).round() / factor
}

pub fn stack_iter<T>(iter: impl IntoIterator<Item = Array1<T>>) -> Array2<T>
where
    T: Clone + Num,
{
    let mut iter = iter.into_iter();
    let first = iter.next().unwrap();
    let shape = [iter.size_hint().0 + 1, first.len()];
    let mut res = Array2::<T>::zeros(shape);
    res.slice_mut(s![0, ..]).assign(&first);
    for (i, s) in iter.enumerate() {
        res.slice_mut(s![i + 1, ..]).assign(&s);
    }
    res
}

/// Returns the upper triangular portion of a matrix.
pub fn triu<T>(a: &Array2<T>) -> Array2<T>
where
    T: Clone + Zero,
{
    let mut out = a.clone();
    for i in 0..a.shape()[0] {
        for j in 0..i {
            out[[i, j]] = T::zero();
        }
    }
    out
}
/// Returns the lower triangular portion of a matrix.
pub fn tril<T>(a: &Array2<T>) -> Array2<T>
where
    T: Clone + Zero,
{
    let mut out = a.clone();
    for i in 0..a.shape()[0] {
        for j in i + 1..a.shape()[1] {
            out[[i, j]] = T::zero();
        }
    }
    out
}
