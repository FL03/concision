/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

use ndarray::prelude::{s, Array, Array1, Array2, Axis, Dimension, NdFloat};
use ndarray::{concatenate, IntoDimension, RemoveAxis, ShapeError};
use num::cast::AsPrimitive;
use num::Float;

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

pub fn round_to<T: Float>(val: T, decimals: usize) -> T {
    let factor = T::from(10).expect("").powi(decimals as i32);
    (val * factor).round() / factor
}

pub fn tril<T>(a: &Array2<T>) -> Array2<T>
where
    T: NdFloat,
{
    let mut out = a.clone();
    for i in 0..a.shape()[0] {
        for j in i + 1..a.shape()[1] {
            out[[i, j]] = T::zero();
        }
    }
    out
}
