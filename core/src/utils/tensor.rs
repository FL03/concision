/*
    Appellation: tensor <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{gen::*, stack::*};
use nd::*;
use num::traits::{NumAssign, Zero};

/// Creates an n-dimensional array from an iterator of n dimensional arrays.
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

pub fn inverse<T>(matrix: &Array2<T>) -> Option<Array2<T>>
where
    T: Copy + NumAssign + ScalarOperand,
{
    let (rows, cols) = matrix.dim();

    if !matrix.is_square() {
        return None; // Matrix must be square for inversion
    }

    let identity = Array2::eye(rows);

    // Construct an augmented matrix by concatenating the original matrix with an identity matrix
    let mut aug = Array2::zeros((rows, 2 * cols));
    aug.slice_mut(s![.., ..cols]).assign(matrix);
    aug.slice_mut(s![.., cols..]).assign(&identity);

    // Perform Gaussian elimination to reduce the left half to the identity matrix
    for i in 0..rows {
        let pivot = aug[[i, i]];

        if pivot == T::zero() {
            return None; // Matrix is singular
        }

        aug.slice_mut(s![i, ..]).mapv_inplace(|x| x / pivot);

        for j in 0..rows {
            if i != j {
                let am = aug.clone();
                let factor = aug[[j, i]];
                let rhs = am.slice(s![i, ..]);
                aug.slice_mut(s![j, ..])
                    .zip_mut_with(&rhs, |x, &y| *x -= y * factor);
            }
        }
    }

    // Extract the inverted matrix from the augmented matrix
    let inverted = aug.slice(s![.., cols..]);

    Some(inverted.to_owned())
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

pub(crate) mod gen {
    use nd::{Array, Array1, Dimension, IntoDimension, ShapeError};
    use num::traits::{Float, FromPrimitive, Num, NumCast};

    pub fn genspace<T: NumCast>(features: usize) -> Array1<T> {
        Array1::from_iter((0..features).map(|x| T::from(x).unwrap()))
    }

    pub fn linarr<A, D>(dim: impl Clone + IntoDimension<Dim = D>) -> Result<Array<A, D>, ShapeError>
    where
        A: Float,
        D: Dimension,
    {
        let dim = dim.into_dimension();
        let n = dim.size();
        Array::linspace(A::zero(), A::from(n - 1).unwrap(), n).into_shape(dim)
    }

    pub fn linspace<T>(start: T, end: T, n: usize) -> Vec<T>
    where
        T: Copy + FromPrimitive + Num,
    {
        if n <= 1 {
            panic!("linspace requires at least two points");
        }

        let step = (end - start) / T::from_usize(n - 1).unwrap();

        (0..n)
            .map(|i| start + step * T::from_usize(i).unwrap())
            .collect()
    }
    /// creates a matrix from the given shape filled with numerical elements [0, n) spaced evenly by 1
    pub fn rangespace<A, D>(dim: impl IntoDimension<Dim = D>) -> Array<A, D>
    where
        A: FromPrimitive,
        D: Dimension,
    {
        let dim = dim.into_dimension();
        let iter = (0..dim.size()).map(|i| A::from_usize(i).unwrap()).collect();
        Array::from_shape_vec(dim, iter).unwrap()
    }
}

pub(crate) mod stack {
    use nd::{s, Array1, Array2};
    use num::Num;
    /// Creates a larger array from an iterator of smaller arrays.
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
    ///
    pub fn hstack<T>(iter: impl IntoIterator<Item = Array1<T>>) -> Array2<T>
    where
        T: Clone + Num,
    {
        let iter = Vec::from_iter(iter);
        let mut res = Array2::<T>::zeros((iter.first().unwrap().len(), iter.len()));
        for (i, s) in iter.iter().enumerate() {
            res.slice_mut(s![.., i]).assign(s);
        }
        res
    }
    ///
    pub fn vstack<T>(iter: impl IntoIterator<Item = Array1<T>>) -> Array2<T>
    where
        T: Clone + Num,
    {
        let iter = Vec::from_iter(iter);
        let mut res = Array2::<T>::zeros((iter.len(), iter.first().unwrap().len()));
        for (i, s) in iter.iter().enumerate() {
            res.slice_mut(s![i, ..]).assign(s);
        }
        res
    }
}
