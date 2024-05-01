/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{arrays::*, assertions::*};
use nd::*;
use num::traits::{Float, Num};

/// Utilitary function that returns a new *n*-dimensional array of dimension `shape` with the same
/// datatype and memory order as the input `arr`.
pub fn array_like<S, A, D, Sh>(arr: &ArrayBase<S, D>, shape: Sh, elem: A) -> Array<A, D>
where
    A: Clone,
    D: Dimension,
    S: Data<Elem = A>,
    Sh: ShapeBuilder<Dim = D>,
{
    // TODO `is_standard_layout` only works on owned arrays. Change it if using `ArrayBase`.
    if arr.is_standard_layout() {
        Array::from_elem(shape, elem)
    } else {
        Array::from_elem(shape.f(), elem)
    }
}
///
pub fn floor_div<T>(numerator: T, denom: T) -> T
where
    T: Copy + Num,
{
    (numerator - (numerator % denom)) / denom
}

/// Round the given value to the given number of decimal places.
pub fn round_to<T: Float>(val: T, decimals: usize) -> T {
    let factor = T::from(10).expect("").powi(decimals as i32);
    (val * factor).round() / factor
}

pub(crate) mod assertions {
    use ndarray::{Array, Dimension, ScalarOperand};
    use num::traits::{FromPrimitive, Signed};
    ///
    pub fn assert_atol<T, D>(a: &Array<T, D>, b: &Array<T, D>, tol: T)
    where
        D: Dimension,
        T: FromPrimitive + PartialOrd + ScalarOperand + Signed + ToString,
    {
        let err = (b - a).mapv(|i| i.abs()).mean().unwrap();
        assert!(err <= tol, "Error: {}", err.to_string());
    }
    /// A function helper for testing that some result is ok
    pub fn assert_ok<T, E>(res: Result<T, E>) -> T
    where
        E: core::fmt::Debug,
    {
        assert!(res.is_ok(), "Error: {:?}", res.err());
        res.unwrap()
    }
    ///
    pub fn assert_approx<T>(a: T, b: T, epsilon: T)
    where
        T: PartialOrd + Signed + ToString,
    {
        let err = (b - a).abs();
        assert!(err < epsilon, "Error: {}", err.to_string())
    }
    ///
    pub fn almost_equal<T>(a: T, b: T, epsilon: T) -> bool
    where
        T: PartialOrd + Signed,
    {
        (b - a).abs() < epsilon
    }
}

pub(crate) mod arrays {
    use ndarray::{
        concatenate, s, Array, Array1, Array2, Axis, Dimension, IntoDimension, RemoveAxis,
        ScalarOperand, ShapeError,
    };
    use num::traits::{Float, Num, NumAssign, NumCast, Zero};

    /// Creates an n-dimensional array from an iterator of n dimensional arrays.
    pub fn concat_iter<D, T>(
        axis: usize,
        iter: impl IntoIterator<Item = Array<T, D>>,
    ) -> Array<T, D>
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

    pub fn genspace<T: NumCast>(features: usize) -> Array1<T> {
        Array1::from_iter((0..features).map(|x| T::from(x).unwrap()))
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
        T: NumCast,
    {
        let dim = dim.into_dimension();
        let n = dim.as_array_view().product();
        Array::from_iter((0..n).map(|x| T::from(x).unwrap())).into_shape(dim)
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
