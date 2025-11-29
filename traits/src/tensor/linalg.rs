/*
    appellation: tensor_ops <module>
    authors: @FL03
*/

/// apply an affine transformation to a tensor;
/// affine transformation is defined as `mul * self + add`
pub trait Affine<X, Y = X> {
    type Output;

    fn affine(&self, mul: X, add: Y) -> Self::Output;
}
/// The [`Inverse`] trait generically establishes an interface for computing the inverse of a
/// type, regardless of if its a tensor, scalar, or some other compatible type.
pub trait Inverse {
    /// the output, or result, of the inverse operation
    type Output;
    /// compute the inverse of the current object, producing some [`Output`](Inverse::Output)
    fn inverse(&self) -> Self::Output;
}
/// The [`MatMul`] trait defines an interface for matrix multiplication.
pub trait MatMul<Rhs = Self> {
    type Output;

    fn matmul(&self, rhs: &Rhs) -> Self::Output;
}
/// The [`MatPow`] trait defines an interface for computing the power of some matrix
pub trait MatPow<Rhs = Self> {
    type Output;

    fn matpow(&self, rhs: Rhs) -> Self::Output;
}

/// The [`Transpose`] trait generically establishes an interface for transposing a type
pub trait Transpose {
    /// the output, or result, of the transposition
    type Output;
    /// transpose a reference to the current object
    fn transpose(&self) -> Self::Output;
}

/*
 ********* Implementations *********
*/
use ndarray::linalg::Dot;
use ndarray::{Array, Array2, ArrayBase, Data, Dimension, Ix2, LinalgScalar, ScalarOperand, s};
use num_traits::{Num, NumAssign};

impl<A, D> Affine<A> for Array<A, D>
where
    A: LinalgScalar + ScalarOperand,
    D: Dimension,
{
    type Output = Array<A, D>;

    fn affine(&self, mul: A, add: A) -> Self::Output {
        self * mul + add
    }
}

// #[cfg(not(feature = "blas"))]
impl<T> Inverse for Array<T, Ix2>
where
    T: Copy + NumAssign + ScalarOperand,
{
    type Output = Option<Self>;

    fn inverse(&self) -> Self::Output {
        let (rows, cols) = self.dim();

        if !self.is_square() {
            return None; // Matrix must be square for inversion
        }

        let identity = Array2::eye(rows);

        // Construct an augmented matrix by concatenating the original matrix with an identity matrix
        let mut aug = Array2::zeros((rows, 2 * cols));
        aug.slice_mut(s![.., ..cols]).assign(self);
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
}
// #[cfg(feature = "blas")]
// impl<T> Inverse for Array<T, Ix2>
// where
//     T: Copy + NumAssign + ScalarOperand,
// {
//     type Output = Option<Self>;

//     fn inverse(&self) -> Self::Output {
//         use ndarray_linalg::solve::Inverse;
//         self.inv().ok()
//     }
// }

impl<A, S, D, X, Y> MatMul<X> for ArrayBase<S, D, A>
where
    A: ndarray::LinalgScalar,
    D: Dimension,
    S: Data<Elem = A>,
    ArrayBase<S, D, A>: Dot<X, Output = Y>,
{
    type Output = Y;

    fn matmul(&self, rhs: &X) -> Self::Output {
        <Self as Dot<X>>::dot(self, rhs)
    }
}

impl<T> MatMul<Vec<T>> for Vec<T>
where
    T: Copy + Num,
{
    type Output = T;

    fn matmul(&self, rhs: &Vec<T>) -> T {
        self.iter()
            .zip(rhs.iter())
            .fold(T::zero(), |acc, (&a, &b)| acc + a * b)
    }
}

impl<T, const N: usize> MatMul<[T; N]> for [T; N]
where
    T: Copy + Num,
{
    type Output = T;

    fn matmul(&self, rhs: &[T; N]) -> T {
        self.iter()
            .zip(rhs.iter())
            .fold(T::zero(), |acc, (&a, &b)| acc + a * b)
    }
}
impl<A, S> MatPow<i32> for ArrayBase<S, ndarray::Ix2>
where
    A: Copy + Num + 'static,
    S: Data<Elem = A>,
    ArrayBase<S, Ix2>: Clone + Dot<ArrayBase<S, Ix2>, Output = Array<A, Ix2>>,
{
    type Output = Array<A, Ix2>;

    fn matpow(&self, rhs: i32) -> Self::Output {
        if !self.is_square() {
            panic!("Matrix must be square to be raised to a power");
        }
        let mut res = Array::eye(self.shape()[0]);
        for _ in 0..rhs {
            res = res.dot(self);
        }
        res
    }
}

impl<'a, A, S, D> Transpose for &'a ArrayBase<S, D, A>
where
    A: 'a,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = ndarray::ArrayView<'a, A, D>;

    fn transpose(&self) -> Self::Output {
        self.t()
    }
}
