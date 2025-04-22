/*
    Appellation: ops <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
/// apply an affine transformation to a tensor;
/// affine transformation is defined as `mul * self + add`
pub trait Affine<X, Y = X> {
    type Output;

    fn affine(&self, mul: X, add: Y) -> Self::Output;
}
/// this trait enables the inversion of a matrix
pub trait Inverse {
    type Output;

    fn inverse(&self) -> Self::Output;
}
/// A trait denoting objects capable of matrix multiplication.
pub trait Matmul<Rhs = Self> {
    type Output;

    fn matmul(&self, rhs: &Rhs) -> Self::Output;
}
/// a trait denoting objects capable of matrix exponentiation
pub trait Matpow<Rhs = Self> {
    type Output;

    fn pow(&self, rhs: Rhs) -> Self::Output;
}

/// the trait denotes the ability to transpose a tensor
pub trait Transpose {
    type Output;

    fn transpose(&self) -> Self::Output;
}

/*
 ********* Implementations *********
*/
use ndarray::linalg::Dot;
use ndarray::{Array, ArrayBase, Data, Dimension, Ix2, LinalgScalar, ScalarOperand};
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

// #[cfg(feature = "blas")]
impl<T> Inverse for Array<T, Ix2>
where
    T: Copy + NumAssign + ScalarOperand,
{
    type Output = Option<Self>;

    fn inverse(&self) -> Self::Output {
        crate::inverse(self)
    }
}

impl<A, S, D, X, Y> Matmul<X> for ArrayBase<S, D>
where
    A: ndarray::LinalgScalar,
    D: Dimension,
    S: Data<Elem = A>,
    ArrayBase<S, D>: Dot<X, Output = Y>,
{
    type Output = Y;

    fn matmul(&self, rhs: &X) -> Self::Output {
        <Self as Dot<X>>::dot(self, rhs)
    }
}

impl<T> Matmul<Vec<T>> for Vec<T>
where
    T: Copy + num::Num,
{
    type Output = T;

    fn matmul(&self, rhs: &Vec<T>) -> T {
        self.iter()
            .zip(rhs.iter())
            .fold(T::zero(), |acc, (&a, &b)| acc + a * b)
    }
}

impl<T, const N: usize> Matmul<[T; N]> for [T; N]
where
    T: Copy + num::Num,
{
    type Output = T;

    fn matmul(&self, rhs: &[T; N]) -> T {
        self.iter()
            .zip(rhs.iter())
            .fold(T::zero(), |acc, (&a, &b)| acc + a * b)
    }
}
impl<A, S> Matpow<i32> for ArrayBase<S, ndarray::Ix2>
where
    A: Copy + Num + 'static,
    S: Data<Elem = A>,
    ArrayBase<S, Ix2>: Clone + Dot<ArrayBase<S, Ix2>, Output = Array<A, Ix2>>,
{
    type Output = Array<A, Ix2>;

    fn pow(&self, rhs: i32) -> Self::Output {
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

impl<'a, A, S, D> Transpose for &'a ArrayBase<S, D>
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
