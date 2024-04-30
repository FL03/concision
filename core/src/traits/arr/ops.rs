/*
   Appellation: arr <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::linalg::Dot;
use nd::*;
use num::traits::{Num, NumAssign};

pub trait Affine<T> {
    type Output;

    fn affine(&self, mul: T, add: T) -> Self::Output;
}

pub trait Inverse {
    type Output;

    fn inverse(&self) -> Self::Output;
}

pub trait Matmul<Rhs = Self> {
    type Output;

    fn matmul(&self, rhs: Rhs) -> Self::Output;
}

pub trait Matpow<Rhs = Self> {
    type Output;

    fn pow(&self, rhs: Rhs) -> Self::Output;
}

/*
 ********* Implementations *********
*/
impl<A, D> Affine<A> for Array<A, D>
where
    A: LinalgScalar + ScalarOperand,
    D: Dimension,
{
    type Output = Array<A, D>;
    fn affine(&self, mul: A, add: A) -> Self {
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

impl<A, B, C> Matmul<B> for A
where
    A: Dot<B, Output = C>,
{
    type Output = C;
    fn matmul(&self, rhs: B) -> Self::Output {
        self.dot(&rhs)
    }
}

impl<A> Matpow<i32> for Array2<A>
where
    A: Clone + Num,
    Array2<A>: Dot<Self, Output = Self>,
{
    type Output = Array2<A>;

    fn pow(&self, rhs: i32) -> Self::Output {
        if !self.is_square() {
            panic!("Matrix must be square to be raised to a power");
        }
        let mut res = Array::eye(self.shape()[0]);
        for _ in 0..rhs {
            res = res.dot(&self);
        }
        res
    }
}
