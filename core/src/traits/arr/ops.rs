/*
   Appellation: arr <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::{Array, Dimension, Ix2, LinalgScalar, ScalarOperand};
use num::traits::NumAssign;

pub trait Affine<T> {
    type Output;

    fn affine(&self, mul: T, add: T) -> Self::Output;
}

pub trait Inverse<T = f64> {
    type Output;

    fn inverse(&self) -> Self::Output;
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

impl<T> Inverse<T> for Array<T, Ix2>
where
    T: Copy + NumAssign + ScalarOperand,
{
    type Output = Option<Self>;
    fn inverse(&self) -> Self::Output {
        crate::inverse(self)
    }
}
