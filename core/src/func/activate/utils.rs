/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::math::Exp;
use nd::prelude::{Array, ArrayBase, Axis, Dimension};
use nd::{Data, RemoveAxis, ScalarOperand};
use num::complex::ComplexFloat;
use num::traits::{One, Zero};

/// Heaviside activation function
pub fn heavyside<T>(x: T) -> T
where
    T: One + PartialOrd + Zero,
{
    if x > T::zero() {
        T::one()
    } else {
        T::zero()
    }
}
///
pub fn relu<T>(args: T) -> T
where
    T: PartialOrd + Zero,
{
    if args > T::zero() {
        args
    } else {
        T::zero()
    }
}
///
pub fn sigmoid<T>(args: T) -> T
where
    T: ComplexFloat,
{
    (T::one() + args.neg().exp()).recip()
}
///
pub fn softmax<A, S, D>(args: &ArrayBase<S, D>) -> Array<A, D>
where
    A: ComplexFloat + ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    let e = args.exp();
    &e / e.sum()
}
///
pub fn softmax_axis<A, S, D>(args: &ArrayBase<S, D>, axis: usize) -> Array<A, D>
where
    A: ComplexFloat + ScalarOperand,
    D: RemoveAxis,
    S: Data<Elem = A>,
{
    let axis = Axis(axis);
    let e = args.exp();
    &e / &e.sum_axis(axis)
}
///
pub fn tanh<T>(args: T) -> T
where
    T: ComplexFloat,
{
    args.tanh()
}
