/*
    Appellation: utils <loss>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use concision_math::{Abs, Squared};
use ndarray::{ArrayBase, Data, Dimension, ScalarOperand};
use num::traits::{FromPrimitive, Num, Pow, Signed};

/// A functional implementation of the mean absolute error loss function which compares two similar
/// [arrays](ndarray::ArrayBase)
pub fn mae<A, S, D>(pred: &ArrayBase<S, D>, target: &ArrayBase<S, D>) -> Option<A>
where
    A: FromPrimitive + Num + ScalarOperand + Signed,
    D: Dimension,
    S: Data<Elem = A>,
{
    (pred - target).abs().mean()
}
/// A functional implementation of the mean squared error loss function that compares two similar
/// [arrays](ndarray::ArrayBase)
pub fn mse<A, S, D>(pred: &ArrayBase<S, D>, target: &ArrayBase<S, D>) -> Option<A>
where
    A: FromPrimitive + Num + Pow<i32, Output = A> + ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    (pred - target).sqrd().mean()
}
