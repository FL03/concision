/*
    Appellation: gradient <module>
    Contrib: @FL03
*/
use crate::traits::L2Norm;
use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::Float;

/// Clip the gradient to a maximum value.
pub fn clip_gradient<A, D>(gradient: &mut Array<A, D>, threshold: A)
where
    A: Float + ScalarOperand,
    D: Dimension,
{
    let norm = gradient.l2_norm();
    if norm > threshold {
        let scale = threshold / norm;
        gradient.mapv_inplace(|x| x * scale);
    }
}

pub fn clip_inf_nan<A, D>(gradient: &mut Array<A, D>, threshold: A)
where
    A: Float + ScalarOperand,
    D: Dimension,
{
    let norm = gradient.l2_norm();
    gradient.mapv_inplace(|x| {
        if x.is_nan() {
            A::one() / norm
        } else if x.is_infinite() {
            threshold / norm
        } else {
            x
        }
    });
}
