/*
    Appellation: gradient <module>
    Contrib: @FL03
*/
use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::Float;

/// Clip the gradient to a maximum value.
pub fn clip_gradient<A, D>(gradient: &mut Array<A, D>, threshold: A)
where
    A: Float + ScalarOperand,
    D: Dimension,
{
    gradient.clamp(-threshold, threshold);
}

pub fn clip_inf_nan<A, D>(gradient: &mut Array<A, D>, threshold: A)
where
    A: Float + ScalarOperand,
    D: Dimension,
{
    let norm = gradient.pow2().sum().sqrt();
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
