/*
    Appellation: loss <module>
    Contrib: @FL03
*/

/// Compute the mean absolute error (MAE) of the object.
pub trait MeanAbsoluteError {
    type Output;

    fn mae(&self) -> Self::Output;
}
/// Compute the mean squared error (MSE) of the object.
pub trait MeanSquaredError {
    type Output;

    fn mse(&self) -> Self::Output;
}

/*
 ************* Implementations *************
*/

use ndarray::{ArrayBase, Data, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};

impl<A, S, D> MeanAbsoluteError for ArrayBase<S, D>
where
    A: Float + FromPrimitive + ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = A;

    fn mae(&self) -> Self::Output {
        self.abs().mean().unwrap()
    }
}

impl<A, S, D> MeanSquaredError for ArrayBase<S, D>
where
    A: Float + FromPrimitive + ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = A;

    fn mse(&self) -> Self::Output {
        self.pow2().mean().unwrap()
    }
}
