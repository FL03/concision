/*
    Appellation: loss <module>
    Contrib: @FL03
*/

/// A trait for computing the cross-entropy loss of a tensor or array
pub trait CrossEntropy {
    type Output;

    fn cross_entropy(&self) -> Self::Output;
}
/// A trait for computing the mean absolute error of a tensor or array
pub trait MeanAbsoluteError {
    type Output;

    fn mae(&self) -> Self::Output;
}
/// A trait for computing the mean squared error of a tensor or array
pub trait MeanSquaredError {
    type Output;

    fn mse(&self) -> Self::Output;
}

/*
 ************* Implementations *************
*/

use ndarray::{ArrayBase, Data, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};

impl<A, S, D> CrossEntropy for ArrayBase<S, D, A>
where
    A: Float + FromPrimitive + ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = A;

    fn cross_entropy(&self) -> Self::Output {
        self.mapv(|x| -x.ln()).mean().unwrap()
    }
}

impl<A, S, D> MeanAbsoluteError for ArrayBase<S, D, A>
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

impl<A, S, D> MeanSquaredError for ArrayBase<S, D, A>
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
