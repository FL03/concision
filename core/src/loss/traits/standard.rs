/*
    Appellation: loss <module>
    Contrib: @FL03
*/

/// Compute the mean absolute error (MAE) of the object; more formally, we define the MAE as
/// the average of the absolute differences between the predicted and actual values:
///
/// ```math
/// Err = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
/// ```
pub trait MeanAbsoluteError {
    type Output;

    fn mae(&self) -> Self::Output;
}
/// The [`MeanSquaredError`] (MSE) is the average of the squared differences between the
/// ($$\hat{y_{i}}$$) and actual values ($`y_{i}`$):
///
/// ```math
/// Err = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
/// ```
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
