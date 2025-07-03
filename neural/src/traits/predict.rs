/*
    Appellation: predict <module>
    Contrib: @FL03
*/

/// [Predict] isn't designed to be implemented directly, rather, as a blanket impl for any
/// entity that implements the [`Forward`](cnc::Forward) trait. This is primarily used to
/// define the base functionality of the [`Model`](crate::Model) trait.
pub trait Predict<Rhs> {
    type Output;

    private!();

    fn predict(&self, input: &Rhs) -> crate::ModelResult<Self::Output>;
}

/// This trait extends the [`Predict`] trait to include a confidence score for the prediction.
/// The confidence score is calculated as the inverse of the variance of the output.
pub trait PredictWithConfidence<Rhs>: Predict<Rhs> {
    type Confidence;

    fn predict_with_confidence(
        &self,
        input: &Rhs,
    ) -> crate::ModelResult<(Self::Output, Self::Confidence)>;
}

/*
 ************* Implementations *************
*/

use cnc::Forward;
use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};

impl<M, U, V> Predict<U> for M
where
    M: Forward<U, Output = V>,
{
    type Output = V;

    seal!();

    fn predict(&self, input: &U) -> crate::ModelResult<Self::Output> {
        self.forward(input).map_err(core::convert::Into::into)
    }
}

impl<M, U, A, D> PredictWithConfidence<U> for M
where
    A: Float + FromPrimitive + ScalarOperand,
    D: Dimension,
    Self: Predict<U, Output = Array<A, D>>,
{
    type Confidence = A;

    fn predict_with_confidence(
        &self,
        input: &U,
    ) -> Result<(Self::Output, Self::Confidence), crate::ModelError> {
        // Get the base prediction
        let prediction = Predict::predict(self, input)?;
        let shape = prediction.shape();
        // Calculate confidence as the inverse of the variance of the output
        // For each sample, compute the variance across the output dimensions
        let batch_size = shape[0];

        let mut variance_sum = A::zero();

        for sample in prediction.rows() {
            // Compute variance
            let variance = sample.var(A::one());
            variance_sum = variance_sum + variance;
        }

        // Average variance across the batch
        let avg_variance = variance_sum / A::from_usize(batch_size).unwrap();
        // Confidence: inverse of variance (clipped to avoid division by zero)
        let confidence = (A::one() + avg_variance).recip();

        Ok((prediction, confidence))
    }
}
