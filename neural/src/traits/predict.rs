/*
    Appellation: predict <module>
    Contrib: @FL03
*/
use cnc::Forward;

/// The [`Predict`] trait is designed as a _**model-specific**_ interface for making
/// predictions. In the future, we may consider opening the trait up allowing for an
/// alternative implementation of the trait, but for now, it is simply implemented for all
/// implementors of the [`Forward`] trait.
///
/// **Note:** The trait is sealed, preventing external implementations, ensuring that only the
/// library can define how predictions are made. This is to maintain consistency and integrity
/// across different model implementations.
pub trait Predict<Rhs> {
    type Output;

    private!();

    fn predict(&self, input: &Rhs) -> crate::NeuralResult<Self::Output>;
}

/// The [`PredictWithConfidence`] trait is an extension of the [`Predict`] trait, providing
/// an additional method to obtain predictions along with a confidence score.
pub trait PredictWithConfidence<Rhs>: Predict<Rhs> {
    type Confidence;

    fn predict_with_confidence(
        &self,
        input: &Rhs,
    ) -> crate::NeuralResult<(Self::Output, Self::Confidence)>;
}

/*
 ************* Implementations *************
*/

use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::{Float, FromPrimitive};

impl<M, U, V> Predict<U> for M
where
    M: Forward<U, Output = V>,
{
    type Output = V;

    seal!();

    fn predict(&self, input: &U) -> crate::NeuralResult<Self::Output> {
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
    ) -> Result<(Self::Output, Self::Confidence), crate::NeuralError> {
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
