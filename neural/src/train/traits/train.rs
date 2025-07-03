/*
    Appellation: train <module>
    Contrib: @FL03
*/
use crate::train::error::TrainingError;

use crate::error::ModelResult;

/// This trait defines the training process for the network
pub trait Train<X, Y> {
    type Output;

    fn train(&mut self, input: &X, target: &Y) -> ModelResult<Self::Output>;

    fn train_for(&mut self, input: &X, target: &Y, epochs: usize) -> ModelResult<Self::Output> {
        let mut output = None;

        for _ in 0..epochs {
            output = match self.train(input, target) {
                Ok(o) => Some(o),
                Err(e) => {
                    #[cfg(feature = "tracing")]
                    tracing::error!("Training failed: {e}");
                    return Err(e);
                }
            }
        }
        output.ok_or_else(|| TrainingError::TrainingFailed.into())
    }
}
