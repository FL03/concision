/*
    Appellation: train <module>
    Contrib: @FL03
*/
//! This module implements various training mechanisms for neural networks. Here, implemented
//! trainers are lazily evaluated providing greater flexibility and performance.
#[doc(inline)]
pub use self::trainer::Trainer;

pub mod trainer;

pub(crate) mod impls {
    pub mod impl_trainer;
}

pub(crate) mod prelude {
    pub use super::trainer::*;
    pub use super::Train;
}


/// This trait defines the training process for the network
pub trait Train<X, Y> {
    type Output;

    fn train(&mut self, input: &X, target: &Y) -> crate::Result<Self::Output>;

    fn train_for(&mut self, input: &X, target: &Y, epochs: usize) -> crate::Result<Self::Output> {
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
        output.ok_or_else(|| crate::error::NeuralError::TrainingFailed("No output".into()))
    }
}