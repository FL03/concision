/*
   Appellation: predict <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::{Forward, PredictError};

pub trait Predict<T> {
    type Output;

    fn predict(&self, input: &T) -> Result<Self::Output, PredictError>;
}

impl<S, T, O> Predict<T> for S
where
    S: Forward<T, Output = O>,
{
    type Output = O;

    fn predict(&self, input: &T) -> Result<Self::Output, PredictError> {
        Ok(self.forward(input))
    }
}
