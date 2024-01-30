/*
   Appellation: predict <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::BoxResult;
use crate::prelude::Forward;

pub trait Predict<T> {
    type Output;

    fn predict(&self, input: &T) -> BoxResult<Self::Output>;
}

impl<S, T, O> Predict<T> for S
where
    S: Forward<T, Output = O>,
{
    type Output = O;

    fn predict(&self, input: &T) -> BoxResult<O> {
        Ok(self.forward(input))
    }
}
