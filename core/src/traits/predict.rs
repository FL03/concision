/*
   Appellation: predict <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::error::PredictError;

pub trait Predict<T> {
    type Output;

    fn predict(&self, args: &T) -> Result<Self::Output, PredictError>;
}

pub trait Compile {
    type Dataset;

    fn compile(&mut self, dataset: &Self::Dataset);
}

pub trait Train: Compile {
    type Output;

    fn train(&mut self) -> Self::Output;
}
