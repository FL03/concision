/*
   Appellation: perceptron <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{Biased, Linear};
use concision::prelude::{Predict, PredictError};
use nd::prelude::*;
use num::Zero;

pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }

    pub fn activate<T>(&self, args: &T) -> T
    where
        T: Clone + PartialOrd + Zero,
    {
        concision::func::relu(args.clone())
    }
}

pub struct Perceptron<F = ReLU, A = f64, K = Biased, D = Ix2>
where
    D: Dimension,
{
    module: Linear<A, K, D>,
    rho: F,
}

impl<F, A, K, D> Perceptron<F, A, K, D>
where
    D: Dimension,
{
    pub fn new(module: Linear<A, K, D>, rho: F) -> Self {
        Self { module, rho }
    }

    pub fn activate<T>(&self, args: &T) -> T
    where
        F: Fn(&T) -> T,
    {
        (self.rho)(args)
    }
}

impl<X, Y, F, A, K, D> Predict<X> for Perceptron<F, A, K, D>
where
    D: Dimension,
    F: for<'a> Fn(&'a Y) -> Y,
    Linear<A, K, D>: Predict<X, Output = Y>,
{
    type Output = Y;

    fn predict(&self, args: &X) -> Result<Self::Output, PredictError> {
        let res = self.module.predict(args)?;
        Ok(self.activate(&res))
    }
}
